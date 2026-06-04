"""V5 DQN environment for layer pruning with explicit stop action (PPL-based).

State:  N*8 (per-layer features + alive) + 4 (meta including progress and budget ratio)
Action: 0..N-1 = prune layer i, N = stop
Reward: prune → -alpha * log(curr_ppl / prev_ppl)
        stop  → 0 (always)
Done:   agent chooses stop OR no alive layers left
Budget: min_prune_budget sets the minimum layers to prune before stop is allowed.
        budget > 0 → stop action is masked; budget == 0 → stop is allowed with reward=0.
"""

import math
import random
from typing import Dict, List, Tuple

import numpy as np

from llm_pruner.llm_wrapper import LLMWrapper, LayerFeatures

FEATURE_NAMES = ["cos_sim", "weight_l1", "weight_l2", "residual_ratio", "output_norm_ratio", "token_logit_kl", "pred_entropy_delta"]


class WelfordNormalizer:
    """Online running std estimator using Welford's algorithm.

    Only divides by std (no mean subtraction) to preserve reward
    sign and avoid bias with variable episode lengths.
    """

    def __init__(self, eps: float = 1e-8):
        self.count = 0
        self.mean = 0.0
        self.M2 = 0.0
        self.eps = eps

    def update(self, x: float):
        self.count += 1
        delta = x - self.mean
        self.mean += delta / self.count
        delta2 = x - self.mean
        self.M2 += delta * delta2

    @property
    def std(self) -> float:
        if self.count < 2:
            return 1.0
        return max(math.sqrt(self.M2 / self.count), self.eps)

    def normalize(self, x: float) -> float:
        return x / self.std


class LayerEnv:
    """Layer pruning environment with stop action, compatible with DQN and PPO."""

    def __init__(
        self,
        llm: LLMWrapper,
        calibration_samples: List[Dict],
        alpha: float = 10.0,
        max_prune_limit: int = 18,
        min_prune_budget: int = 0,
        ppl_clip_max: float = 1000.0,
        ppl_penalty: float = -30.0,
        ppl_ratio_threshold: float = 0.3,
        normalize_reward: bool = False,
        budget_candidates: List[int] | None = None,
        sample_pool: List[Dict] | None = None,
        num_cal: int | None = None,
        rng_seed: int = 42,
        protected_layers: List[int] | None = None,
    ):
        self.llm = llm
        self.cal_samples = calibration_samples
        self.alpha = alpha
        self.max_prune_limit = max_prune_limit
        self.min_prune_budget = min_prune_budget
        self.ppl_clip_max = ppl_clip_max
        self.ppl_penalty = ppl_penalty
        self.ppl_ratio_threshold = ppl_ratio_threshold
        self.normalize_reward = normalize_reward
        self._reward_normalizer = WelfordNormalizer() if normalize_reward else None
        self.budget_candidates = budget_candidates  # e.g. [3, 6, 9, 12, 15]
        self.protected_layers = set(protected_layers) if protected_layers else set()

        self._sample_pool = sample_pool
        self._num_cal = num_cal or len(calibration_samples)
        self._rng = random.Random(rng_seed)

        self.n_layers = llm.original_num_layers
        self.num_actions = self.n_layers + 1  # +1 for stop action
        self.stop_action = self.n_layers
        # 4 meta features: prune_progress, norm_ppl, delta_ppl, budget_ratio
        self.state_dim = self.n_layers * (len(FEATURE_NAMES) + 1) + 4

        self._features: Dict[int, LayerFeatures] = {}
        self._ppl = 0.0
        self._prev_ppl = 0.0
        self._initial_ppl = 1.0
        self._total_pruned = 0
        self._budget = self.min_prune_budget
        self._alive = np.ones(self.n_layers, dtype=np.float32)

        self._l1_min = 0.0
        self._l1_max = 1.0
        self._l2_min = 0.0
        self._l2_max = 1.0

        self._initial_state_cache = None
        self._initial_features_cache = None
        self._initial_ppl_cache = None
        self._cache_enabled = self._sample_pool is None and not self.budget_candidates

        self.cumulative_rewards: List[float] = []
        self.step_records: List[Dict] = []

    def set_snapshot(self, snap):
        self._snapshot = snap

    def precompute_norms(self):
        self.llm.load_snapshot(self._snapshot)
        feats = self.llm.compute_features(self.cal_samples, seed=0)
        l1_vals = [f.weight_l1 for f in feats.values()]
        l2_vals = [f.weight_l2 for f in feats.values()]
        self._l1_min, self._l1_max = min(l1_vals), max(l1_vals)
        self._l2_min, self._l2_max = min(l2_vals), max(l2_vals)

    def _sample_episode_sets(self):
        if self._sample_pool is not None:
            self.cal_samples = self._rng.sample(self._sample_pool, self._num_cal)

    def reset(self) -> np.ndarray:
        self.llm.load_snapshot(self._snapshot)
        self._total_pruned = 0
        # Randomize budget if candidates are provided
        if self.budget_candidates:
            # Budget-proportional weighting: larger budgets sampled more often
            # because they have larger strategy spaces and need more exploration
            weights = [float(b) for b in self.budget_candidates]
            total_w = sum(weights)
            weights = [w / total_w for w in weights]
            budget = self._rng.choices(self.budget_candidates, weights=weights, k=1)[0]
            self.max_prune_limit = budget
            self.min_prune_budget = budget
        self._budget = self.min_prune_budget
        self._alive = np.ones(self.n_layers, dtype=np.float32)
        self.cumulative_rewards = []
        self.step_records = []

        if self._cache_enabled and self._initial_state_cache is not None:
            self._features = {k: v for k, v in self._initial_features_cache.items()}
            self._ppl = self._initial_ppl_cache
            self._prev_ppl = self._ppl
            self._initial_ppl = self._ppl
            return self._initial_state_cache.copy()

        self._sample_episode_sets()
        self._features, raw_ppl = self.llm.compute_features_and_ppl(self.cal_samples, seed=0)
        self._ppl = min(raw_ppl, self.ppl_clip_max) if np.isfinite(raw_ppl) else self.ppl_clip_max
        self._prev_ppl = self._ppl
        self._initial_ppl = self._ppl
        state = self._get_state()

        if self._cache_enabled:
            self._initial_features_cache = {k: v for k, v in self._features.items()}
            self._initial_ppl_cache = self._ppl
            self._initial_state_cache = state.copy()

        return state

    def _norm_l1(self, value: float) -> float:
        value_range = self._l1_max - self._l1_min
        return (value - self._l1_min) / (value_range + 1e-8) if value_range > 0 else 0.0

    def _norm_l2(self, value: float) -> float:
        value_range = self._l2_max - self._l2_min
        return (value - self._l2_min) / (value_range + 1e-8) if value_range > 0 else 0.0

    def _get_state(self) -> np.ndarray:
        layer_feats = np.zeros((self.n_layers, len(FEATURE_NAMES) + 1), dtype=np.float32)
        for idx in range(self.n_layers):
            if self._alive[idx] > 0 and idx in self._features:
                feature = self._features[idx]
                layer_feats[idx] = [
                    feature.cos_sim,
                    self._norm_l1(feature.weight_l1),
                    self._norm_l2(feature.weight_l2),
                    feature.residual_ratio,
                    feature.output_norm_ratio,
                    feature.token_logit_kl,
                    feature.pred_entropy_delta,
                    1.0,
                ]

        ratio = self.llm.num_remaining_layers / self.llm.original_num_layers
        norm_ppl = self._initial_ppl / max(self._ppl, 1e-6)
        delta_ppl = (self._prev_ppl - self._ppl) / max(self._prev_ppl, 1.0)
        budget_ratio = self.max_prune_limit / self.n_layers
        meta = np.array(
            [
                self._total_pruned / max(self.max_prune_limit, 1),
                norm_ppl,
                delta_ppl,
                budget_ratio,
            ],
            dtype=np.float32,
        )
        return np.concatenate([layer_feats.flatten(), meta])

    def get_action_mask(self) -> np.ndarray:
        mask = np.zeros(self.num_actions, dtype=np.float32)
        if self._total_pruned < self.max_prune_limit:
            for idx in range(self.n_layers):
                if self._alive[idx] > 0 and idx not in self.protected_layers:
                    mask[idx] = 1.0
        # Stop action is only valid when budget is exhausted (budget == 0)
        if self._budget <= 0:
            mask[self.stop_action] = 1.0
        return mask

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        # --- Stop action ---
        if action == self.stop_action:
            reward = 0.0
            done = True

            cumulative_reward = (self.cumulative_rewards[-1] if self.cumulative_rewards else 0.0) + reward
            self.cumulative_rewards.append(cumulative_reward)
            self.step_records.append(
                {
                    "step": self._total_pruned,
                    "action": "stop",
                    "ppl": self._ppl,
                    "reward": reward,
                    "cumulative_reward": cumulative_reward,
                }
            )
            return self._get_state(), reward, done, self._info_stop()

        # --- Invalid prune action ---
        if action < 0 or action >= self.n_layers or self._alive[action] < 1 or action in self.protected_layers:
            return self._get_state(), -1.0, True, self._info(0)

        # --- Valid prune action ---
        self._alive[action] = 0
        self.llm.remove_layers([action])
        self._total_pruned += 1
        self._budget = max(self._budget - 1, 0)

        self._prev_ppl = self._ppl
        self._features, raw_ppl = self.llm.compute_features_and_ppl(self.cal_samples, seed=self._total_pruned)

        # PPL clip: if enabled (ppl_clip_max > 0) and raw PPL exceeds clip_max, give penalty and end episode
        ppl_clip_enabled = self.ppl_clip_max > 0
        ppl_clipped = ppl_clip_enabled and (not np.isfinite(raw_ppl) or raw_ppl >= self.ppl_clip_max)
        if ppl_clip_enabled:
            self._ppl = min(raw_ppl, self.ppl_clip_max) if np.isfinite(raw_ppl) else self.ppl_clip_max
        else:
            self._ppl = raw_ppl if np.isfinite(raw_ppl) else 1e12

        if ppl_clipped:
            reward = self.ppl_penalty
            cumulative_reward = (self.cumulative_rewards[-1] if self.cumulative_rewards else 0.0) + reward
            self.cumulative_rewards.append(cumulative_reward)
            self.step_records.append(
                {
                    "step": self._total_pruned,
                    "pruned_layer": action,
                    "ppl": self._ppl,
                    "log_ratio": float('inf'),
                    "reward": reward,
                    "cumulative_reward": cumulative_reward,
                }
            )
            return self._get_state(), reward, True, self._info(1, action)

        log_ratio = math.log(max(self._ppl, 1e-6) / max(self._prev_ppl, 1e-6))
        raw_reward = -self.alpha * log_ratio
        # Clamp positive rewards to 0 (PPL decrease after pruning is noise)
        raw_reward = min(0.0, raw_reward)

        reward = float(raw_reward)
        if not np.isfinite(reward):
            reward = -2.0

        # Welford normalization: divide by running std
        if self._reward_normalizer is not None:
            self._reward_normalizer.update(reward)
            reward = self._reward_normalizer.normalize(reward)

        cumulative_reward = (self.cumulative_rewards[-1] if self.cumulative_rewards else 0.0) + reward
        self.cumulative_rewards.append(cumulative_reward)
        self.step_records.append(
            {
                "step": self._total_pruned,
                "pruned_layer": action,
                "ppl": self._ppl,
                "log_ratio": log_ratio,
                "reward": reward,
                "cumulative_reward": cumulative_reward,
            }
        )

        # Early-stop if PPL ratio (current / previous) exceeds threshold
        ppl_ratio = self._ppl / max(self._prev_ppl, 1e-6)
        if self.ppl_ratio_threshold > 0 and ppl_ratio > (1.0 + self.ppl_ratio_threshold):
            reward = self.ppl_penalty
            done = True
            return self._get_state(), reward, done, self._info(1, action)

        # Episode does NOT end on prune; agent must explicitly choose stop.
        # Exception: if literally no layers left (degenerate case).
        done = self._alive.sum() < 1
        return self._get_state(), reward, done, self._info(1, action)

    def get_best_prefix(self) -> dict:
        if not self.cumulative_rewards:
            return {"best_k": 0, "best_reward": 0.0, "steps": []}
        best_k = int(np.argmax(self.cumulative_rewards))
        return {
            "best_k": best_k + 1,
            "best_reward": self.cumulative_rewards[best_k],
            "best_ppl": self.step_records[best_k].get("ppl", 0.0),
            "steps": self.step_records[: best_k + 1],
        }

    @property
    def current_budget(self) -> int:
        """Return the budget for the current episode."""
        return self.max_prune_limit

    def _info_stop(self):
        return {
            "total_pruned": self._total_pruned,
            "ppl": self._ppl,
            "pruned_layer": None,
            "newly_pruned": 0,
            "is_stop": True,
        }

    def _info(self, pruned: int, layer_idx: int | None = None):
        return {
            "total_pruned": self._total_pruned,
            "ppl": self._ppl,
            "pruned_layer": layer_idx,
            "newly_pruned": pruned,
            "is_stop": False,
        }
