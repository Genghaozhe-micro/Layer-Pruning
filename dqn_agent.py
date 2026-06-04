"""DQN Agent with Double DQN + Replay Buffer.

Key components:
  - Q-Network: MLP that outputs Q(s,a) for all actions
  - Target Network: periodically synced copy of Q-Network for stable targets
  - Replay Buffer: stores transitions for off-policy learning
  - Double DQN: use Q-Network to select action, Target to evaluate (reduces overestimation)
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from collections import deque


@dataclass
class Transition:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool
    action_mask: np.ndarray       # valid actions at current state
    next_action_mask: np.ndarray  # valid actions at next state
    budget: int = 0               # budget of the episode (for stratified buffer)


class ReplayBuffer:
    """Fixed-size experience replay buffer with uniform sampling."""

    def __init__(self, capacity: int = 50000):
        self.buffer = deque(maxlen=capacity)

    def push(self, t: Transition):
        self.buffer.append(t)

    def sample(self, batch_size: int) -> List[Transition]:
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))

    def __len__(self):
        return len(self.buffer)


class StratifiedReplayBuffer:
    """Per-budget replay buffer. Each budget gets its own FIFO queue.

    Sampling draws equally from each budget's buffer, ensuring
    small-budget data is not diluted by large-budget transitions.
    """

    def __init__(self, total_capacity: int, budgets: List[int]):
        self.budgets = sorted(budgets)
        per_budget = max(total_capacity // len(budgets), 100)
        self.buffers: Dict[int, deque] = {b: deque(maxlen=per_budget) for b in budgets}
        self._fallback = deque(maxlen=total_capacity)  # for unknown budgets

    def push(self, t: Transition):
        if t.budget in self.buffers:
            self.buffers[t.budget].append(t)
        else:
            self._fallback.append(t)

    def sample(self, batch_size: int) -> List[Transition]:
        # Collect from all non-empty buckets equally
        non_empty = [b for b in self.budgets if len(self.buffers[b]) > 0]
        if not non_empty:
            if len(self._fallback) > 0:
                return random.sample(self._fallback, min(batch_size, len(self._fallback)))
            return []
        per_bucket = max(batch_size // len(non_empty), 1)
        samples = []
        for b in non_empty:
            buf = self.buffers[b]
            k = min(per_bucket, len(buf))
            samples.extend(random.sample(list(buf), k))
        # If we got fewer than batch_size, top up from largest bucket
        while len(samples) < batch_size and non_empty:
            largest = max(non_empty, key=lambda b: len(self.buffers[b]))
            buf = self.buffers[largest]
            extra = min(batch_size - len(samples), len(buf))
            samples.extend(random.sample(list(buf), extra))
            break
        return samples[:batch_size]

    def __len__(self):
        return sum(len(b) for b in self.buffers.values()) + len(self._fallback)


class QNetwork(nn.Module):
    """Dueling DQN architecture.

    Splits into Value stream V(s) and Advantage stream A(s,a):
      Q(s,a) = V(s) + A(s,a) - mean(A(s,:))
    """

    def __init__(self, state_dim: int, num_actions: int, hidden: int = 256):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        # Value stream
        self.value_head = nn.Sequential(
            nn.Linear(hidden, hidden // 2), nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )
        # Advantage stream
        self.advantage_head = nn.Sequential(
            nn.Linear(hidden, hidden // 2), nn.ReLU(),
            nn.Linear(hidden // 2, num_actions),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Returns Q-values for all actions: (batch, num_actions)."""
        h = self.shared(state)
        v = self.value_head(h)            # (batch, 1)
        a = self.advantage_head(h)        # (batch, num_actions)
        # Dueling: Q = V + A - mean(A)
        q = v + a - a.mean(dim=-1, keepdim=True)
        return q


class TransformerQNetwork(nn.Module):
    """Transformer-based Dueling DQN.

    Treats each layer as a token (8 features each), meta features as a [CLS]
    token. Attention captures inter-layer dependencies. Per-layer Q values
    come from corresponding layer tokens; stop Q from [CLS].
    """

    def __init__(
        self,
        n_layers: int = 36,
        layer_feat_dim: int = 8,  # 7 features + 1 alive flag
        meta_dim: int = 4,
        d_model: int = 128,
        nhead: int = 4,
        num_encoder_layers: int = 2,
        dropout: float = 0.0,
        # state_dim and num_actions kept for interface compatibility
        state_dim: int = 0,
        num_actions: int = 0,
        hidden: int = 0,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.layer_feat_dim = layer_feat_dim
        self.meta_dim = meta_dim

        # Embed layer features → d_model
        self.layer_embed = nn.Linear(layer_feat_dim, d_model)
        # Embed meta features → d_model (for [CLS] token)
        self.meta_embed = nn.Linear(meta_dim, d_model)
        # Learnable position embeddings: [CLS] + N layers
        self.pos_embed = nn.Parameter(torch.randn(1, n_layers + 1, d_model) * 0.02)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        # Dueling heads
        # Value stream: from [CLS] token
        self.value_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.ReLU(),
            nn.Linear(d_model // 2, 1),
        )
        # Advantage: per-layer Q from layer tokens
        self.layer_advantage = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.ReLU(),
            nn.Linear(d_model // 2, 1),
        )
        # Advantage: stop action from [CLS]
        self.stop_advantage = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.ReLU(),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """state: (batch, state_dim) flattened. Returns (batch, n_layers+1)."""
        batch_size = state.shape[0]

        # Parse flat state → layer features + meta
        layer_end = self.n_layers * self.layer_feat_dim
        layer_feats = state[:, :layer_end].view(batch_size, self.n_layers, self.layer_feat_dim)
        meta = state[:, layer_end:]

        # Embed
        layer_tokens = self.layer_embed(layer_feats)       # (batch, n_layers, d_model)
        cls_token = self.meta_embed(meta).unsqueeze(1)     # (batch, 1, d_model)

        # [CLS, layer_0, layer_1, ..., layer_{N-1}]
        tokens = torch.cat([cls_token, layer_tokens], dim=1)  # (batch, N+1, d_model)
        tokens = tokens + self.pos_embed

        # Transformer
        encoded = self.encoder(tokens)  # (batch, N+1, d_model)

        cls_out = encoded[:, 0, :]      # (batch, d_model)
        layer_out = encoded[:, 1:, :]   # (batch, N, d_model)

        # Dueling
        value = self.value_head(cls_out)                              # (batch, 1)
        layer_adv = self.layer_advantage(layer_out).squeeze(-1)       # (batch, N)
        stop_adv = self.stop_advantage(cls_out)                       # (batch, 1)
        advantage = torch.cat([layer_adv, stop_adv], dim=1)           # (batch, N+1)

        q = value + advantage - advantage.mean(dim=-1, keepdim=True)
        return q


class DQNAgent:
    """Double Dueling DQN Agent."""

    def __init__(
        self,
        state_dim: int,
        num_actions: int,
        hidden: int = 256,
        lr: float = 1e-4,
        gamma: float = 0.99,
        buffer_capacity: int = 3000,
        batch_size: int = 64,
        target_update_freq: int = 100,  # sync target network every N updates
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay_steps: int = 3000,
        q_arch: str = "mlp",  # "mlp" or "transformer"
        # Transformer-specific params
        n_layers: int = 36,
        d_model: int = 128,
        nhead: int = 4,
        num_encoder_layers: int = 2,
        device: str = "cpu",
        budget_candidates: List[int] | None = None,
    ):
        self.state_dim = state_dim
        self.num_actions = num_actions
        self.hidden = hidden
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.device = torch.device(device)
        self.q_arch = q_arch
        self.n_layers = n_layers
        self.d_model = d_model
        self.nhead = nhead
        self.num_encoder_layers = num_encoder_layers

        # Epsilon schedule
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = epsilon_decay_steps
        self.total_steps = 0

        # Networks
        if q_arch == "transformer":
            self.q_net = TransformerQNetwork(
                n_layers=n_layers, d_model=d_model, nhead=nhead,
                num_encoder_layers=num_encoder_layers,
            )
            self.target_net = TransformerQNetwork(
                n_layers=n_layers, d_model=d_model, nhead=nhead,
                num_encoder_layers=num_encoder_layers,
            )
        else:
            self.q_net = QNetwork(state_dim, num_actions, hidden)
            self.target_net = QNetwork(state_dim, num_actions, hidden)
        self.q_net.to(self.device)
        self.target_net.to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)
        if budget_candidates:
            self.buffer = StratifiedReplayBuffer(buffer_capacity, budget_candidates)
        else:
            self.buffer = ReplayBuffer(buffer_capacity)
        self.update_count = 0

    @property
    def epsilon(self) -> float:
        """Linearly decay epsilon from start to end."""
        frac = min(1.0, self.total_steps / self.epsilon_decay_steps)
        return self.epsilon_start + frac * (self.epsilon_end - self.epsilon_start)

    def select_action(self, state: np.ndarray, action_mask: np.ndarray,
                      deterministic: bool = False) -> int:
        """Epsilon-greedy action selection with masking."""
        valid_actions = np.where(action_mask > 0)[0]
        if len(valid_actions) == 0:
            return 0  # fallback

        if not deterministic and random.random() < self.epsilon:
            return int(np.random.choice(valid_actions))

        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_net(state_t).squeeze(0).cpu().numpy()
            # Mask invalid actions with -inf
            q_values[action_mask == 0] = -np.inf
            return int(np.argmax(q_values))

    def store(self, transition: Transition):
        # Reject transitions with NaN/Inf
        if (np.all(np.isfinite(transition.state)) and
            np.all(np.isfinite(transition.next_state)) and
            np.isfinite(transition.reward)):
            self.buffer.push(transition)
        self.total_steps += 1

    def update(self) -> Optional[Dict[str, float]]:
        """One gradient step of Double DQN.

        Returns None if buffer too small, otherwise dict of metrics.
        """
        if len(self.buffer) < self.batch_size:
            return None

        batch = self.buffer.sample(self.batch_size)

        # Filter out transitions with NaN/Inf states
        clean = [t for t in batch
                 if np.all(np.isfinite(t.state)) and np.all(np.isfinite(t.next_state))
                 and np.isfinite(t.reward)]
        if len(clean) < 4:
            return None

        states = torch.FloatTensor(np.array([t.state for t in clean])).to(self.device)
        actions = torch.LongTensor([t.action for t in clean]).to(self.device)
        rewards = torch.FloatTensor([t.reward for t in clean]).to(self.device)
        next_states = torch.FloatTensor(np.array([t.next_state for t in clean])).to(self.device)
        dones = torch.FloatTensor([float(t.done) for t in clean]).to(self.device)
        next_masks = torch.FloatTensor(np.array([t.next_action_mask for t in clean])).to(self.device)

        # Current Q-values: Q(s, a)
        q_values = self.q_net(states)
        q_taken = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # Double DQN target:
        # 1. Use Q-network to SELECT best action in next state
        # 2. Use Target-network to EVALUATE that action's value
        with torch.no_grad():
            next_q = self.q_net(next_states)
            next_q[next_masks == 0] = -1e9  # mask invalid
            best_next_actions = next_q.argmax(dim=1)

            next_q_target = self.target_net(next_states)
            next_q_value = next_q_target.gather(1, best_next_actions.unsqueeze(1)).squeeze(1)

            target = rewards + self.gamma * next_q_value * (1 - dones)

        # TD error and MSE loss
        td_error = (q_taken - target).detach()
        loss = nn.MSELoss()(q_taken, target)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 10.0)
        self.optimizer.step()

        self.update_count += 1

        # Periodically sync target network
        if self.update_count % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        return {
            "loss": loss.item(),
            "q_mean": q_taken.mean().item(),
            "q_max": q_taken.max().item(),
            "target_q_mean": next_q_value.mean().item(),
            "target_mean": target.mean().item(),
            "td_error_mean": td_error.mean().item(),
            "td_error_abs_mean": td_error.abs().mean().item(),
            "td_error_max": td_error.abs().max().item(),
            "epsilon": self.epsilon,
        }

    def buffer_reward_stats(self) -> Optional[Dict[str, float]]:
        """Compute top-k% reward statistics over the replay buffer."""
        if len(self.buffer) < 10:
            return None
        if hasattr(self.buffer, "buffer"):
            transitions = list(self.buffer.buffer)
        else:
            transitions = []
            for bucket in getattr(self.buffer, "buffers", {}).values():
                transitions.extend(bucket)
            transitions.extend(getattr(self.buffer, "_fallback", []))
        rewards = np.array([t.reward for t in transitions if np.isfinite(t.reward)])
        if len(rewards) < 10:
            return None
        sorted_r = np.sort(rewards)[::-1]  # descending
        n = len(sorted_r)
        stats = {}
        for pct in [1, 5, 10]:
            k = max(1, int(n * pct / 100))
            top = sorted_r[:k]
            stats[f"buffer/top{pct}pct_mean"] = float(np.mean(top))
            stats[f"buffer/top{pct}pct_std"] = float(np.std(top))
        stats["buffer/reward_mean"] = float(np.mean(rewards))
        stats["buffer/reward_std"] = float(np.std(rewards))
        return stats

    def save(self, path: str):
        torch.save({
            "q_net": self.q_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "total_steps": self.total_steps,
            "update_count": self.update_count,
            "q_arch": self.q_arch,
            "q_config": {
                "state_dim": self.state_dim,
                "num_actions": self.num_actions,
                "hidden": self.hidden,
                "q_arch": self.q_arch,
                "n_layers": self.n_layers,
                "d_model": self.d_model,
                "nhead": self.nhead,
                "num_encoder_layers": self.num_encoder_layers,
            },
        }, path)

    def load(self, path: str):
        ckpt = torch.load(path, weights_only=True, map_location=self.device)
        self.q_net.load_state_dict(ckpt["q_net"])
        self.target_net.load_state_dict(ckpt["target_net"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.total_steps = ckpt["total_steps"]
        self.update_count = ckpt["update_count"]

    def to(self, device: str) -> "DQNAgent":
        """Move Q networks to the specified device."""
        self.device = torch.device(device)
        self.q_net.to(self.device)
        self.target_net.to(self.device)
        return self

    @staticmethod
    def detect_arch(ckpt_path: str) -> str:
        """Detect Q-network architecture from a checkpoint file."""
        ckpt = torch.load(ckpt_path, weights_only=True, map_location="cpu")
        if "q_arch" in ckpt:
            return ckpt["q_arch"]
        # Fallback: check for transformer-specific keys
        if any("layer_embed" in k for k in ckpt["q_net"]):
            return "transformer"
        return "mlp"

    @classmethod
    def from_checkpoint(cls, ckpt_path: str, state_dim: int, num_actions: int,
                        n_layers: int = 36, hidden: int = 256,
                        d_model: int = 128, nhead: int = 4,
                        num_encoder_layers: int = 2,
                        device: str = "cuda:0") -> "DQNAgent":
        """Create a DQNAgent and load weights, auto-detecting architecture."""
        ckpt = torch.load(ckpt_path, weights_only=True, map_location="cpu")
        q_config = ckpt.get("q_config", {})
        arch = q_config.get("q_arch") or ckpt.get("q_arch")
        if arch is None:
            arch = "transformer" if any("layer_embed" in k for k in ckpt["q_net"]) else "mlp"
        agent = cls(
            state_dim=q_config.get("state_dim", state_dim),
            num_actions=q_config.get("num_actions", num_actions),
            hidden=q_config.get("hidden", hidden),
            q_arch=arch,
            n_layers=q_config.get("n_layers", n_layers),
            d_model=q_config.get("d_model", d_model),
            nhead=q_config.get("nhead", nhead),
            num_encoder_layers=q_config.get("num_encoder_layers", num_encoder_layers),
            device=device,
        )
        agent.load(ckpt_path)
        return agent
