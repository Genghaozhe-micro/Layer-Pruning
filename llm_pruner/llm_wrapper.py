"""LLM Wrapper: model loading, feature extraction, layer pruning, and evaluation."""

import torch
import torch.nn as nn
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional, Set


@dataclass
class LayerFeatures:
    layer_idx: int
    cos_sim: float = 0.0
    attn_score: float = 0.0
    weight_l1: float = 0.0
    weight_l2: float = 0.0
    taylor: float = 0.0
    residual_ratio: float = 0.0
    output_norm_ratio: float = 0.0
    token_logit_kl: float = 0.0
    pred_entropy_delta: float = 0.0


class LLMWrapper:
    def __init__(self, model_name: str = "Qwen/Qwen3-8B",
                 device: str = "cuda:0", mock: bool = False,
                 mock_num_layers: int = 32):
        self.model_name = model_name
        self.device = device
        self.mock = mock
        self.model = None
        self.tokenizer = None
        self._original_num_layers = mock_num_layers
        self._removed_indices: Set[int] = set()
        self._cached_weight_norms: Dict[int, tuple] = {}
        if not mock:
            self._load_model()

    def _load_model(self):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.tokenizer.add_bos_token = False
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name, torch_dtype=torch.bfloat16, device_map=self.device,
            attn_implementation="flash_attention_2")
        self.model.eval()
        self._original_num_layers = len(self._get_layers())
        self._removed_indices = set()
        self._cached_weight_norms = {}

    def _get_layers(self) -> nn.ModuleList:
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            return self.model.model.layers
        if hasattr(self.model, "transformer") and hasattr(self.model.transformer, "h"):
            return self.model.transformer.h
        raise AttributeError("Cannot locate transformer layers.")

    def _get_final_norm(self):
        if hasattr(self.model, "model") and hasattr(self.model.model, "norm"):
            return self.model.model.norm
        if hasattr(self.model, "transformer") and hasattr(self.model.transformer, "ln_f"):
            return self.model.transformer.ln_f
        return None

    def _get_lm_head(self):
        if hasattr(self.model, "lm_head"):
            return self.model.lm_head
        if hasattr(self.model, "embed_out"):
            return self.model.embed_out
        raise AttributeError("Cannot locate LM head.")

    @staticmethod
    def _compute_choice_margin(choice_logits: torch.Tensor, answer_indices: torch.Tensor) -> torch.Tensor:
        batch_indices = torch.arange(choice_logits.size(0), device=choice_logits.device)
        correct_logits = choice_logits[batch_indices, answer_indices]
        masked_logits = choice_logits.clone()
        masked_logits[batch_indices, answer_indices] = -torch.inf
        wrong_logits = masked_logits.max(dim=-1).values
        return correct_logits - wrong_logits

    def _project_choice_logits(
        self,
        hidden: torch.Tensor,
        choice_weight: torch.Tensor,
        choice_bias: torch.Tensor | None,
        final_norm,
    ) -> torch.Tensor:
        if final_norm is not None:
            hidden = final_norm(hidden)
        hidden = hidden.float()
        logits = hidden @ choice_weight.transpose(0, 1)
        if choice_bias is not None:
            logits = logits + choice_bias
        return logits

    # ------------------------------------------------------------------
    # Feature extraction — batched, single pass, efficient attention
    # ------------------------------------------------------------------
    def compute_features(self, samples: List[Dict], seed: int = 0) -> Dict[int, LayerFeatures]:
        if self.mock:
            return self._mock_features(seed)
        return self._real_features(samples)

    def _tokenize_batch(self, samples: List[Dict]):
        """Tokenize all samples into a padded batch.
        If samples already contain 'input_ids', skip tokenization."""
        if samples and "input_ids" in samples[0]:
            import torch
            ids = [torch.LongTensor(s["input_ids"]) for s in samples]
            ids = torch.nn.utils.rnn.pad_sequence(ids, batch_first=True, padding_value=self.tokenizer.pad_token_id or 0)
            attn = (ids != (self.tokenizer.pad_token_id or 0)).long()
            from transformers import BatchEncoding
            return BatchEncoding({"input_ids": ids, "attention_mask": attn}).to(self.device)
        texts = [s.get("text", s.get("prompt", "")) for s in samples]
        return self.tokenizer(
            texts, return_tensors="pt", truncation=True,
            max_length=128, padding=True,
        ).to(self.device)

    def _real_features(self, samples: List[Dict]) -> Dict[int, LayerFeatures]:
        """Compute per-layer features: single forward pass, no backward.

        Delegates to the PPL path which computes task-agnostic features.
        """
        features, _ = self._real_features_and_ppl(samples)
        return features

    def compute_features_and_accuracy(self, samples: List[Dict], seed: int = 0) -> tuple[Dict[int, LayerFeatures], float]:
        """Compute features and accuracy in a single forward pass."""
        if self.mock:
            return self._mock_features(seed), self._mock_accuracy()
        return self._real_features_and_accuracy(samples, compute_accuracy=True)

    def compute_features_and_ppl(self, samples: List[Dict], seed: int = 0) -> tuple[Dict[int, LayerFeatures], float]:
        """Compute per-layer features and PPL in a single forward pass."""
        if self.mock:
            return self._mock_features(seed), self._mock_ppl()
        return self._real_features_and_ppl(samples)

    def _real_features_and_accuracy(self, samples: List[Dict], compute_accuracy: bool = True) -> tuple[Dict[int, LayerFeatures], float]:
        """Compute per-layer features and optionally accuracy in one forward pass.

        Hooks compute per-layer functional profile signals in one forward pass.
        Accuracy is computed from final logits.
        Weight norms (L1/L2) are cached since skip-pruning never modifies weights.
        """
        layers = self._get_layers()
        n = len(layers)
        alive = set(self.get_remaining_layer_indices())

        layer_dots = {i: [] for i in range(n)}
        layer_residual_ratios = {i: [] for i in range(n)}
        layer_output_norm_ratios = {i: [] for i in range(n)}
        layer_choice_kls = {i: [] for i in range(n)}
        layer_choice_margin_deltas = {i: [] for i in range(n)}

        hooks = []
        choice_labels = ["A", "B", "C", "D"]
        choice_tids = {
            label: self.tokenizer.encode(f" {label}", add_special_tokens=False)[-1]
            for label in choice_labels
        }
        choice_tid_tensor = torch.tensor([choice_tids[label] for label in choice_labels], device=self.device)
        answer_to_idx = {label: index for index, label in enumerate(choice_labels)}
        final_norm = self._get_final_norm()
        lm_head = self._get_lm_head()
        choice_weight = lm_head.weight[choice_tid_tensor].detach().float()
        choice_bias = None
        if getattr(lm_head, "bias", None) is not None:
            choice_bias = lm_head.bias[choice_tid_tensor].detach().float()

        last_positions = None
        batch_indices = None
        answer_choice_indices = None

        for idx in range(n):
            if idx not in alive:
                continue

            def _layer_hook(li):
                def fn(mod, inp, out):
                    hidden_in = inp[0].detach()
                    hidden_out = (out[0] if isinstance(out, tuple) else out).detach()

                    cos = nn.functional.cosine_similarity(
                        hidden_in.float(), hidden_out.float(), dim=-1).mean()
                    layer_dots[li].append(cos.item())

                    input_norm = hidden_in.float().norm(dim=-1).clamp_min(1e-8)
                    output_norm = hidden_out.float().norm(dim=-1).clamp_min(1e-8)
                    residual_norm = (hidden_out.float() - hidden_in.float()).norm(dim=-1)
                    layer_residual_ratios[li].append((residual_norm / input_norm).mean().item())
                    layer_output_norm_ratios[li].append((output_norm / input_norm).mean().item())

                    if last_positions is None or batch_indices is None or answer_choice_indices is None:
                        return

                    hidden_in_last = hidden_in[batch_indices, last_positions]
                    hidden_out_last = hidden_out[batch_indices, last_positions]
                    choice_logits_in = self._project_choice_logits(
                        hidden_in_last,
                        choice_weight,
                        choice_bias,
                        final_norm,
                    )
                    choice_logits_out = self._project_choice_logits(
                        hidden_out_last,
                        choice_weight,
                        choice_bias,
                        final_norm,
                    )

                    log_p_in = torch.log_softmax(choice_logits_in, dim=-1)
                    log_p_out = torch.log_softmax(choice_logits_out, dim=-1)
                    p_out = torch.softmax(choice_logits_out, dim=-1)
                    choice_kl = (p_out * (log_p_out - log_p_in)).sum(dim=-1).mean()
                    margin_in = self._compute_choice_margin(choice_logits_in, answer_choice_indices)
                    margin_out = self._compute_choice_margin(choice_logits_out, answer_choice_indices)
                    layer_choice_kls[li].append(choice_kl.item())
                    layer_choice_margin_deltas[li].append((margin_out - margin_in).mean().item())
                return fn
            hooks.append(layers[idx].register_forward_hook(_layer_hook(idx)))

        # Single forward pass: collect cos_sim via hooks + logits for accuracy
        self.model.eval()

        correct = 0

        with torch.no_grad():
            enc = self._tokenize_batch(samples)
            last_positions = enc["attention_mask"].sum(dim=1) - 1
            batch_indices = torch.arange(enc["input_ids"].size(0), device=enc["input_ids"].device)
            answer_choice_indices = torch.tensor(
                [answer_to_idx.get(sample.get("answer", "A"), 0) for sample in samples],
                device=enc["input_ids"].device,
                dtype=torch.long,
            )
            out = self.model(**enc)

            if compute_accuracy:
                for i, sample in enumerate(samples):
                    attn_mask = enc["attention_mask"][i]
                    last_pos = attn_mask.sum().item() - 1
                    logits = out.logits[i, int(last_pos)]
                    scores = {c: logits[tid].item() for c, tid in choice_tids.items()}
                    if max(scores, key=scores.get) == sample["answer"]:
                        correct += 1

            del out, enc

        for h in hooks:
            h.remove()

        # Compute features
        features = {}
        for idx in range(n):
            if idx not in alive:
                continue

            dot_val = float(np.mean(layer_dots[idx])) if layer_dots[idx] else 0.0
            residual_ratio = float(np.mean(layer_residual_ratios[idx])) if layer_residual_ratios[idx] else 0.0
            output_norm_ratio = float(np.mean(layer_output_norm_ratios[idx])) if layer_output_norm_ratios[idx] else 1.0
            choice_kl_val = float(np.mean(layer_choice_kls[idx])) if layer_choice_kls[idx] else 0.0
            choice_margin_delta_val = (
                float(np.mean(layer_choice_margin_deltas[idx])) if layer_choice_margin_deltas[idx] else 0.0
            )

            # Reuse cached weight norms (weights unchanged by skip-pruning)
            if idx in self._cached_weight_norms:
                w_l1, w_l2 = self._cached_weight_norms[idx]
            else:
                w_l1, w_l2 = 0.0, 0.0
                for p in layers[idx].parameters():
                    pf = p.data.float()
                    w_l1 += pf.abs().sum().item()
                    w_l2 += (pf ** 2).sum().item()
                w_l2 = w_l2 ** 0.5
                self._cached_weight_norms[idx] = (w_l1, w_l2)

            features[idx] = LayerFeatures(
                layer_idx=idx,
                cos_sim=dot_val,
                attn_score=0.0,
                weight_l1=w_l1,
                weight_l2=w_l2,
                taylor=0.0,
                residual_ratio=residual_ratio,
                output_norm_ratio=output_norm_ratio,
                token_logit_kl=choice_kl_val,
                pred_entropy_delta=choice_margin_delta_val,
            )

        torch.cuda.empty_cache()
        accuracy = correct / max(len(samples), 1) if compute_accuracy else 0.0
        return features, accuracy

    def _real_features_and_ppl(self, samples: List[Dict]) -> tuple[Dict[int, LayerFeatures], float]:
        """Compute per-layer features and PPL in one forward pass.

        Hooks collect:
          - cos_sim, residual_ratio, output_norm_ratio (representation-level)
          - token_logit_kl: KL(P_out || P_in) of the next-token distribution
            at the last non-padding position, measuring how much the layer
            changes the model's prediction.
          - pred_entropy_delta: H(P_out) - H(P_in), measuring whether the
            layer increases or decreases prediction uncertainty.
        PPL is computed from the causal-LM cross-entropy on the same batch.
        """
        layers = self._get_layers()
        n = len(layers)
        alive = set(self.get_remaining_layer_indices())

        layer_dots = {i: [] for i in range(n)}
        layer_residual_ratios = {i: [] for i in range(n)}
        layer_output_norm_ratios = {i: [] for i in range(n)}
        layer_token_kls = {i: [] for i in range(n)}
        layer_entropy_deltas = {i: [] for i in range(n)}
        layer_attn = {i: [] for i in range(n)}

        hooks = []
        final_norm = self._get_final_norm()
        lm_head = self._get_lm_head()
        lm_weight = lm_head.weight.detach().float()  # [V, H]
        lm_bias = lm_head.bias.detach().float() if getattr(lm_head, "bias", None) is not None else None

        last_positions = None
        batch_indices = None

        for idx in range(n):
            if idx not in alive:
                continue

            # Precompute attention geometry for Q/K last-token scoring
            attn_mod = getattr(layers[idx], "self_attn", None)
            head_dim = attn_mod.head_dim if attn_mod else 128
            nq = attn_mod.config.num_attention_heads if attn_mod else 32
            nkv = attn_mod.config.num_key_value_heads if attn_mod else 8
            hpk = nq // nkv

            def _layer_hook(li, am, hd, _nq, _nkv, _hpk):
                def fn(mod, inp, out):
                    hidden_in = inp[0].detach()
                    hidden_out = (out[0] if isinstance(out, tuple) else out).detach()
                    cos = nn.functional.cosine_similarity(
                        hidden_in.float(), hidden_out.float(), dim=-1,
                    ).mean()
                    layer_dots[li].append(cos.item())
                    input_norm = hidden_in.float().norm(dim=-1).clamp_min(1e-8)
                    output_norm = hidden_out.float().norm(dim=-1).clamp_min(1e-8)
                    residual_norm = (hidden_out.float() - hidden_in.float()).norm(dim=-1)
                    layer_residual_ratios[li].append((residual_norm / input_norm).mean().item())
                    layer_output_norm_ratios[li].append((output_norm / input_norm).mean().item())

                    if last_positions is None or batch_indices is None:
                        return

                    # Project last-position hidden states to full vocab logits
                    h_in = hidden_in[batch_indices, last_positions].float()
                    h_out = hidden_out[batch_indices, last_positions].float()
                    if final_norm is not None:
                        h_in = final_norm(h_in)
                        h_out = final_norm(h_out)
                    logits_in = h_in @ lm_weight.T
                    logits_out = h_out @ lm_weight.T
                    if lm_bias is not None:
                        logits_in = logits_in + lm_bias
                        logits_out = logits_out + lm_bias

                    # KL(P_out || P_in)
                    log_p_in = torch.log_softmax(logits_in, dim=-1)
                    log_p_out = torch.log_softmax(logits_out, dim=-1)
                    p_out = torch.softmax(logits_out, dim=-1)
                    kl = (p_out * (log_p_out - log_p_in)).sum(dim=-1).mean()
                    layer_token_kls[li].append(kl.item())

                    # Entropy delta: H(P_out) - H(P_in)
                    p_in = torch.softmax(logits_in, dim=-1)
                    entropy_in = -(p_in * log_p_in).sum(dim=-1)
                    entropy_out = -(p_out * log_p_out).sum(dim=-1)
                    layer_entropy_deltas[li].append((entropy_out - entropy_in).mean().item())

                    # Q/K last-token self-attention score
                    if am is not None:
                        seq_len = hidden_in.size(1)
                        q = am.q_proj(hidden_in[:, -1:, :])
                        k = am.k_proj(hidden_in)
                        q = q.view(q.size(0), 1, _nq, hd).transpose(1, 2).float()
                        k = k.view(k.size(0), seq_len, _nkv, hd).transpose(1, 2).float()
                        k = k.repeat_interleave(_hpk, dim=1)
                        scores = torch.matmul(q, k.transpose(-2, -1)) / (hd ** 0.5)
                        probs = torch.softmax(scores, dim=-1)
                        layer_attn[li].append(probs[:, :, 0, -1].mean().item())

                return fn
            hooks.append(layers[idx].register_forward_hook(
                _layer_hook(idx, attn_mod, head_dim, nq, nkv, hpk)))

        self.model.eval()
        total_nll = 0.0
        total_tokens = 0

        with torch.no_grad():
            enc = self._tokenize_batch(samples)
            last_positions = enc["attention_mask"].sum(dim=1) - 1
            batch_indices = torch.arange(enc["input_ids"].size(0), device=enc["input_ids"].device)
            out = self.model(**enc)

            # Causal LM cross-entropy for PPL
            logits = out.logits[:, :-1, :].contiguous()
            labels = enc["input_ids"][:, 1:].contiguous()
            mask = enc["attention_mask"][:, 1:].contiguous()
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
                reduction="none",
            ).view(labels.size()) * mask.float()
            total_nll = loss.sum().item()
            total_tokens = mask.sum().item()
            del out, enc

        for h in hooks:
            h.remove()

        features = {}
        for idx in range(n):
            if idx not in alive:
                continue
            dot_val = float(np.mean(layer_dots[idx])) if layer_dots[idx] else 0.0
            residual_ratio = float(np.mean(layer_residual_ratios[idx])) if layer_residual_ratios[idx] else 0.0
            output_norm_ratio = float(np.mean(layer_output_norm_ratios[idx])) if layer_output_norm_ratios[idx] else 1.0
            token_logit_kl = float(np.mean(layer_token_kls[idx])) if layer_token_kls[idx] else 0.0
            pred_entropy_delta = float(np.mean(layer_entropy_deltas[idx])) if layer_entropy_deltas[idx] else 0.0
            attn_val = float(np.mean(layer_attn[idx])) if layer_attn[idx] else 0.5

            if idx in self._cached_weight_norms:
                w_l1, w_l2 = self._cached_weight_norms[idx]
            else:
                w_l1, w_l2 = 0.0, 0.0
                for p in layers[idx].parameters():
                    pf = p.data.float()
                    w_l1 += pf.abs().sum().item()
                    w_l2 += (pf ** 2).sum().item()
                w_l2 = w_l2 ** 0.5
                self._cached_weight_norms[idx] = (w_l1, w_l2)

            features[idx] = LayerFeatures(
                layer_idx=idx,
                cos_sim=dot_val,
                attn_score=attn_val,
                weight_l1=w_l1,
                weight_l2=w_l2,
                taylor=0.0,
                residual_ratio=residual_ratio,
                output_norm_ratio=output_norm_ratio,
                token_logit_kl=token_logit_kl,
                pred_entropy_delta=pred_entropy_delta,
            )

        torch.cuda.empty_cache()
        ppl = float(np.exp(total_nll / max(total_tokens, 1)))
        return features, ppl

    def _mock_features(self, seed=0):
        rng = np.random.default_rng(seed)
        features = {}
        for idx in self.get_remaining_layer_indices():
            d = idx / max(self._original_num_layers - 1, 1)
            features[idx] = LayerFeatures(
                layer_idx=idx,
                cos_sim=float(np.clip(rng.normal(0.85 + 0.05*d, 0.08), 0, 1)),
                attn_score=float(np.clip(rng.normal(0.5 + 0.15*d, 0.12), 0, 1)),
                weight_l1=float(rng.exponential(100) + 50),
                weight_l2=float(rng.exponential(30) + 10),
                taylor=float(rng.exponential(5) + 0.1),
                residual_ratio=float(np.clip(rng.normal(0.12 + 0.05 * (1.0 - d), 0.03), 0.01, 0.5)),
                output_norm_ratio=float(np.clip(rng.normal(1.02 + 0.05 * d, 0.04), 0.7, 1.5)),
                token_logit_kl=float(np.clip(rng.exponential(0.05 + 0.1 * (1.0 - d)), 0.0, 1.0)),
                pred_entropy_delta=float(rng.normal(-0.1 * (1.0 - d), 0.05)),
            )
        return features

    # ------------------------------------------------------------------
    # Layer pruning — skip via forward hooks, architecture stays intact
    # ------------------------------------------------------------------
    def remove_layers(self, layer_indices: List[int]):
        """Mark layers as removed by replacing their forward with identity.

        Incremental: only newly added layers are patched.
        The nn.ModuleList is NEVER modified — state_dict keys stay constant.
        """
        if not layer_indices:
            return
        new_indices = [i for i in layer_indices if i not in self._removed_indices]
        self._removed_indices.update(layer_indices)
        if not self.mock and new_indices:
            self._install_skips(new_indices)

    def _install_skips(self, new_indices: List[int]):
        """Replace forward of specified layers with zero-cost identity."""
        layers = self._get_layers()
        for idx in new_indices:
            if idx >= len(layers):
                continue
            layer = layers[idx]
            if not hasattr(layer, '_original_forward'):
                layer._original_forward = layer.forward

            def _make_skip():
                def skip_forward(*args, **kwargs):
                    # Return first positional arg (hidden_states) unchanged
                    if args:
                        return args[0]
                    return kwargs.get('hidden_states')
                return skip_forward
            layer.forward = _make_skip()

    def _restore_all_layers(self):
        """Restore original forward methods for all patched layers."""
        if self.mock or self.model is None:
            return
        for layer in self._get_layers():
            if hasattr(layer, '_original_forward'):
                layer.forward = layer._original_forward
                del layer._original_forward

    def get_remaining_layer_indices(self) -> List[int]:
        return [i for i in range(self._original_num_layers) if i not in self._removed_indices]

    @property
    def num_remaining_layers(self):
        return self._original_num_layers - len(self._removed_indices)

    @property
    def original_num_layers(self):
        return self._original_num_layers

    @property
    def num_pruned(self):
        return len(self._removed_indices)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    def evaluate_accuracy(self, samples: List[Dict]) -> float:
        if self.mock:
            return self._mock_accuracy()
        correct, total = self.evaluate_accuracy_counts(samples)
        return correct / max(total, 1)

    def evaluate_accuracy_counts(self, samples: List[Dict]) -> tuple[int, int]:
        if self.mock:
            acc = self._mock_accuracy()
            total = max(len(samples), 1)
            return int(round(acc * total)), total
        return self._real_accuracy_counts(samples)

    @torch.no_grad()
    def _real_accuracy_counts(self, samples: List[Dict]) -> tuple[int, int]:
        """Micro-batched accuracy evaluation returning correct/total."""
        self.model.eval()
        choice_tids = {}
        for c in ["A", "B", "C", "D"]:
            choice_tids[c] = self.tokenizer.encode(f" {c}", add_special_tokens=False)[-1]

        EVAL_BATCH = 16
        correct = 0
        for mb_start in range(0, len(samples), EVAL_BATCH):
            mb = samples[mb_start:mb_start + EVAL_BATCH]
            enc = self._tokenize_batch(mb)
            out = self.model(**enc)
            for i, sample in enumerate(mb):
                attn_mask = enc["attention_mask"][i]
                last_pos = attn_mask.sum().item() - 1
                logits = out.logits[i, int(last_pos)]
                scores = {c: logits[tid].item() for c, tid in choice_tids.items()}
                if max(scores, key=scores.get) == sample["answer"]:
                    correct += 1
            del out, enc
        return correct, len(samples)

    # ------------------------------------------------------------------
    # PPL evaluation
    # ------------------------------------------------------------------
    def evaluate_ppl(self, samples: List[Dict]) -> float:
        """Compute perplexity on text samples."""
        if self.mock:
            return self._mock_ppl()
        total_nll, total_tokens = self.evaluate_ppl_stats(samples)
        return float(np.exp(total_nll / max(total_tokens, 1)))

    def evaluate_ppl_stats(self, samples: List[Dict]) -> tuple[float, int]:
        """Return ``(total_nll, total_tokens)`` for distributed PPL aggregation."""
        if self.mock:
            ppl = self._mock_ppl()
            total_tokens = max(len(samples), 1) * 64
            total_nll = float(np.log(max(ppl, 1e-6))) * total_tokens
            return total_nll, total_tokens
        return self._real_ppl_stats(samples)

    @torch.no_grad()
    def _real_ppl_stats(self, samples: List[Dict]) -> tuple[float, int]:
        """Standard PPL evaluation: concatenate text, tokenize once, split into
        contiguous non-overlapping chunks of ``seqlen`` tokens (no padding).

        This matches the canonical baseline PPL evaluation used by
        SLEB / FLAP / wanda / LLM-Pruner etc.
        """
        self.model.eval()
        seqlen = 4096

        # Build token sequence: either from pre-tokenized input_ids or from text
        if samples and "input_ids" in samples[0]:
            import torch as _torch
            all_ids = []
            for s in samples:
                all_ids.extend(s["input_ids"])
            testenc = _torch.LongTensor(all_ids).unsqueeze(0)
        else:
            # Concatenate all text with \n\n separator
            full_text = "\n\n".join(
                s.get("text", s.get("prompt", "")) for s in samples
            )
            testenc = self.tokenizer(full_text, return_tensors="pt").input_ids

        # 3. Split into seqlen chunks
        nsamples = testenc.numel() // seqlen
        if nsamples == 0:
            # Fallback for very short text: use whatever tokens we have
            inputs = testenc.to(self.device)
            out = self.model(inputs)
            logits = out.logits[:, :-1, :].contiguous()
            labels = inputs[:, 1:].contiguous()
            loss = nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
            )
            total_tokens = labels.numel()
            return loss.item() * total_tokens, total_tokens

        total_nll = 0.0
        total_tokens = 0
        EVAL_BATCH = 4
        for i in range(0, nsamples, EVAL_BATCH):
            j = min(i + EVAL_BATCH, nsamples)
            inputs = testenc[:, (i * seqlen):(j * seqlen)].to(self.device)
            inputs = inputs.reshape(j - i, seqlen)

            lm_logits = self.model(inputs).logits
            shift_logits = lm_logits[:, :-1, :].contiguous()
            shift_labels = inputs[:, 1:]

            loss = nn.functional.cross_entropy(
                shift_logits.reshape(-1, shift_logits.size(-1)),
                shift_labels.reshape(-1),
            )
            nll = loss.float() * seqlen * (j - i)
            total_nll += nll.item()
            total_tokens += seqlen * (j - i)
            del inputs, lm_logits

        return total_nll, total_tokens

    def _mock_accuracy(self):
        ratio = self.num_remaining_layers / self._original_num_layers
        return float(np.clip(0.65 * ratio + np.random.normal(0, 0.03), 0, 1))

    def _mock_ppl(self):
        ratio = self.num_remaining_layers / self._original_num_layers
        base_ppl = 8.0
        return float(max(base_ppl / max(ratio, 0.1) + np.random.normal(0, 0.5), 1.0))

    # ------------------------------------------------------------------
    # Snapshot / restore — near-zero cost with skip-hook pruning
    # ------------------------------------------------------------------
    def save_snapshot(self) -> dict:
        """No-op: skip-hook pruning never modifies weights, nothing to save."""
        return {"_placeholder": True}

    def load_snapshot(self, snap: dict):
        """Restore model to unpruned state: restore original forwards + clear gradients.
        No weight copying needed — forward replacement never modifies parameters."""
        self._restore_all_layers()
        self._removed_indices = set()
        if not self.mock and self.model is not None:
            self.model.zero_grad()
            self.model.eval()

    def reset(self):
        if not self.mock:
            self._load_model()
        self._removed_indices = set()
