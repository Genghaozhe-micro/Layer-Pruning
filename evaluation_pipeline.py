"""Shared evaluation pipeline for V5 DQN with stop action (PPL-based).

This module is used by both training and standalone evaluation so that
checkpoint selection, pruning-order rollout, and final reporting all go
through the same code path.
"""

from __future__ import annotations

import json
import os
import sys
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

import torch
import torch.multiprocessing as mp

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

from llm_pruner.data_utils import (
    load_calibration, load_calibration_bookcorpus,
    load_bookcorpus_validation, load_wikitext2, load_c4_validation,
)
from llm_pruner.llm_wrapper import LLMWrapper
from dqn_agent import DQNAgent
from environment import LayerEnv

_EVAL_SHUTDOWN = "SHUTDOWN"


@dataclass
class DatasetBundle:
    cal_pool: List[Dict]
    validation: List[Dict]
    test: List[Dict]
    rollout_calibration: List[Dict]
    c4_validation: List[Dict] | None = None
    bookcorpus_validation: List[Dict] | None = None


def extract_pruning_plans(val_history: Dict) -> Dict[str, Dict]:
    plans = {}
    for ckpt_key, result in val_history.items():
        pruning_order = result.get("pruning_order")
        steps = result.get("steps")
        if pruning_order is None or steps is None:
            continue
        plan = {
            "pruning_order": pruning_order,
            "steps": steps,
            "natural_stop_k": result.get("natural_stop_k", len(pruning_order)),
        }
        if "budget_plans" in result:
            plan["budget_plans"] = result["budget_plans"]
        plans[str(ckpt_key)] = plan
    return plans


def save_pruning_plans(output_dir: str, val_history: Dict) -> str:
    plan_path = os.path.join(output_dir, "pruning_plans.json")
    with open(plan_path, "w") as handle:
        json.dump(extract_pruning_plans(val_history), handle, indent=2)
    return plan_path


def load_pruning_plans(exp_dir: str) -> Dict[str, Dict]:
    plan_path = os.path.join(exp_dir, "pruning_plans.json")
    if os.path.exists(plan_path):
        with open(plan_path) as handle:
            raw = json.load(handle)
        return {str(key): value for key, value in raw.items()}

    val_history_path = os.path.join(exp_dir, "val_history.json")
    if os.path.exists(val_history_path):
        with open(val_history_path) as handle:
            val_history = json.load(handle)
        return extract_pruning_plans(val_history)

    return {}


def parse_prune_counts(spec: str | Sequence[int]) -> List[int]:
    if isinstance(spec, str):
        return [int(part) for part in spec.split(",") if part.strip()]
    return [int(value) for value in spec]


def load_pipeline_datasets(
    cal_pool_size: int,
    num_cal: int,
    num_val: int,
    num_test: int,
    seed: int,
    model_name: str = "Qwen/Qwen3-8B",
    cal_source: str = "bookcorpus",
    cal_seq_len: int = 128,
) -> DatasetBundle:
    if cal_source == "bookcorpus":
        cal_pool = load_calibration_bookcorpus(
            pool_size=cal_pool_size,
            seq_len=cal_seq_len,
            model_name=model_name,
            seed=seed,
        )
    else:
        cal_pool = load_calibration(
            pool_size=cal_pool_size,
            seq_len=cal_seq_len,
            model_name=model_name,
            seed=seed,
        )
    validation = load_wikitext2("validation", num_val, seed=seed + 1)
    test = load_wikitext2("test", num_test, seed=seed + 2)
    try:
        c4_val = load_c4_validation(num_samples=1000, seed=seed)
    except Exception as e:
        print(f"  Warning: failed to load C4 validation: {e}")
        c4_val = None
    bc_val = None
    if cal_source == "bookcorpus":
        try:
            bc_val = load_bookcorpus_validation(num_rows=10000)
        except Exception as e:
            print(f"  Warning: failed to load BookCorpus validation: {e}")
    return DatasetBundle(
        cal_pool=cal_pool,
        validation=validation,
        test=test,
        rollout_calibration=cal_pool[:num_cal],
        c4_validation=c4_val,
        bookcorpus_validation=bc_val,
    )


def get_pruning_order(agent: DQNAgent, env: LayerEnv, max_steps: int) -> tuple[List[Dict], int]:
    """Run a deterministic rollout and return (steps, natural_stop_k).

    ``natural_stop_k`` is the number of layers the agent chose to prune
    before selecting the stop action.  If the agent never explicitly
    stopped (e.g. ran out of budget), ``natural_stop_k`` equals the total
    number of prune steps collected.
    """
    state = env.reset()
    steps = []
    natural_stop_k = None
    for _ in range(max_steps + 1):  # +1 to allow a stop after max prune steps
        mask = env.get_action_mask()
        if mask.sum() == 0:
            break
        prev_ppl = env._ppl
        action = agent.select_action(state, mask, deterministic=True)
        state, reward, done, info = env.step(action)
        if info.get("is_stop"):
            natural_stop_k = len(steps)
            break
        steps.append(
            {
                "layer": info["pruned_layer"],
                "cal_ppl": round(info["ppl"], 4),
                "delta": round(info["ppl"] - prev_ppl, 4),
                "reward": round(reward, 4),
            }
        )
        if done:
            break
    if natural_stop_k is None:
        natural_stop_k = len(steps)
    return steps, natural_stop_k


def collect_checkpoint_plan(
    model_name: str,
    ckpt_path: str,
    hidden: int,
    alpha: float,
    d_model: int,
    nhead: int,
    num_encoder_layers: int,
    rollout_calibration: List[Dict],
    prune_counts: Sequence[int],
    max_prune_limit: int,
    device: str,
    min_prune_budget: int = 0,
    protected_layers: Sequence[int] | None = None,
) -> Dict:
    llm = LLMWrapper(model_name, device, mock=False)
    snap = llm.save_snapshot()
    env = LayerEnv(
        llm,
        rollout_calibration,
        alpha=alpha,
        max_prune_limit=max_prune_limit,
        min_prune_budget=min_prune_budget,
        ppl_ratio_threshold=0,
        protected_layers=list(protected_layers or []),
    )
    env.set_snapshot(snap)
    env.precompute_norms()

    agent = DQNAgent.from_checkpoint(
        ckpt_path,
        env.state_dim,
        env.num_actions,
        n_layers=env.n_layers,
        hidden=hidden,
        d_model=d_model,
        nhead=nhead,
        num_encoder_layers=num_encoder_layers,
        device=device,
    )
    agent.q_net.eval()

    steps, natural_stop_k = get_pruning_order(agent, env, max_steps=max_prune_limit)
    pruning_order = [step["layer"] for step in steps]

    del agent
    del llm
    torch.cuda.empty_cache()

    return {
        "pruning_order": pruning_order,
        "steps": steps,
        "natural_stop_k": natural_stop_k,
    }


def find_checkpoints(exp_dir: str, ckpt_iters: Sequence[int | str] | None = None) -> Dict[int | str, str]:
    ckpt_dir = os.path.join(exp_dir, "checkpoints")
    if not os.path.exists(ckpt_dir):
        ckpt_dir = exp_dir

    found = {}
    for file_name in os.listdir(ckpt_dir):
        if not file_name.endswith(".pt"):
            continue
        path = os.path.join(ckpt_dir, file_name)
        if "iter" in file_name:
            try:
                found[int(file_name.replace("dqn_iter", "").replace(".pt", ""))] = path
            except ValueError:
                continue
        elif "final" in file_name:
            found["final"] = path

    if ckpt_iters is None:
        return found

    filtered = {}
    for key in ckpt_iters:
        if key in found:
            filtered[key] = found[key]
    return filtered


def _split_shards(items: List[Dict], num_shards: int) -> List[List[Dict]]:
    if num_shards <= 1:
        return [items]
    return [items[idx::num_shards] for idx in range(num_shards)]


def _distributed_eval_worker(gpu_id: int, datasets: Dict[str, List[Dict]], task_queue, result_queue, model_name: str):
    import numpy as np
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    device = f"cuda:{gpu_id}"
    llm = LLMWrapper(model_name, device, mock=False)
    snap = llm.save_snapshot()

    result_queue.put(("READY", gpu_id))
    while True:
        task = task_queue.get()
        if task == _EVAL_SHUTDOWN:
            break

        task_id, dataset_name, layers = task
        llm.load_snapshot(snap)
        if layers:
            llm.remove_layers(list(layers))
        nll, tokens = llm.evaluate_ppl_stats(datasets[dataset_name])
        result_queue.put((task_id, dataset_name, nll, tokens))


class DistributedPPLEvaluator(AbstractContextManager):
    def __init__(self, model_name: str, datasets: Dict[str, List[Dict]], gpu_ids: Sequence[int]):
        self.model_name = model_name
        self.datasets = datasets
        self.gpu_ids = list(gpu_ids)
        self.task_queues = []
        self.result_queue = None
        self.workers = []
        self._task_id = 0

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def start(self):
        if self.workers:
            return
        if not self.gpu_ids:
            raise RuntimeError("DistributedPPLEvaluator requires at least one CUDA GPU id.")
        ctx = mp.get_context("spawn")
        self.result_queue = ctx.Queue()
        dataset_shards = [
            {name: shards[idx] for name, shards in self._build_dataset_shards().items()}
            for idx in range(len(self.gpu_ids))
        ]
        for idx, gpu_id in enumerate(self.gpu_ids):
            task_queue = ctx.Queue()
            worker = ctx.Process(
                target=_distributed_eval_worker,
                args=(gpu_id, dataset_shards[idx], task_queue, self.result_queue, self.model_name),
            )
            worker.daemon = True
            worker.start()
            self.task_queues.append(task_queue)
            self.workers.append(worker)

        ready = 0
        while ready < len(self.gpu_ids):
            message = self.result_queue.get()
            if message[0] == "READY":
                ready += 1

    def _build_dataset_shards(self) -> Dict[str, List[List[Dict]]]:
        return {name: _split_shards(samples, len(self.gpu_ids)) for name, samples in self.datasets.items()}

    def evaluate(self, dataset_name: str, layers: Sequence[int]) -> Dict:
        import numpy as np
        self._task_id += 1
        task_id = self._task_id
        active_workers = len(self.task_queues)
        for task_queue in self.task_queues:
            task_queue.put((task_id, dataset_name, list(layers)))

        total_nll = 0.0
        total_tokens = 0
        received = 0
        while received < active_workers:
            result = self.result_queue.get()
            if result[0] != task_id:
                raise RuntimeError(f"Unexpected evaluation task id: expected {task_id}, got {result[0]}")
            _, _, part_nll, part_tokens = result
            total_nll += part_nll
            total_tokens += part_tokens
            received += 1

        ppl = float(np.exp(total_nll / max(total_tokens, 1)))
        return {
            "total_nll": total_nll,
            "total_tokens": total_tokens,
            "ppl": ppl,
        }

    def close(self):
        for task_queue in self.task_queues:
            task_queue.put(_EVAL_SHUTDOWN)
        for worker in self.workers:
            worker.join(timeout=10)
        self.task_queues = []
        self.workers = []
        self.result_queue = None


def evaluate_experiment(
    *,
    model_name: str,
    ckpts: Dict[int | str, str],
    prune_counts: Sequence[int],
    datasets: DatasetBundle,
    hidden: int,
    alpha: float,
    max_prune_limit: int,
    order_device: str,
    eval_gpu_ids: Sequence[int],
    experiment_name: str,
    d_model: int = 128,
    nhead: int = 4,
    num_encoder_layers: int = 2,
    precomputed_plans: Dict[str, Dict] | None = None,
    min_prune_budget: int = 0,
    protected_layers: Sequence[int] | None = None,
) -> Dict:
    precomputed_plans = precomputed_plans or {}
    plans = {}
    sorted_ckpts = sorted(ckpts.keys(), key=lambda value: value if isinstance(value, int) else 10**9)
    for ckpt_key in sorted_ckpts:
        cached_plan = precomputed_plans.get(str(ckpt_key))
        if cached_plan and cached_plan.get("pruning_order") is not None and cached_plan.get("steps") is not None:
            plans[str(ckpt_key)] = {
                "pruning_order": list(cached_plan["pruning_order"]),
                "steps": list(cached_plan["steps"]),
                "natural_stop_k": cached_plan.get("natural_stop_k", len(cached_plan["pruning_order"])),
            }
        else:
            plans[str(ckpt_key)] = collect_checkpoint_plan(
                model_name=model_name,
                ckpt_path=ckpts[ckpt_key],
                hidden=hidden,
                alpha=alpha,
                d_model=d_model,
                nhead=nhead,
                num_encoder_layers=num_encoder_layers,
                rollout_calibration=datasets.rollout_calibration,
                prune_counts=prune_counts,
                max_prune_limit=max_prune_limit,
                device=order_device,
                min_prune_budget=min_prune_budget,
                protected_layers=protected_layers,
            )

    eval_datasets = {
        "cal_pool": datasets.cal_pool,
        "validation": datasets.validation,
        "test": datasets.test,
    }
    if datasets.c4_validation:
        eval_datasets["c4_validation"] = datasets.c4_validation
    if datasets.bookcorpus_validation:
        eval_datasets["bookcorpus_validation"] = datasets.bookcorpus_validation

    with DistributedPPLEvaluator(model_name, eval_datasets, eval_gpu_ids) as evaluator:
        baseline = {}
        for dataset_name in eval_datasets:
            baseline[dataset_name] = evaluator.evaluate(dataset_name, [])

        results = {
            "experiment": experiment_name,
            "model": model_name,
            "prune_counts": list(prune_counts),
            "baseline": baseline,
            "checkpoints": {},
        }

        for ckpt_key in sorted_ckpts:
            plan = plans[str(ckpt_key)]
            natural_stop_k = plan["natural_stop_k"]
            eval_counts = sorted(set(list(prune_counts) + ([natural_stop_k] if natural_stop_k > 0 else [])))
            ckpt_result = {
                "pruning_order": plan["pruning_order"],
                "steps": plan["steps"],
                "natural_stop_k": natural_stop_k,
                "eval": {},
            }
            for prune_count in eval_counts:
                layers = plan["pruning_order"][:prune_count]
                if len(layers) < prune_count:
                    continue
                ckpt_result["eval"][str(prune_count)] = {}
                for dataset_name in eval_datasets:
                    metric = evaluator.evaluate(dataset_name, layers)
                    metric["change"] = metric["ppl"] - baseline[dataset_name]["ppl"]
                    metric["layers"] = layers
                    ckpt_result["eval"][str(prune_count)][dataset_name] = metric

            results["checkpoints"][str(ckpt_key)] = ckpt_result

    torch.cuda.empty_cache()

    return results


def pick_best_checkpoint(results: Dict, dataset_name: str = "validation") -> Dict:
    """Pick best checkpoint based on lowest PPL at natural_stop_k."""
    best = None
    for ckpt_key, ckpt_result in results.get("checkpoints", {}).items():
        natural_stop_k = ckpt_result.get("natural_stop_k", 0)
        metric = ckpt_result.get("eval", {}).get(str(natural_stop_k), {}).get(dataset_name)
        if metric is None:
            # Fallback: lowest PPL across all prune counts
            for prune_count, eval_result in ckpt_result.get("eval", {}).items():
                m = eval_result.get(dataset_name)
                if m is None:
                    continue
                candidate = {
                    "checkpoint": ckpt_key,
                    "prune_count": int(prune_count),
                    "ppl": m["ppl"],
                    "change": m["change"],
                }
                if best is None or candidate["ppl"] < best["ppl"]:
                    best = candidate
            continue
        candidate = {
            "checkpoint": ckpt_key,
            "prune_count": natural_stop_k,
            "ppl": metric["ppl"],
            "change": metric["change"],
        }
        if best is None or candidate["ppl"] < best["ppl"]:
            best = candidate
    return best or {}


def write_evaluation_artifacts(output_dir: str, results: Dict):
    json_path = os.path.join(output_dir, "eval_results.json")
    md_path = os.path.join(output_dir, "eval_results.md")
    with open(json_path, "w") as handle:
        json.dump(results, handle, indent=2)

    best = pick_best_checkpoint(results)
    prune_counts = results["prune_counts"]
    ckpt_items = sorted(
        results["checkpoints"].items(),
        key=lambda item: int(item[0]) if item[0].isdigit() else 10**9,
    )

    with open(md_path, "w") as handle:
        handle.write(f"# Evaluation: {results['experiment']}\n\n")
        handle.write(f"- Model: {results['model']}\n")
        if best:
            handle.write(
                f"- Best checkpoint on validation: **iter {best['checkpoint']} / prune {best['prune_count']}** "
                f"(PPL={best['ppl']:.2f})\n"
            )
        handle.write("\n")

        handle.write("## Baselines\n\n")
        handle.write("| Dataset | PPL | Tokens |\n")
        handle.write("|---------|-----|--------|\n")
        for dataset_name, metric in results["baseline"].items():
            handle.write(
                f"| {dataset_name} | {metric['ppl']:.2f} | {metric['total_tokens']} |\n"
            )
        handle.write("\n")

        # Collect all prune counts that appear in any checkpoint
        all_counts = set(prune_counts)
        for _, ckpt_result in ckpt_items:
            all_counts.update(int(k) for k in ckpt_result.get("eval", {}).keys())
        sorted_counts = sorted(all_counts)

        for dataset_name in ["test", "validation", "cal_pool"]:
            handle.write(f"## {dataset_name} Results\n\n")
            header = "| Checkpoint | Stop@k | Pruning Order |"
            separator = "|------------|--------|---------------|"
            for prune_count in sorted_counts:
                header += f" {prune_count} layers |"
                separator += "----------|"
            handle.write(header + "\n")
            handle.write(separator + "\n")

            for ckpt_key, ckpt_result in ckpt_items:
                natural_k = ckpt_result.get("natural_stop_k", "?")
                row = f"| iter {ckpt_key} | {natural_k} | {ckpt_result['pruning_order']} |"
                for prune_count in sorted_counts:
                    metric = ckpt_result["eval"].get(str(prune_count), {}).get(dataset_name)
                    if metric is None:
                        row += " N/A |"
                    else:
                        row += f" {metric['ppl']:.2f} ({metric['change']:+.2f}) |"
                handle.write(row + "\n")
            handle.write("\n")

        handle.write("## Detailed Results\n\n")
        for ckpt_key, ckpt_result in ckpt_items:
            handle.write(f"### iter {ckpt_key}\n\n")
            handle.write(f"Pruning order: `{ckpt_result['pruning_order']}`\n")
            handle.write(f"Natural stop: after {ckpt_result.get('natural_stop_k', '?')} layers\n\n")
            handle.write("| Step | Layer | Cal PPL | Delta | Reward |\n")
            handle.write("|------|-------|---------|-------|--------|\n")
            for index, step in enumerate(ckpt_result["steps"], start=1):
                handle.write(
                    f"| {index} | L{step['layer']} | {step['cal_ppl']:.4f} | {step['delta']:+.4f} | {step['reward']:+.4f} |\n"
                )
            handle.write("\n")

    return json_path, md_path


# ---------------------------------------------------------------------------
# Multi-budget evaluation
# ---------------------------------------------------------------------------

def evaluate_experiment_multibudget(
    *,
    model_name: str,
    ckpts: Dict[int | str, str],
    budgets: Sequence[int],
    datasets: DatasetBundle,
    hidden: int,
    alpha: float,
    order_device: str,
    eval_gpu_ids: Sequence[int],
    experiment_name: str,
    d_model: int = 128,
    nhead: int = 4,
    num_encoder_layers: int = 2,
    protected_layers: Sequence[int] | None = None,
) -> Dict:
    """Evaluate multiple checkpoints, each with per-budget rollouts.

    Unlike ``evaluate_experiment`` which uses a single pruning order per
    checkpoint, this function runs a separate deterministic rollout for each
    (checkpoint, budget) pair so the agent can choose a budget-conditioned
    pruning strategy.
    """
    sorted_ckpts = sorted(ckpts.keys(), key=lambda v: v if isinstance(v, int) else 10**9)

    # Step 1: collect per-budget pruning orders on a single GPU
    llm = LLMWrapper(model_name, order_device, mock=False)
    snap = llm.save_snapshot()

    plans: Dict[str, Dict[int, Dict]] = {}  # ckpt_key -> {budget -> plan}
    for ckpt_key in sorted_ckpts:
        plans[str(ckpt_key)] = {}
        for budget in sorted(budgets):
            env = LayerEnv(
                llm, datasets.rollout_calibration,
                alpha=alpha, max_prune_limit=budget,
                min_prune_budget=budget, ppl_ratio_threshold=0,
                protected_layers=list(protected_layers or []),
            )
            env.set_snapshot(snap)
            env.precompute_norms()

            agent = DQNAgent.from_checkpoint(
                ckpts[ckpt_key],
                env.state_dim,
                env.num_actions,
                n_layers=env.n_layers,
                hidden=hidden,
                d_model=d_model,
                nhead=nhead,
                num_encoder_layers=num_encoder_layers,
                device=order_device,
            )
            agent.q_net.eval()

            steps, natural_stop = get_pruning_order(agent, env, max_steps=budget)
            pruning_order = [s["layer"] for s in steps]
            plans[str(ckpt_key)][budget] = {
                "pruning_order": pruning_order,
                "steps": steps,
                "natural_stop_k": natural_stop,
            }
            del agent

    del llm
    import torch
    torch.cuda.empty_cache()

    # Step 2: evaluate PPL on all requested datasets with the shared evaluator.
    eval_datasets = {
        "cal_pool": datasets.cal_pool,
        "validation": datasets.validation,
        "test": datasets.test,
    }
    if datasets.c4_validation:
        eval_datasets["c4_validation"] = datasets.c4_validation
    if datasets.bookcorpus_validation:
        eval_datasets["bookcorpus_validation"] = datasets.bookcorpus_validation

    sorted_budgets = sorted(budgets)

    with DistributedPPLEvaluator(model_name, eval_datasets, eval_gpu_ids) as evaluator:
        baseline = {}
        for dataset_name in eval_datasets:
            baseline[dataset_name] = evaluator.evaluate(dataset_name, [])

        checkpoints = {}
        for ckpt_key in [str(k) for k in sorted_ckpts]:
            ckpt_result: Dict[str, Any] = {}
            for budget in sorted_budgets:
                plan = plans[ckpt_key][budget]
                layers = plan["pruning_order"]
                budget_eval: Dict[str, Any] = {"plan": plan}
                for dataset_name in eval_datasets:
                    metric = evaluator.evaluate(dataset_name, layers)
                    metric["change"] = metric["ppl"] - baseline[dataset_name]["ppl"]
                    metric["layers"] = layers
                    budget_eval[dataset_name] = metric
                ckpt_result[str(budget)] = budget_eval
            checkpoints[ckpt_key] = ckpt_result

    torch.cuda.empty_cache()

    return {
        "experiment": experiment_name,
        "model": model_name,
        "budgets": sorted_budgets,
        "baseline": baseline,
        "checkpoints": checkpoints,
        "multi_budget": True,
    }


def pick_best_checkpoint_multibudget(
    results: Dict,
    dataset_name: str = "test",
) -> Dict:
    """Pick best checkpoint by minimizing sum of PPL across all budgets."""
    budgets = results["budgets"]
    best = None
    for ckpt_key, ckpt_result in results.get("checkpoints", {}).items():
        total = 0.0
        valid = True
        for budget in budgets:
            metric = ckpt_result.get(str(budget), {}).get(dataset_name)
            if metric is None:
                valid = False
                break
            total += metric["ppl"]
        if not valid:
            continue
        candidate = {"checkpoint": ckpt_key, "ppl_sum": total}
        if best is None or candidate["ppl_sum"] < best["ppl_sum"]:
            best = candidate
    return best or {}


def write_evaluation_artifacts_multibudget(output_dir: str, results: Dict):
    """Write eval_results.json and eval_results.md for multi-budget experiments."""
    json_path = os.path.join(output_dir, "eval_results.json")
    md_path = os.path.join(output_dir, "eval_results.md")
    with open(json_path, "w") as handle:
        json.dump(results, handle, indent=2)

    budgets = results["budgets"]
    ckpt_items = sorted(
        results["checkpoints"].items(),
        key=lambda item: int(item[0]) if item[0].isdigit() else 10**9,
    )
    best = pick_best_checkpoint_multibudget(results, "test")
    best_ckpt = best.get("checkpoint", "")

    with open(md_path, "w") as f:
        f.write(f"# Evaluation: {results['experiment']}\n\n")
        f.write(f"- Model: {results['model']}\n")
        f.write(f"- Budgets: {budgets}\n")
        if best:
            f.write(f"- Best checkpoint (min PPL sum on test): **iter {best_ckpt}** (sum={best['ppl_sum']:.2f})\n")
        f.write("\n")

        # Baselines
        f.write("## Baselines\n\n")
        f.write("| Dataset | PPL | Tokens |\n")
        f.write("|---------|-----|--------|\n")
        for ds_name, metric in results["baseline"].items():
            f.write(f"| {ds_name} | {metric['ppl']:.2f} | {metric['total_tokens']} |\n")
        f.write("\n")

        # Best checkpoint selection table
        f.write("## Best Checkpoint Selection\n\n")
        header = "| Iter |"
        sep = "|------|"
        for b in budgets:
            header += f" test@{b} |"
            sep += "--------|"
        header += " Sum | Rank |"
        sep += "-----|------|"
        f.write(header + "\n")
        f.write(sep + "\n")

        # Compute sums and rank
        iter_sums = []
        for ckpt_key, ckpt_result in ckpt_items:
            total = sum(
                ckpt_result.get(str(b), {}).get("test", {}).get("ppl", float("inf"))
                for b in budgets
            )
            iter_sums.append((ckpt_key, total))
        iter_sums.sort(key=lambda x: x[1])
        rank_map = {k: i + 1 for i, (k, _) in enumerate(iter_sums)}

        for ckpt_key, ckpt_result in ckpt_items:
            is_best = ckpt_key == best_ckpt
            prefix = "**" if is_best else ""
            row = f"| {prefix}{ckpt_key}{prefix} |"
            total = 0.0
            for b in budgets:
                ppl = ckpt_result.get(str(b), {}).get("test", {}).get("ppl")
                if ppl is not None:
                    row += f" {prefix}{ppl:.2f}{prefix} |"
                    total += ppl
                else:
                    row += " N/A |"
            row += f" {prefix}{total:.2f}{prefix} | {prefix}{rank_map[ckpt_key]}{prefix} |"
            f.write(row + "\n")
        f.write("\n")

        # Final results for best checkpoint
        f.write(f"## Final Results (iter {best_ckpt})\n\n")
        best_result = results["checkpoints"].get(best_ckpt, {})
        for ds_name in ["test", "validation", "cal_pool"]:
            f.write(f"### {ds_name}\n\n")
            f.write("| Budget | Pruning Order | PPL | Change |\n")
            f.write("|--------|---------------|-----|--------|\n")
            for b in budgets:
                plan = best_result.get(str(b), {}).get("plan", {})
                metric = best_result.get(str(b), {}).get(ds_name, {})
                order = plan.get("pruning_order", [])
                ppl = metric.get("ppl", float("nan"))
                change = metric.get("change", float("nan"))
                f.write(f"| {b} | {order} | {ppl:.2f} | {change:+.2f} |\n")
            f.write("\n")

        # Grid: Checkpoint x Budget for test and validation
        for ds_name in ["test", "validation"]:
            f.write(f"## Checkpoint x Budget Grid ({ds_name})\n\n")
            header = "| Iter |"
            sep = "|------|"
            for b in budgets:
                header += f" {b} layers |"
                sep += "----------|"
            f.write(header + "\n")
            f.write(sep + "\n")
            for ckpt_key, ckpt_result in ckpt_items:
                is_best = ckpt_key == best_ckpt
                prefix = "**" if is_best else ""
                row = f"| {prefix}{ckpt_key}{prefix} |"
                for b in budgets:
                    ppl = ckpt_result.get(str(b), {}).get(ds_name, {}).get("ppl")
                    if ppl is not None:
                        row += f" {prefix}{ppl:.2f}{prefix} |"
                    else:
                        row += " N/A |"
                f.write(row + "\n")
            f.write("\n")

        # Pruning orders table
        f.write("## Pruning Orders per Checkpoint\n\n")
        header = "| Iter |"
        sep = "|------|"
        for b in budgets:
            header += f" Budget={b} |"
            sep += "----------|"
        f.write(header + "\n")
        f.write(sep + "\n")
        for ckpt_key, ckpt_result in ckpt_items:
            is_best = ckpt_key == best_ckpt
            prefix = "**" if is_best else ""
            row = f"| {prefix}{ckpt_key}{prefix} |"
            for b in budgets:
                plan = ckpt_result.get(str(b), {}).get("plan", {})
                order = plan.get("pruning_order", [])
                row += f" {prefix}{order}{prefix} |"
            f.write(row + "\n")
        f.write("\n")

        # Detailed steps for best checkpoint
        f.write(f"## Detailed Steps (iter {best_ckpt})\n\n")
        for b in budgets:
            plan = best_result.get(str(b), {}).get("plan", {})
            steps = plan.get("steps", [])
            order = plan.get("pruning_order", [])
            f.write(f"### Budget={b}\n\n")
            f.write(f"Pruning order: `{order}`\n\n")
            f.write("| Step | Layer | Cal PPL | Delta | Reward |\n")
            f.write("|------|-------|---------|-------|--------|\n")
            for index, step in enumerate(steps, start=1):
                f.write(
                    f"| {index} | L{step['layer']} | {step['cal_ppl']:.4f} "
                    f"| {step['delta']:+.4f} | {step['reward']:+.4f} |\n"
                )
            f.write("\n")

    return json_path, md_path
