"""V5 DQN training pipeline with explicit stop action.

This entrypoint covers the full workflow:
  1. train with separate calibration/reward pools
  2. validate every N iterations on cal_pool/validation
  3. pick the best checkpoint on validation (accuracy or budget mode)
  4. run final multi-GPU evaluation on test automatically
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime

import numpy as np
import torch
import torch.multiprocessing as mp

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

from dqn_agent import DQNAgent, Transition
from environment import LayerEnv
from evaluation_pipeline import (
    evaluate_experiment,
    evaluate_experiment_multibudget,
    extract_pruning_plans,
    get_pruning_order,
    load_pipeline_datasets,
    pick_best_checkpoint_multibudget,
    save_pruning_plans,
    write_evaluation_artifacts,
    write_evaluation_artifacts_multibudget,
)

_SHUTDOWN = "SHUTDOWN"
_COLLECT = "COLLECT"
_VALIDATE = "VALIDATE"


def persistent_worker(gpu_id, task_queue, result_queue, init_args, cal_pool, validation_samples):
    import random

    if THIS_DIR not in sys.path:
        sys.path.insert(0, THIS_DIR)

    from llm_pruner.llm_wrapper import LLMWrapper

    device = f"cuda:{gpu_id}"
    np.random.seed(init_args.seed + gpu_id)
    torch.manual_seed(init_args.seed + gpu_id)
    random.seed(init_args.seed + gpu_id)

    llm = LLMWrapper(init_args.model, device, mock=False)
    snap = llm.save_snapshot()
    budget_candidates = None
    if init_args.budget_candidates:
        budget_candidates = [int(x) for x in init_args.budget_candidates.split(",")]

    env = LayerEnv(
        llm,
        cal_pool[: init_args.num_cal],
        alpha=init_args.alpha,
        max_prune_limit=init_args.max_prune_limit,
        min_prune_budget=init_args.min_prune_budget,
        ppl_clip_max=init_args.ppl_clip_max,
        ppl_penalty=init_args.ppl_penalty,
        ppl_ratio_threshold=init_args.ppl_ratio_threshold,
        normalize_reward=init_args.normalize_reward,
        budget_candidates=budget_candidates,
        sample_pool=None if init_args.fixed_cal else cal_pool,
        num_cal=init_args.num_cal,
        rng_seed=init_args.seed + gpu_id,
        protected_layers=init_args.protected_layers_list,
    )
    env.set_snapshot(snap)
    env.precompute_norms()

    agent = DQNAgent(
        state_dim=env.state_dim,
        num_actions=env.num_actions,
        hidden=init_args.hidden,
        epsilon_start=init_args.epsilon_start,
        epsilon_end=init_args.epsilon_end,
        epsilon_decay_steps=init_args.epsilon_decay_steps,
        q_arch=getattr(init_args, 'q_arch', 'mlp'),
        n_layers=env.n_layers,
        d_model=getattr(init_args, 'd_model', 128),
        nhead=getattr(init_args, 'nhead', 4),
        num_encoder_layers=getattr(init_args, 'num_encoder_layers', 2),
        device=device,
    )

    result_queue.put(("READY", gpu_id))

    while True:
        task = task_queue.get()
        if task == _SHUTDOWN:
            break

        task_type = task[0]
        if task_type == _COLLECT:
            _, q_net_sd, episode_ids, total_steps = task
            agent.q_net.load_state_dict(q_net_sd)
            agent.q_net.to(device)
            agent.q_net.eval()
            agent.total_steps = total_steps

            transitions = []
            episode_infos = []
            for ep_id in episode_ids:
                t_reset_start = time.time()
                state = env.reset()
                t_reset = time.time() - t_reset_start
                done = False
                ep_reward = 0.0
                step_times = []
                ep_stopped = False

                while not done:
                    mask = env.get_action_mask()
                    t_step_start = time.time()
                    action = agent.select_action(state, mask)
                    next_state, reward, done, info = env.step(action)
                    t_step = time.time() - t_step_start
                    next_mask = env.get_action_mask()

                    transitions.append(
                        Transition(
                            state=state,
                            action=action,
                            reward=reward,
                            next_state=next_state,
                            done=done,
                            action_mask=mask,
                            next_action_mask=next_mask,
                            budget=env.current_budget,
                        )
                    )
                    step_times.append(round(t_step, 3))
                    state = next_state
                    ep_reward += reward

                    if info.get("is_stop"):
                        ep_stopped = True

                episode_infos.append(
                    {
                        "episode_id": ep_id,
                        "gpu": gpu_id,
                        "total_pruned": info["total_pruned"],
                        "final_ppl": info["ppl"],
                        "ep_reward": ep_reward,
                        "stopped": ep_stopped,
                        "pruning_order": [r["pruned_layer"] for r in env.step_records if "pruned_layer" in r],
                        "t_reset": round(t_reset, 3),
                        "step_times": step_times,
                        "t_episode": round(sum(step_times) + t_reset, 3),
                    }
                )

            result_queue.put(("COLLECT_DONE", gpu_id, transitions, episode_infos))

        elif task_type == _VALIDATE:
            _, q_net_sd = task
            agent.q_net.load_state_dict(q_net_sd)
            agent.q_net.to(device)
            agent.q_net.eval()

            # Determine budgets and eval points
            if init_args.budget_candidates:
                val_budgets = sorted([int(x) for x in init_args.budget_candidates.split(",")])
            else:
                val_budgets = sorted(set([4, 8, 12, init_args.max_prune_limit]))

            baseline = {}
            for dataset_name, samples in (
                ("cal_pool", cal_pool),
                ("validation", validation_samples),
            ):
                llm.load_snapshot(snap)
                baseline[dataset_name] = llm.evaluate_ppl(samples)

            if init_args.budget_candidates:
                # Multi-budget: separate rollout per budget
                budget_plans = {}
                for vb in val_budgets:
                    eval_env = LayerEnv(
                        llm,
                        cal_pool[: init_args.num_cal],
                        alpha=init_args.alpha,
                        max_prune_limit=vb,
                        min_prune_budget=vb,
                        ppl_ratio_threshold=0,
                        protected_layers=init_args.protected_layers_list,
                    )
                    eval_env.set_snapshot(snap)
                    eval_env.precompute_norms()

                    steps, natural_stop_k = get_pruning_order(agent, eval_env, max_steps=vb)
                    pruning_order = [step["layer"] for step in steps]
                    budget_plans[vb] = {
                        "pruning_order": pruning_order,
                        "steps": steps,
                        "natural_stop_k": natural_stop_k,
                    }
            else:
                # Single-budget: one rollout, eval at prefix subsets
                eval_env = LayerEnv(
                    llm,
                    cal_pool[: init_args.num_cal],
                    alpha=init_args.alpha,
                    max_prune_limit=init_args.max_prune_limit,
                    min_prune_budget=init_args.max_prune_limit,
                    ppl_ratio_threshold=0,
                    protected_layers=init_args.protected_layers_list,
                )
                eval_env.set_snapshot(snap)
                eval_env.precompute_norms()

                steps, natural_stop_k = get_pruning_order(agent, eval_env, max_steps=init_args.max_prune_limit)
                full_order = [step["layer"] for step in steps]
                budget_plans = {}
                for vb in val_budgets:
                    budget_plans[vb] = {
                        "pruning_order": full_order[:vb],
                        "steps": steps[:vb],
                        "natural_stop_k": min(natural_stop_k, vb),
                    }

            main_budget = max(val_budgets)
            main_plan = budget_plans[main_budget]

            eval_results = {
                "pruning_order": main_plan["pruning_order"],
                "steps": main_plan["steps"],
                "natural_stop_k": main_plan["natural_stop_k"],
                "baseline": baseline,
                "eval": {},
                "budget_plans": {str(b): budget_plans[b] for b in val_budgets},
            }

            for vb in val_budgets:
                layers = budget_plans[vb]["pruning_order"]
                eval_results["eval"][str(vb)] = {}
                for dataset_name, samples in (
                    ("cal_pool", cal_pool),
                    ("validation", validation_samples),
                ):
                    llm.load_snapshot(snap)
                    llm.remove_layers(layers)
                    ppl = llm.evaluate_ppl(samples)
                    eval_results["eval"][str(vb)][dataset_name] = {
                        "ppl": ppl,
                        "change": ppl - baseline[dataset_name],
                    }

            result_queue.put(("VALIDATE_DONE", gpu_id, eval_results))


def parse_args():
    parser = argparse.ArgumentParser(description="V5 DQN training pipeline (stop action)")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-8B")
    parser.add_argument("--num-gpus", type=int, default=8)
    parser.add_argument("--num-iters", type=int, default=300)
    parser.add_argument("--episodes-per-iter", type=int, default=8)
    parser.add_argument("--updates-per-iter", type=int, default=50)
    parser.add_argument("--update-every", type=int, default=1,
                        help="Only run updates every N iterations. Useful when using fewer GPUs to match "
                             "the effective batch of a larger GPU setup (e.g. --update-every 2 with 4 GPUs "
                             "matches 8 GPU training).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cal-pool-size", type=int, default=128)
    parser.add_argument("--cal-seq-len", type=int, default=512,
                        help="Token length per calibration sample. BookCorpus first 10k rows are concatenated and split into chunks of this size.")
    parser.add_argument("--num-cal", type=int, default=8)
    parser.add_argument("--num-val", type=int, default=0, help="0 means full validation split")
    parser.add_argument("--num-test", type=int, default=0, help="0 means full test split")
    # --- V5 core hyperparameters ---
    parser.add_argument("--budget", type=int, default=8,
                        help="Number of layers the agent must prune (sets both max_prune_limit and min_prune_budget).")
    parser.add_argument("--alpha", type=float, default=10.0,
                        help="Scale for log-PPL-ratio reward.")
    parser.add_argument("--ppl-clip-max", type=float, default=1000000,
                        help="PPL clip ceiling. Set very large (e.g. 1e9) to disable.")
    parser.add_argument("--ppl-ratio-threshold", type=float, default=2.0,
                        help="Early-stop episode when PPL ratio (curr/prev) exceeds 1+threshold. 0 to disable.")
    parser.add_argument("--ppl-penalty", type=float, default=-30.0,
                        help="Reward penalty when PPL clip or PPL ratio threshold is triggered.")
    parser.add_argument("--normalize-reward", action="store_true",
                        help="Normalize reward by running std (Welford). Only divides by std, no mean subtraction.")
    parser.add_argument("--budget-candidates", type=str, default=None,
                        help="Comma-separated budget values for multi-budget training, e.g. '3,6,9,12,15'. "
                             "Each episode randomly samples one. Overrides --min-prune-budget and --max-prune-limit.")
    # --- DQN hyperparameters ---
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--gamma", type=float, default=0.999)
    parser.add_argument("--hidden", type=int, default=256, help="Hidden dimension for MLP Q-networks.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--buffer-capacity", type=int, default=10000)
    parser.add_argument("--target-update-freq", type=int, default=100)
    parser.add_argument("--epsilon-start", type=float, default=1.0)
    parser.add_argument("--epsilon-end", type=float, default=0.05)
    parser.add_argument("--epsilon-decay-steps", type=int, default=10000)
    parser.add_argument("--warmup-iters", type=int, default=100,
                        help="Number of iterations to collect data before starting training updates. "
                             "During warmup, episodes are collected and stored but no gradient updates are performed.")
    parser.add_argument("--q-arch", type=str, default="mlp", choices=["mlp", "transformer"],
                        help="Q-network architecture: 'mlp' (Dueling MLP) or 'transformer'.")
    parser.add_argument("--d-model", type=int, default=128, help="Transformer d_model.")
    parser.add_argument("--nhead", type=int, default=4, help="Transformer attention heads.")
    parser.add_argument("--num-encoder-layers", type=int, default=2, help="Transformer encoder layers.")
    # --- Logging / checkpointing ---
    parser.add_argument("--log-interval", type=int, default=1)
    parser.add_argument("--val-interval", type=int, default=10000)
    parser.add_argument("--ckpt-interval", type=int, default=100)
    # --- Final evaluation ---
    parser.add_argument("--final-eval-gpus", type=int, default=0, help="0 means reuse all training GPUs")
    parser.add_argument("--fixed-cal", action="store_true",
                        help="Use fixed calibration set (no random sampling per episode).")
    parser.add_argument("--cal-source", type=str, default="bookcorpus", choices=["wikitext2", "bookcorpus"],
                        help="Calibration data source: 'wikitext2' or 'bookcorpus' (default).")
    parser.add_argument("--wandb-project", type=str, default="llm-layer-pruning")
    parser.add_argument("--wandb-run", type=str, default=None)
    parser.add_argument("--exp-name", type=str, default=None)
    # --- Resume ---
    parser.add_argument("--resume-ckpt", type=str, default=None,
                        help="Path to a checkpoint .pt file to resume training from.")
    parser.add_argument("--resume-iter", type=int, default=None,
                        help="Iteration number of the resume checkpoint (to skip completed iters).")
    parser.add_argument("--wandb-id", type=str, default=None,
                        help="Existing wandb run ID to resume logging into the same run.")
    parser.add_argument("--protected-layers", type=str, default="0,1",
                        help="Comma-separated layer indices that cannot be pruned. Default: '0,1' (protect first two layers). "
                             "Set to empty string '' to disable.")
    return parser.parse_args()


def make_experiment_dir(args):
    if args.exp_name:
        exp_name = args.exp_name
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_short = args.model.split("/")[-1]
        exp_name = (
            f"{timestamp}_{model_short}_iter{args.num_iters}_budget{args.budget}"
            f"_pool{args.cal_pool_size}_cal{args.num_cal}"
        )

    exp_dir = os.path.join(os.path.dirname(__file__), "experiments", exp_name)
    ckpt_dir = os.path.join(exp_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    return exp_name, exp_dir, ckpt_dir


def save_json(path, data):
    with open(path, "w") as handle:
        json.dump(data, handle, indent=2)


def pick_best_validation_checkpoint(val_history, budget, budget_candidates=None):
    """Select best checkpoints by minimizing unweighted PPL sums.

    Returns a dict with:
      - 'best_sum_partial': best ckpt by Σ PPL of all budgets except the largest
      - 'best_sum_all': best ckpt by Σ PPL of all budgets
      - 'best' (legacy): same as 'best_sum_all' for backward compatibility
    """
    if budget_candidates:
        budgets = sorted([int(x) for x in budget_candidates.split(",")])
    else:
        budgets = sorted(set([4, 8, 12, budget]))

    partial_budgets = budgets[:-1]  # e.g. [4, 8, 12]
    all_budgets = budgets            # e.g. [4, 8, 12, 16]

    best_partial = None
    best_all = None

    for iteration_str, result in val_history.items():
        ppls = {}
        valid = True
        for b in all_budgets:
            val_metric = result.get("eval", {}).get(str(b), {}).get("validation")
            if val_metric is None:
                valid = False
                break
            ppls[b] = val_metric["ppl"]
        if not valid:
            continue

        sum_partial = sum(ppls[b] for b in partial_budgets)
        sum_all = sum(ppls[b] for b in all_budgets)
        it = int(iteration_str)

        candidate_partial = {"iteration": it, "prune_count": max(partial_budgets), "ppl": sum_partial, "budgets": partial_budgets}
        candidate_all = {"iteration": it, "prune_count": max(all_budgets), "ppl": sum_all, "budgets": all_budgets}

        if best_partial is None or sum_partial < best_partial["ppl"]:
            best_partial = candidate_partial
        if best_all is None or sum_all < best_all["ppl"]:
            best_all = candidate_all

    # Print summary
    label_partial = "+".join(str(b) for b in partial_budgets)
    label_all = "+".join(str(b) for b in all_budgets)
    if best_partial:
        print(f"  Best Σ({label_partial}): iter {best_partial['iteration']} = {best_partial['ppl']:.2f}")
    if best_all:
        print(f"  Best Σ({label_all}): iter {best_all['iteration']} = {best_all['ppl']:.2f}")

    # Return best_all as the primary (legacy compatibility)
    if best_all:
        best_all["ppl_weighted"] = best_all["ppl"]
        best_all["best_partial"] = best_partial
    return best_all


def write_validation_report(exp_dir, prune_counts, val_history):
    report_path = os.path.join(exp_dir, "val_results.md")
    sorted_items = sorted(val_history.items(), key=lambda item: int(item[0]))

    # Collect all prune counts that appear in any validation result
    all_counts = set(prune_counts)
    for _, result in sorted_items:
        all_counts.update(int(k) for k in result.get("eval", {}).keys())
    sorted_counts = sorted(all_counts)

    # Check if multi-budget (budget_plans present)
    has_budget_plans = any(
        "budget_plans" in result for _, result in sorted_items
    )

    with open(report_path, "w") as handle:
        handle.write("# Validation Results (V5 – Stop Action)\n\n")
        for dataset_name in ["validation", "cal_pool"]:
            handle.write(f"## {dataset_name}\n\n")
            if has_budget_plans:
                # Multi-budget: one column per budget showing its own order + PPL
                # Build sum column specs: consecutive subsequences starting from smallest
                sum_cols = []
                for end_idx in range(2, len(sorted_counts) + 1):
                    subset = sorted_counts[:end_idx]
                    label = "Σ(" + "+".join(str(c) for c in subset) + ")"
                    sum_cols.append((label, subset))

                header = "| Checkpoint |"
                separator = "|------------|"
                for prune_count in sorted_counts:
                    header += f" Order@{prune_count} | PPL@{prune_count} |"
                    separator += "----------|----------|"
                for label, _ in sum_cols:
                    header += f" {label} |"
                    separator += "----------|"
                handle.write(header + "\n")
                handle.write(separator + "\n")

                # Track sums for finding best
                sum_records = {label: [] for label, _ in sum_cols}

                for iteration_str, result in sorted_items:
                    row = f"| iter {iteration_str} |"
                    budget_plans = result.get("budget_plans", {})
                    ppl_values = {}
                    for prune_count in sorted_counts:
                        plan = budget_plans.get(str(prune_count), {})
                        order = plan.get("pruning_order", result.get("pruning_order", []))
                        if not plan:
                            order = result.get("pruning_order", [])[:prune_count]
                        row += f" {order} |"
                        metric = result.get("eval", {}).get(str(prune_count), {}).get(dataset_name)
                        if metric is None:
                            row += " N/A |"
                        else:
                            ppl_values[prune_count] = metric['ppl']
                            row += f" {metric['ppl']:.2f} ({metric['change']:+.2f}) |"
                    for label, subset in sum_cols:
                        if all(c in ppl_values for c in subset):
                            s = sum(ppl_values[c] for c in subset)
                            row += f" {s:.2f} |"
                            sum_records[label].append((iteration_str, s))
                        else:
                            row += " N/A |"
                    handle.write(row + "\n")

                # Write best summary
                handle.write("\n")
                for label, _ in sum_cols:
                    if sum_records[label]:
                        best_iter, best_val = min(sum_records[label], key=lambda x: x[1])
                        handle.write(f"Best {label}: **iter {best_iter}** ({best_val:.2f})\n\n")
            else:
                # Single-budget: original format
                header = "| Checkpoint | Stop@k | Pruning Order |"
                separator = "|------------|--------|---------------|"
                for prune_count in sorted_counts:
                    header += f" {prune_count} layers |"
                    separator += "----------|"
                handle.write(header + "\n")
                handle.write(separator + "\n")
                for iteration_str, result in sorted_items:
                    natural_k = result.get("natural_stop_k", "?")
                    row = f"| iter {iteration_str} | {natural_k} | {result['pruning_order']} |"
                    for prune_count in sorted_counts:
                        metric = result.get("eval", {}).get(str(prune_count), {}).get(dataset_name)
                        if metric is None:
                            row += " N/A |"
                        else:
                            row += f" {metric['ppl']:.2f} ({metric['change']:+.2f}) |"
                    handle.write(row + "\n")
            handle.write("\n")
    return report_path


def main():
    args = parse_args()

    # Derive max_prune_limit and min_prune_budget from --budget
    args.max_prune_limit = args.budget
    args.min_prune_budget = args.budget
    args.best_ckpt_mode = "budget"
    prune_counts = [args.budget]

    # Parse protected layers
    if args.protected_layers:
        args.protected_layers_list = [int(x) for x in args.protected_layers.split(",")]
    else:
        args.protected_layers_list = []

    available_gpus = min(args.num_gpus, torch.cuda.device_count())
    if available_gpus <= 0:
        raise RuntimeError("No CUDA devices available.")
    args.episodes_per_iter = available_gpus

    exp_name, exp_dir, ckpt_dir = make_experiment_dir(args)
    save_json(os.path.join(exp_dir, "config.json"), vars(args))

    print("=" * 70)
    print("V5 DQN Training Pipeline (Stop Action)")
    print(f"  Experiment: {exp_name}")
    print(f"  GPUs: {available_gpus}")
    print(f"  Iterations: {args.num_iters}")
    print(f"  Budget: {args.budget}")
    print(f"  Validation interval: {args.val_interval}")
    print(f"  Final eval GPUs: {args.final_eval_gpus or available_gpus}")
    print(f"  Warmup iters: {args.warmup_iters}")
    print("=" * 70)

    datasets = load_pipeline_datasets(
        cal_pool_size=args.cal_pool_size,
        num_cal=args.num_cal,
        num_val=args.num_val,
        num_test=args.num_test,
        seed=args.seed,
        model_name=args.model,
        cal_source=args.cal_source,
        cal_seq_len=args.cal_seq_len,
    )
    print(
        f"Data loaded: cal_pool={len(datasets.cal_pool)} "
        f"validation={len(datasets.validation)} test={len(datasets.test)}"
    )

    from llm_pruner.llm_wrapper import LLMWrapper

    temp_llm = LLMWrapper(args.model, "cuda:0", mock=False)
    n_layers = temp_llm.original_num_layers
    state_dim = n_layers * 8 + 4  # 7 per-layer features + 1 alive flag + 4 meta features
    num_actions = n_layers + 1    # N prune actions + 1 stop action
    del temp_llm
    torch.cuda.empty_cache()

    budget_candidates = None
    if args.budget_candidates:
        budget_candidates = [int(x) for x in args.budget_candidates.split(",")]

    agent = DQNAgent(
        state_dim=state_dim,
        num_actions=num_actions,
        hidden=args.hidden,
        lr=args.lr,
        gamma=args.gamma,
        buffer_capacity=args.buffer_capacity,
        batch_size=args.batch_size,
        target_update_freq=args.target_update_freq,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay_steps=args.epsilon_decay_steps,
        q_arch=args.q_arch,
        n_layers=n_layers,
        d_model=args.d_model,
        nhead=args.nhead,
        num_encoder_layers=args.num_encoder_layers,
        device="cuda:0",
        budget_candidates=budget_candidates,
    )

    # --- Resume from checkpoint ---
    start_iter = 0
    val_history = {}
    history = {
        "rewards": [],
        "ppls": [],
        "losses": [],
        "q_means": [],
        "q_maxs": [],
        "td_errors": [],
        "epsilons": [],
        "stop_ratios": [],
        "avg_pruned_layers": [],
    }
    checkpoint_paths = {}

    if args.resume_ckpt:
        agent.load(args.resume_ckpt)
        start_iter = args.resume_iter or 0
        print(f"Resumed from {args.resume_ckpt}, starting at iter {start_iter + 1}")
        # Load existing val_history and history if available
        val_hist_path = os.path.join(exp_dir, "val_history.json")
        if os.path.exists(val_hist_path):
            with open(val_hist_path) as f:
                val_history = json.load(f)
            print(f"  Loaded {len(val_history)} validation records")
        hist_path = os.path.join(exp_dir, "history.json")
        if os.path.exists(hist_path):
            with open(hist_path) as f:
                history = json.load(f)
            print(f"  Loaded history with {len(history.get('rewards', []))} entries")
        # Rebuild checkpoint_paths from existing files
        for fname in os.listdir(ckpt_dir):
            if fname.endswith(".pt") and "iter" in fname:
                try:
                    it = int(fname.replace("dqn_iter", "").replace(".pt", ""))
                    checkpoint_paths[it] = os.path.join(ckpt_dir, fname)
                except ValueError:
                    pass

    import wandb

    wandb_kwargs = {
        "project": args.wandb_project,
        "name": args.wandb_run or exp_name,
        "config": vars(args),
        "tags": ["v5", "dqn", "stop-action", "pipeline", args.model.split("/")[-1]],
    }
    if args.wandb_id:
        wandb_kwargs["id"] = args.wandb_id
        wandb_kwargs["resume"] = "must"
    elif args.resume_ckpt:
        wandb_kwargs["resume"] = "allow"
    wandb.init(**wandb_kwargs)

    mp.set_start_method("spawn", force=True)
    task_queues = [mp.Queue() for _ in range(available_gpus)]
    result_queue = mp.Queue()
    workers = []
    print(f"Launching {available_gpus} persistent workers...")
    # Use BookCorpus validation for checkpoint selection when available, else WikiText-2
    worker_validation = datasets.bookcorpus_validation if datasets.bookcorpus_validation else datasets.validation
    for gpu_id in range(available_gpus):
        worker = mp.Process(
            target=persistent_worker,
            args=(
                gpu_id,
                task_queues[gpu_id],
                result_queue,
                args,
                datasets.cal_pool,
                worker_validation,
            ),
        )
        worker.daemon = True
        worker.start()
        workers.append(worker)

    ready = 0
    while ready < available_gpus:
        message = result_queue.get()
        if message[0] == "READY":
            ready += 1
            print(f"  [GPU {message[1]}] ready")

    episode_counter = start_iter * available_gpus
    pruning_plans_path = os.path.join(exp_dir, "pruning_plans.json")

    for iteration in range(start_iter, args.num_iters):
        iter_start = time.time()
        episode_ids = list(range(episode_counter, episode_counter + available_gpus))
        episode_counter += available_gpus

        collect_q_state_dict = {name: tensor.cpu() for name, tensor in agent.q_net.state_dict().items()}
        for gpu_id in range(available_gpus):
            task_queues[gpu_id].put((_COLLECT, collect_q_state_dict, [episode_ids[gpu_id]], agent.total_steps))

        all_transitions = []
        all_infos = []
        for _ in range(available_gpus):
            message = result_queue.get()
            if message[0] != "COLLECT_DONE":
                raise RuntimeError(f"Unexpected worker message during collection: {message[0]}")
            _, _, transitions, infos = message
            all_transitions.extend(transitions)
            all_infos.extend(infos)

        collect_time = time.time() - iter_start
        for transition in all_transitions:
            agent.store(transition)

        update_start = time.time()
        iter_losses = []
        iter_td_errors = []
        last_metrics = None
        in_warmup = iteration < args.warmup_iters
        should_update = (not in_warmup) and ((iteration + 1) % args.update_every == 0)
        if should_update:
            for _ in range(args.updates_per_iter):
                metrics = agent.update()
                if metrics is None:
                    continue
                last_metrics = metrics
                if np.isfinite(metrics["loss"]):
                    iter_losses.append(metrics["loss"])
                    iter_td_errors.append(metrics["td_error_abs_mean"])
        update_time = time.time() - update_start
        total_time = time.time() - iter_start

        rewards = [info["ep_reward"] for info in all_infos]
        ppls = [info["final_ppl"] for info in all_infos]
        avg_reward = float(np.mean(rewards))
        avg_ppl = float(np.mean(ppls))
        avg_loss = float(np.mean(iter_losses)) if iter_losses else 0.0
        avg_td_error = float(np.mean(iter_td_errors)) if iter_td_errors else 0.0
        stop_ratio = sum(1 for info in all_infos if info["stopped"]) / max(len(all_infos), 1)
        avg_pruned = float(np.mean([info["total_pruned"] for info in all_infos]))

        history["rewards"].append(avg_reward)
        history["ppls"].append(avg_ppl)
        history["losses"].append(avg_loss)
        history["q_means"].append(float(last_metrics["q_mean"]) if last_metrics else 0.0)
        history["q_maxs"].append(float(last_metrics["q_max"]) if last_metrics else 0.0)
        history["td_errors"].append(avg_td_error)
        history["epsilons"].append(float(agent.epsilon))
        history["stop_ratios"].append(stop_ratio)
        history["avg_pruned_layers"].append(avg_pruned)

        log_payload = {
            "reward/mean": avg_reward,
            "reward/max": max(rewards),
            "reward/min": min(rewards),
            "reward/var": float(np.var(rewards)),
            "ppl/mean": avg_ppl,
            "ppl/min": min(ppls),
            "dqn/loss": avg_loss,
            "dqn/epsilon": agent.epsilon,
            "dqn/q_mean": last_metrics["q_mean"] if last_metrics else 0.0,
            "dqn/q_max": last_metrics["q_max"] if last_metrics else 0.0,
            "dqn/td_error_abs_mean": avg_td_error,
            "dqn/td_error_mean": last_metrics["td_error_mean"] if last_metrics else 0.0,
            "dqn/td_error_max": last_metrics["td_error_max"] if last_metrics else 0.0,
            "dqn/buffer_size": len(agent.buffer),
            "dqn/updates": agent.update_count,
            "training/iter_sec": total_time,
            "training/collect_sec": collect_time,
            "training/update_sec": update_time,
            "training/total_eps": episode_counter,
            "training/in_warmup": int(in_warmup),
            "pruning/avg_layers": avg_pruned,
            "pruning/stop_ratio": stop_ratio,
        }

        if (iteration + 1) % args.log_interval == 0 or iteration == 0:
            q_info = ""
            if last_metrics:
                q_info = (
                    f"Q_mean={last_metrics['q_mean']:.4f} "
                    f"Q_max={last_metrics['q_max']:.4f} TD={avg_td_error:.6f} "
                )
            warmup_tag = "[WARMUP] " if in_warmup else ""
            print(
                f"{warmup_tag}Iter {iteration + 1:3d}/{args.num_iters} | R={avg_reward:.3f} PPL={avg_ppl:.2f} | "
                f"Loss={avg_loss:.6f} e={agent.epsilon:.3f} {q_info}"
                f"Buf={len(agent.buffer)} Prune={avg_pruned:.1f} Stop={stop_ratio:.0%} | {total_time:.1f}s "
                f"(collect={collect_time:.1f}s update={update_time:.1f}s)",
                flush=True,
            )

        should_checkpoint = (iteration + 1) % args.ckpt_interval == 0 or (iteration + 1) == args.num_iters
        should_validate = (iteration + 1) % args.val_interval == 0 or (iteration + 1) == args.num_iters

        if should_checkpoint:
            ckpt_path = os.path.join(ckpt_dir, f"dqn_iter{iteration + 1}.pt")
            agent.save(ckpt_path)
            checkpoint_paths[iteration + 1] = ckpt_path

        if should_validate:
            if iteration + 1 not in checkpoint_paths:
                ckpt_path = os.path.join(ckpt_dir, f"dqn_iter{iteration + 1}.pt")
                agent.save(ckpt_path)
                checkpoint_paths[iteration + 1] = ckpt_path

            print(f"\n[Validation] iter {iteration + 1}", flush=True)
            validation_q_state_dict = {name: tensor.cpu() for name, tensor in agent.q_net.state_dict().items()}
            task_queues[0].put((_VALIDATE, validation_q_state_dict))
            message = result_queue.get()
            while message[0] != "VALIDATE_DONE":
                if message[0] == "COLLECT_DONE":
                    raise RuntimeError("Received stale COLLECT_DONE while waiting for validation")
                message = result_queue.get()
            _, _, validation_result = message
            val_history[str(iteration + 1)] = validation_result

            natural_k = validation_result.get("natural_stop_k", "?")

            # Merge all reported prune counts (include all eval'd budgets)
            all_eval_counts = sorted(int(k) for k in validation_result.get("eval", {}).keys())
            reported_counts = sorted(set(all_eval_counts + list(prune_counts) + ([natural_k] if isinstance(natural_k, int) and natural_k > 0 else [])))
            for prune_count in reported_counts:
                metric = validation_result.get("eval", {}).get(str(prune_count), {})
                for dataset_name, dataset_metric in metric.items():
                    log_payload[f"val/{dataset_name}_prune{prune_count}_ppl"] = dataset_metric["ppl"]
            log_payload["val/natural_stop_k"] = natural_k if isinstance(natural_k, int) else 0

            print(f"  order={validation_result['pruning_order']}  stop@{natural_k}", flush=True)
            budget_plans = validation_result.get("budget_plans", {})
            for prune_count in reported_counts:
                metric = validation_result.get("eval", {}).get(str(prune_count))
                if not metric:
                    continue
                parts = []
                for dn in ["cal_pool", "validation"]:
                    if dn in metric:
                        parts.append(f"{dn[:3]}={metric[dn]['ppl']:.2f}")
                suffix = " ← stop" if prune_count == natural_k else ""
                plan = budget_plans.get(str(prune_count), {})
                order_str = f" order={plan['pruning_order']}" if plan else ""
                print(f"  prune {prune_count}:{order_str} {' '.join(parts)}{suffix}", flush=True)

            save_json(os.path.join(exp_dir, "val_history.json"), val_history)
            pruning_plans_path = save_pruning_plans(exp_dir, val_history)
            write_validation_report(exp_dir, reported_counts, val_history)

        wandb.log(log_payload, step=iteration + 1)

    for gpu_id in range(available_gpus):
        task_queues[gpu_id].put(_SHUTDOWN)
    for worker in workers:
        worker.join(timeout=10)

    final_ckpt_path = os.path.join(ckpt_dir, "dqn_final.pt")
    agent.save(final_ckpt_path)
    save_json(os.path.join(exp_dir, "history.json"), history)
    save_json(os.path.join(exp_dir, "val_history.json"), val_history)
    pruning_plans_path = save_pruning_plans(exp_dir, val_history)

    best = pick_best_validation_checkpoint(
        val_history,
        args.budget,
        budget_candidates=args.budget_candidates,
    )
    if best is None:
        raise RuntimeError("No validation results were produced; cannot run final evaluation.")

    best_partial = best.get("best_partial")
    best_all_iter = best["iteration"]
    best_partial_iter = best_partial["iteration"] if best_partial else best_all_iter

    # Collect unique best checkpoint iters for final eval
    eval_ckpt_iters = sorted(set([best_all_iter, best_partial_iter]))
    eval_ckpts = {it: checkpoint_paths[it] for it in eval_ckpt_iters if it in checkpoint_paths}

    final_eval_gpu_count = args.final_eval_gpus or available_gpus
    eval_gpu_ids = list(range(min(final_eval_gpu_count, torch.cuda.device_count())))

    label_partial = "+".join(str(b) for b in best_partial["budgets"]) if best_partial else "?"
    label_all = "+".join(str(b) for b in best["budgets"])
    print("\n" + "=" * 70)
    print(f"Best Σ({label_partial}): iter {best_partial_iter} = {best_partial['ppl']:.2f}" if best_partial else "")
    print(f"Best Σ({label_all}): iter {best_all_iter} = {best['ppl']:.2f}")
    print(f"Final evaluation on checkpoints: {eval_ckpt_iters}")
    print(f"Running final evaluation on GPUs: {eval_gpu_ids}")
    print(f"Using persisted pruning plans: {pruning_plans_path}")

    # --- Multi-budget final evaluation ---
    budget_candidates = None
    if args.budget_candidates:
        budget_candidates = [int(x) for x in args.budget_candidates.split(",")]

    if budget_candidates:
        # Evaluate best checkpoints (partial + all) with per-budget rollouts
        final_results = evaluate_experiment_multibudget(
            model_name=args.model,
            ckpts=eval_ckpts,
            budgets=budget_candidates,
            datasets=datasets,
            hidden=args.hidden,
            alpha=args.alpha,
            d_model=args.d_model,
            nhead=args.nhead,
            num_encoder_layers=args.num_encoder_layers,
            order_device=f"cuda:{eval_gpu_ids[0]}",
            eval_gpu_ids=eval_gpu_ids,
            experiment_name=exp_name,
            protected_layers=args.protected_layers_list,
        )
        eval_json_path, eval_md_path = write_evaluation_artifacts_multibudget(exp_dir, final_results)

        final_summary = {
            "best_sum_all": {"iteration": best_all_iter, "ppl_sum": best["ppl"], "budgets": best["budgets"]},
            "best_sum_partial": {"iteration": best_partial_iter, "ppl_sum": best_partial["ppl"], "budgets": best_partial["budgets"]} if best_partial else None,
            "budget_candidates": budget_candidates,
            "pruning_plans_path": pruning_plans_path,
            "eval_results_json": eval_json_path,
            "eval_results_md": eval_md_path,
        }
        save_json(os.path.join(exp_dir, "pipeline_summary.json"), final_summary)

        wandb.summary["best_all_iter"] = best_all_iter
        wandb.summary["best_all_ppl_sum"] = best["ppl"]
        if best_partial:
            wandb.summary["best_partial_iter"] = best_partial_iter
            wandb.summary["best_partial_ppl_sum"] = best_partial["ppl"]
    else:
        # --- Single-budget final evaluation (original path) ---
        final_results = evaluate_experiment(
            model_name=args.model,
            ckpts=eval_ckpts,
            prune_counts=prune_counts,
            datasets=datasets,
            hidden=args.hidden,
            alpha=args.alpha,
            d_model=args.d_model,
            nhead=args.nhead,
            num_encoder_layers=args.num_encoder_layers,
            max_prune_limit=args.max_prune_limit,
            order_device=f"cuda:{eval_gpu_ids[0]}",
            eval_gpu_ids=eval_gpu_ids,
            experiment_name=exp_name,
            precomputed_plans=extract_pruning_plans(val_history),
            min_prune_budget=args.min_prune_budget,
            protected_layers=args.protected_layers_list,
        )

        eval_json_path, eval_md_path = write_evaluation_artifacts(exp_dir, final_results)
        final_summary = {
            "best_sum_all": {"iteration": best_all_iter, "ppl_sum": best["ppl"], "budgets": best["budgets"]},
            "best_sum_partial": {"iteration": best_partial_iter, "ppl_sum": best_partial["ppl"], "budgets": best_partial["budgets"]} if best_partial else None,
            "best_checkpoint_mode": args.best_ckpt_mode,
            "min_prune_budget": args.min_prune_budget,
            "pruning_plans_path": pruning_plans_path,
            "eval_results_json": eval_json_path,
            "eval_results_md": eval_md_path,
        }
        save_json(os.path.join(exp_dir, "pipeline_summary.json"), final_summary)

        wandb.summary["best_all_iter"] = best_all_iter
        wandb.summary["best_all_ppl_sum"] = best["ppl"]
        if best_partial:
            wandb.summary["best_partial_iter"] = best_partial_iter
            wandb.summary["best_partial_ppl_sum"] = best_partial["ppl"]

    wandb.finish()
    print(f"Validation report: {os.path.join(exp_dir, 'val_results.md')}")
    print(f"Final eval report: {eval_md_path}")
    print(f"Pipeline summary: {os.path.join(exp_dir, 'pipeline_summary.json')}")


if __name__ == "__main__":
    main()
