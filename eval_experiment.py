"""Evaluate DQN checkpoints from a V5 experiment directory.

This script shares the same implementation as the training pipeline and
supports multi-GPU distributed evaluation for long test runs.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Sequence

import torch

from evaluation_pipeline import (
    evaluate_experiment,
    evaluate_experiment_multibudget,
    find_checkpoints,
    load_pipeline_datasets,
    load_pruning_plans,
    write_evaluation_artifacts,
    write_evaluation_artifacts_multibudget,
)


def _parse_ints(spec: str | Sequence[int] | None) -> list[int]:
    if spec is None:
        return []
    if isinstance(spec, str):
        return [int(part) for part in spec.split(",") if part.strip()]
    return [int(value) for value in spec]


def _coalesce(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _load_config(exp_dir: str) -> dict:
    config_path = os.path.join(exp_dir, "config.json")
    if not os.path.exists(config_path):
        return {}
    with open(config_path) as handle:
        return json.load(handle)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate V5 DQN experiment checkpoints")
    parser.add_argument("--exp-dir", type=str, required=True)
    parser.add_argument("--mode", choices=["auto", "single", "multi"], default="auto",
                        help="auto reads config.json; single uses one pruning order; multi rolls out per budget.")
    parser.add_argument("--ckpts", type=str, default=None,
                        help="Comma-separated checkpoint iters to evaluate, e.g. 50,100,final")
    parser.add_argument("--prune-counts", type=str, default=None,
                        help="Single-budget prefix lengths to evaluate. Default: experiment budget.")
    parser.add_argument("--budgets", type=str, default=None,
                        help="Multi-budget values. Default: config.json budget_candidates.")
    parser.add_argument("--cal-pool-size", type=int, default=320)
    parser.add_argument("--num-cal", type=int, default=32)
    parser.add_argument("--num-val", type=int, default=0)
    parser.add_argument("--num-test", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--cal-source", choices=["bookcorpus", "wikitext2"], default=None)
    parser.add_argument("--cal-seq-len", type=int, default=None)
    parser.add_argument("--hidden", type=int, default=None)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--max-prune-limit", type=int, default=None)
    parser.add_argument("--min-prune-budget", type=int, default=None,
                        help="Minimum layers to prune before stop is allowed.")
    parser.add_argument("--d-model", type=int, default=None)
    parser.add_argument("--nhead", type=int, default=None)
    parser.add_argument("--num-encoder-layers", type=int, default=None)
    parser.add_argument("--protected-layers", type=str, default=None,
                        help="Comma-separated protected layers. Default: config.json or '0,1'.")
    parser.add_argument("--order-device", type=str, default="cuda:0")
    parser.add_argument("--eval-gpus", type=int, default=0,
                        help="Number of GPUs to use for distributed evaluation. 0 means all visible GPUs.")
    return parser.parse_args()


def main():
    args = parse_args()
    config = _load_config(args.exp_dir)

    model_name = args.model or config.get("model", "Qwen/Qwen3-8B")
    hidden = int(_coalesce(args.hidden, config.get("hidden"), 256))
    alpha = float(_coalesce(args.alpha, config.get("alpha"), 10.0))
    d_model = int(_coalesce(args.d_model, config.get("d_model"), 128))
    nhead = int(_coalesce(args.nhead, config.get("nhead"), 4))
    num_encoder_layers = int(_coalesce(args.num_encoder_layers, config.get("num_encoder_layers"), 2))
    budget_from_config = config.get("budget")
    max_prune_limit = int(_coalesce(args.max_prune_limit, config.get("max_prune_limit"), budget_from_config, 18))
    min_prune_budget = int(_coalesce(args.min_prune_budget, config.get("min_prune_budget"), budget_from_config, max_prune_limit))
    cal_source = _coalesce(args.cal_source, config.get("cal_source"), "bookcorpus")
    cal_seq_len = int(_coalesce(args.cal_seq_len, config.get("cal_seq_len"), 128))

    protected_spec = _coalesce(args.protected_layers, config.get("protected_layers"), "0,1")
    protected_layers = _parse_ints(protected_spec)

    budgets = _parse_ints(args.budgets) or _parse_ints(config.get("budget_candidates"))
    is_multi_budget = args.mode == "multi" or (args.mode == "auto" and bool(budgets))
    if args.mode == "single":
        is_multi_budget = False

    if args.prune_counts:
        prune_counts = _parse_ints(args.prune_counts)
    else:
        prune_counts = [min_prune_budget if min_prune_budget > 0 else max_prune_limit]

    ckpt_iters = None
    if args.ckpts:
        ckpt_iters = []
        for value in args.ckpts.split(","):
            value = value.strip()
            if not value:
                continue
            ckpt_iters.append(value if value == "final" else int(value))

    ckpts = find_checkpoints(args.exp_dir, ckpt_iters)
    if not ckpts:
        raise RuntimeError(f"No checkpoints found under {args.exp_dir}")

    datasets = load_pipeline_datasets(
        cal_pool_size=args.cal_pool_size,
        num_cal=args.num_cal,
        num_val=args.num_val,
        num_test=args.num_test,
        seed=args.seed,
        model_name=model_name,
        cal_source=cal_source,
        cal_seq_len=cal_seq_len,
    )

    eval_gpu_count = args.eval_gpus or torch.cuda.device_count()
    eval_gpu_ids = list(range(min(eval_gpu_count, torch.cuda.device_count())))
    if not eval_gpu_ids:
        raise RuntimeError("No CUDA devices available for evaluation.")

    print("=" * 70)
    print(f"Experiment: {os.path.basename(args.exp_dir)}")
    print(f"Model: {model_name}")
    print(f"Checkpoints: {sorted(ckpts.keys(), key=lambda value: value if isinstance(value, int) else 10**9)}")
    print(f"Mode: {'multi-budget' if is_multi_budget else 'single-budget'}")
    print(f"Budgets: {budgets if is_multi_budget else prune_counts}")
    print(f"Max prune limit: {max_prune_limit}")
    print(f"Min prune budget: {min_prune_budget}")
    print(f"Protected layers: {protected_layers}")
    print(f"Distributed evaluation GPUs: {eval_gpu_ids}")
    print("=" * 70)

    if is_multi_budget:
        if not budgets:
            raise ValueError("Multi-budget evaluation requires --budgets or config.json budget_candidates.")
        results = evaluate_experiment_multibudget(
            model_name=model_name,
            ckpts=ckpts,
            budgets=budgets,
            datasets=datasets,
            hidden=hidden,
            alpha=alpha,
            order_device=args.order_device,
            eval_gpu_ids=eval_gpu_ids,
            experiment_name=os.path.basename(args.exp_dir),
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            protected_layers=protected_layers,
        )
        json_path, md_path = write_evaluation_artifacts_multibudget(args.exp_dir, results)
    else:
        precomputed_plans = load_pruning_plans(args.exp_dir)
        if precomputed_plans:
            sorted_plan_keys = sorted(precomputed_plans.keys(), key=lambda value: int(value) if value.isdigit() else 10**9)
            print(f"Using persisted pruning plans for checkpoints: {sorted_plan_keys}")
        results = evaluate_experiment(
            model_name=model_name,
            ckpts=ckpts,
            prune_counts=prune_counts,
            datasets=datasets,
            hidden=hidden,
            alpha=alpha,
            max_prune_limit=max_prune_limit,
            order_device=args.order_device,
            eval_gpu_ids=eval_gpu_ids,
            experiment_name=os.path.basename(args.exp_dir),
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            precomputed_plans=precomputed_plans,
            min_prune_budget=min_prune_budget,
            protected_layers=protected_layers,
        )
        json_path, md_path = write_evaluation_artifacts(args.exp_dir, results)

    print(f"Results: {json_path}")
    print(f"Report: {md_path}")


if __name__ == "__main__":
    main()
