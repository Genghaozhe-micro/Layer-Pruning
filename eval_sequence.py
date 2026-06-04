"""Generic evaluator: given a pruning sequence, report PPL on WikiText-2 and C4.

The V5 MDP frames layer pruning as a sequence of layer-removal actions. Once
training produces an order, this script lets you re-evaluate any custom
sequence (e.g. from a checkpoint, a baseline method, or a hand-crafted order)
without re-running the agent.

Usage:
    python eval_sequence.py --order "27,25,28,24,26,23"
    python eval_sequence.py --order-file order.json --prune-counts "1,2,4,8"

Outputs JSON to stdout (and optionally --output) with baseline PPL plus PPL
for each prefix length in --prune-counts on WT2-test and C4-validation.
"""

from __future__ import annotations

import argparse
import json
from typing import List, Sequence

import torch

from llm_pruner.data_utils import load_wikitext2, load_c4_validation
from llm_pruner.llm_wrapper import LLMWrapper


def parse_int_list(spec: str) -> List[int]:
    return [int(part) for part in spec.split(",") if part.strip()]


def load_order(args) -> List[int]:
    if args.order:
        return parse_int_list(args.order)
    if args.order_file:
        with open(args.order_file) as fh:
            payload = json.load(fh)
        if isinstance(payload, list):
            return [int(x) for x in payload]
        if isinstance(payload, dict):
            for key in ("pruning_order", "order", "layers"):
                if key in payload:
                    return [int(x) for x in payload[key]]
        raise ValueError(f"Cannot find pruning order in {args.order_file}")
    raise ValueError("Must provide --order or --order-file")


def evaluate_prefix(
    llm: LLMWrapper,
    snap,
    layers: Sequence[int],
    datasets: dict,
) -> dict:
    llm.load_snapshot(snap)
    if layers:
        llm.remove_layers(list(layers))
    out = {}
    for name, samples in datasets.items():
        out[name] = float(llm.evaluate_ppl(samples))
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--order", type=str, default=None,
                        help="Comma-separated pruning order, e.g. '27,25,28'")
    parser.add_argument("--order-file", type=str, default=None,
                        help="JSON file containing list or dict with 'pruning_order'")
    parser.add_argument("--prune-counts", type=str, default="",
                        help="Comma-separated prefix lengths to evaluate. "
                             "Empty = evaluate the full order only.")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-8B")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--wt2-num-samples", type=int, default=0,
                        help="0 = use full WikiText-2 test split")
    parser.add_argument("--c4-num-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument("--output", type=str, default=None,
                        help="Optional JSON path to write the results")
    args = parser.parse_args()

    order = load_order(args)
    if args.prune_counts.strip():
        prune_counts = sorted(set(parse_int_list(args.prune_counts)))
    else:
        prune_counts = [len(order)]

    invalid = [k for k in prune_counts if k < 0 or k > len(order)]
    if invalid:
        raise ValueError(f"prune-counts {invalid} out of range for order of length {len(order)}")

    print(f"Loading WikiText-2 test ({args.wt2_num_samples or 'all'} samples)...", file=sys.stderr)
    wt2 = load_wikitext2("test", args.wt2_num_samples, seed=args.seed)
    print(f"Loading C4 validation ({args.c4_num_samples} samples)...", file=sys.stderr)
    c4 = load_c4_validation(num_samples=args.c4_num_samples, seed=args.seed)
    datasets = {"wikitext2": wt2, "c4": c4}

    device = f"cuda:{args.gpu}"
    print(f"Loading model {args.model} on {device}...", file=sys.stderr)
    llm = LLMWrapper(args.model, device, mock=False)
    snap = llm.save_snapshot()

    results = {
        "model": args.model,
        "pruning_order": order,
        "prune_counts": prune_counts,
        "datasets": {"wikitext2": len(wt2), "c4": len(c4)},
    }

    if not args.skip_baseline:
        print("Evaluating baseline (no pruning)...", file=sys.stderr)
        results["baseline"] = evaluate_prefix(llm, snap, [], datasets)
        print(f"  baseline: {results['baseline']}", file=sys.stderr)

    eval_results = {}
    for k in prune_counts:
        layers = order[:k]
        print(f"Evaluating prefix k={k}: layers={layers}", file=sys.stderr)
        ppl = evaluate_prefix(llm, snap, layers, datasets)
        eval_results[str(k)] = {"layers": layers, **ppl}
        print(f"  k={k}: {ppl}", file=sys.stderr)

    results["evaluations"] = eval_results

    text = json.dumps(results, indent=2)
    print(text)
    if args.output:
        with open(args.output, "w") as fh:
            fh.write(text)
        print(f"Saved to {args.output}", file=sys.stderr)

    del llm
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
