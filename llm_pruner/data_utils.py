"""Data loading and formatting utilities (MMLU + BookCorpus)."""

import os
import random
from typing import List, Dict, Tuple, Sequence


CHOICE_LABELS = ["A", "B", "C", "D"]


# ======================================================================
# BookCorpus — plain text for PPL-based evaluation
# ======================================================================

def load_bookcorpus_disjoint_subsets(
    subset_sizes: Sequence[int],
    dataset_name: str = "bookcorpus",
    seed: int = 42,
) -> List[List[Dict]]:
    """Load multiple disjoint subsets from BookCorpus for PPL-based evaluation.

    Each sample: ``{"text": str}``.
    """
    items = _load_raw_bookcorpus(dataset_name=dataset_name, seed=seed)
    total_needed = sum(max(0, s) for s in subset_sizes)
    if total_needed > len(items):
        raise ValueError(
            f"Requested {total_needed} BookCorpus samples but only {len(items)} available."
        )
    subsets: List[List[Dict]] = []
    start = 0
    for size in subset_sizes:
        end = start + max(0, size)
        subsets.append(items[start:end])
        start = end
    return subsets


def load_bookcorpus_split(
    num_samples: int = 320,
    dataset_name: str = "bookcorpus",
    seed: int = 42,
    offset: int = 0,
) -> List[Dict]:
    """Load a contiguous slice of shuffled BookCorpus.

    Use *offset* to get a non-overlapping slice (e.g. for validation vs test).
    ``num_samples=0`` loads everything after *offset*.
    """
    items = _load_raw_bookcorpus(dataset_name=dataset_name, seed=seed)
    items = items[offset:]
    if num_samples > 0:
        items = items[:num_samples]
    return items


def _load_raw_bookcorpus(dataset_name: str, seed: int) -> List[Dict]:
    from datasets import load_dataset
    try:
        ds = load_dataset(dataset_name, split="train", revision="refs/convert/parquet")
    except Exception as e:
        raise RuntimeError(
            f"Failed to load BookCorpus ({dataset_name}): {e}\n"
            "Ensure datasets<4.0 is installed: pip install 'datasets>=3.0,<4.0'"
        ) from e

    # Random-sample a manageable subset instead of iterating all 74M rows
    rng = random.Random(seed)
    total = len(ds)
    CAP = 100000  # more than enough for any downstream use
    sample_size = min(CAP * 2, total)  # oversample to account for short-text filtering
    sampled_indices = rng.sample(range(total), sample_size)

    items: List[Dict] = []
    for idx in sampled_indices:
        text = ds[idx]["text"]
        if len(text) > 30:
            items.append({"text": text})
            if len(items) >= CAP:
                break

    # Shuffle the final list for consistent downstream ordering
    rng2 = random.Random(seed)
    rng2.shuffle(items)
    print(f"  Loaded BookCorpus: {len(items)} samples (from {total} total)")
    return items

def _generate_synthetic_bookcorpus(n: int, seed: int = 42) -> List[Dict]:
    """Synthetic plain-text samples for offline testing."""
    rng = random.Random(seed)
    words = ["the", "a", "of", "and", "to", "in", "is", "it", "that", "was",
             "for", "on", "are", "with", "as", "at", "be", "this", "have", "from"]
    items = []
    for _ in range(n):
        length = rng.randint(20, 80)
        text = " ".join(rng.choice(words) for _ in range(length))
        items.append({"text": text})
    return items


def load_calibration(
    pool_size: int = 320,
    seq_len: int = 128,
    model_name: str = "Qwen/Qwen3-8B",
    seed: int = 42,
) -> List[Dict]:
    """Load calibration samples from WikiText-2 train split.

    Process:
      1. Load WikiText-2 train split.
      2. Randomly sample individual paragraphs with >= ``seq_len`` tokens.
      3. From each selected paragraph, randomly crop ``seq_len`` contiguous
         tokens, then decode back to text.

    Returns:
        List of ``{"text": str, "input_ids": List[int]}`` dicts where each
        text corresponds to exactly ``seq_len`` tokens.
    """
    from datasets import load_dataset
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")

    rng = random.Random(seed)
    total = len(ds)
    history: set = set()
    items: List[Dict] = []

    for _ in range(pool_size):
        while True:
            i = rng.randint(0, total - 1)
            if i in history:
                continue
            text = ds[i]["text"]
            if len(text.strip()) < 30:
                continue
            tokenized = tokenizer(
                text, return_tensors="pt", add_special_tokens=True,
            )
            if tokenized.input_ids.shape[1] >= seq_len:
                history.add(i)
                break
        j = rng.randint(0, tokenized.input_ids.shape[1] - seq_len)
        chunk_ids = tokenized.input_ids[0, j:j + seq_len].tolist()
        items.append({"input_ids": chunk_ids})

    print(f"  Calibration pool: {len(items)} samples, each {seq_len} tokens "
          f"(from WikiText-2 train, {total} rows)")
    return items


_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".data_cache")


def _cache_path(name: str) -> str:
    os.makedirs(_CACHE_DIR, exist_ok=True)
    return os.path.join(_CACHE_DIR, name)


def _load_or_build(cache_name: str, build_fn):
    """Load from JSON cache if available, otherwise build and save."""
    import json as _json
    path = _cache_path(cache_name)
    if os.path.exists(path):
        with open(path) as f:
            items = _json.load(f)
        print(f"  Loaded cached {cache_name}: {len(items)} samples")
        return items
    items = build_fn()
    with open(path, "w") as f:
        _json.dump(items, f)
    print(f"  Built and cached {cache_name}: {len(items)} samples")
    return items


def load_calibration_bookcorpus(
    pool_size: int = 64,
    seq_len: int = 512,
    model_name: str = "Qwen/Qwen3-8B",
    seed: int = 42,
    sample_offset: int = 10000,
) -> List[Dict]:
    """Load calibration samples from BookCorpus (with disk cache).

    Concatenates the first ``sample_offset`` rows of BookCorpus in order,
    tokenizes as one long sequence, then splits into non-overlapping chunks
    of ``seq_len`` tokens. A random subset of ``pool_size`` chunks is returned.

    Returns:
        List of ``{"input_ids": List[int]}`` dicts, each with exactly ``seq_len`` tokens.
    """
    model_short = model_name.split("/")[-1]
    cache_name = f"cal_bookcorpus_p{pool_size}_s{seq_len}_o{sample_offset}_seed{seed}_{model_short}.json"

    def build():
        from datasets import load_dataset
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        ds = load_dataset("bookcorpus", split="train", revision="refs/convert/parquet")

        # Concatenate first sample_offset rows in order
        texts = [ds[i]["text"] for i in range(min(sample_offset, len(ds))) if ds[i]["text"].strip()]
        full_text = "\n\n".join(texts)
        all_ids = tokenizer(full_text, add_special_tokens=False).input_ids

        # Split into non-overlapping chunks of seq_len
        nchunks = len(all_ids) // seq_len
        chunks = [all_ids[i * seq_len:(i + 1) * seq_len] for i in range(nchunks)]

        # Randomly sample pool_size chunks
        rng = random.Random(seed)
        if len(chunks) <= pool_size:
            selected = chunks
        else:
            selected = rng.sample(chunks, pool_size)

        items = [{"input_ids": chunk} for chunk in selected]
        print(f"  BookCorpus calibration: {len(all_ids)} total tokens → {nchunks} chunks of {seq_len} → selected {len(items)}")
        return items

    return _load_or_build(cache_name, build)


def load_bookcorpus_validation(
    num_rows: int = 10000,
) -> List[Dict]:
    """Load first ``num_rows`` rows of BookCorpus as validation set (with disk cache).

    The rows are kept in original order (sequential text) so that
    concatenation preserves narrative coherence.  This should be
    evaluated on a single GPU to avoid shard-induced context breaks.

    Returns:
        List of ``{"text": str}`` dicts.
    """
    cache_name = f"val_bookcorpus_{num_rows}.json"

    def build():
        from datasets import load_dataset
        ds = load_dataset("bookcorpus", split="train", revision="refs/convert/parquet")
        items = [{"text": ds[i]["text"]} for i in range(min(num_rows, len(ds)))
                 if len(ds[i]["text"].strip()) > 0]
        return items

    return _load_or_build(cache_name, build)


# ======================================================================
# WikiText-2 — standard LM benchmark for validation / test
# ======================================================================

def load_wikitext2(
    split: str = "validation",
    num_samples: int = 0,
    seed: int = 42,
) -> List[Dict]:
    """Load WikiText-2 as plain-text samples for PPL evaluation.

    Args:
        split: 'validation' or 'test'.
        num_samples: 0 = load all paragraphs in the split.
        seed: random seed for shuffling before truncation.

    Returns:
        List of ``{"text": str}`` dicts.
    """
    try:
        from datasets import load_dataset
        ds = load_dataset("wikitext", "wikitext-2-raw-v1", split=split)
        items = [{"text": row["text"]} for row in ds]
    except Exception:
        items = _generate_synthetic_bookcorpus(2000, seed)

    if num_samples > 0:
        items = items[:num_samples]
    return items


# ======================================================================
# C4 — Common Crawl validation set for PPL evaluation
# ======================================================================

_C4_VALIDATION_PATH_ENV = "C4_VALIDATION_JSON_GZ"


def load_c4_validation(
    num_samples: int = 1000,
    seed: int = 42,
) -> List[Dict]:
    """Load C4 validation set for PPL evaluation.

    Args:
        num_samples: number of samples to use. 0 = use all.
        seed: random seed (unused, kept for interface consistency).

    Returns:
        List of ``{"text": str}`` dicts.
    """
    from datasets import load_dataset
    c4_path = os.environ.get(_C4_VALIDATION_PATH_ENV)
    if c4_path:
        ds = load_dataset("json", data_files={"valid": c4_path}, split="valid")
    else:
        ds = load_dataset("allenai/c4", "en", split="validation", streaming=True)
        rows = []
        limit = num_samples if num_samples > 0 else 1000
        for row in ds:
            rows.append({"text": row["text"]})
            if len(rows) >= limit:
                break
        print(f"  C4 validation: {len(rows)} samples")
        return rows
    if num_samples > 0:
        ds = ds.select(range(min(num_samples, len(ds))))
    items = [{"text": row["text"]} for row in ds]
    print(f"  C4 validation: {len(items)} samples")
    return items


def load_mmlu_samples(
    num_calibration: int = 10,
    num_val: int = 100,
    dataset_name: str = "cais/mmlu",
    dataset_config: str = "all",
    seed: int = 42,
) -> Tuple[List[Dict], List[Dict]]:
    """Load MMLU samples for calibration and validation.

    Returns:
        (calibration_samples, validation_samples)
        Each sample: {"prompt": str, "answer": str, "choices": List[str]}
    """
    try:
        from datasets import load_dataset
        ds = load_dataset(dataset_name, dataset_config, split="test")
        items = list(ds)
    except Exception:
        # Fallback: generate synthetic MMLU-like data for testing
        items = _generate_synthetic_mmlu(num_calibration + num_val, seed)

    rng = random.Random(seed)
    rng.shuffle(items)

    samples = []
    for item in items[: num_calibration + num_val]:
        samples.append(_format_mmlu_item(item))

    cal = samples[:num_calibration]
    val = samples[num_calibration : num_calibration + num_val]
    return cal, val


def load_mmlu_disjoint_subsets(
    split: str,
    subset_sizes: Sequence[int],
    dataset_name: str = "cais/mmlu",
    dataset_config: str = "all",
    seed: int = 42,
) -> List[List[Dict]]:
    """Load multiple disjoint subsets from the same MMLU split.

    This is used to construct non-overlapping calibration/reward pools.
    """
    items = _load_raw_mmlu_split(
        split=split,
        dataset_name=dataset_name,
        dataset_config=dataset_config,
        seed=seed,
    )

    total_needed = sum(max(0, size) for size in subset_sizes)
    if total_needed > len(items):
        raise ValueError(
            f"Requested {total_needed} samples from split '{split}', "
            f"but only {len(items)} are available."
        )

    subsets = []
    start = 0
    for size in subset_sizes:
        end = start + max(0, size)
        subsets.append([_format_mmlu_item(item) for item in items[start:end]])
        start = end
    return subsets


def load_mmlu_split(
    split: str = "auxiliary_train",
    num_samples: int = 320,
    dataset_name: str = "cais/mmlu",
    dataset_config: str = "all",
    seed: int = 42,
) -> List[Dict]:
    """Load samples from a specific MMLU split.

    Args:
        split: One of 'test', 'validation', 'dev', 'auxiliary_train'.
        num_samples: Number of samples to load. 0 or negative = load all.
        seed: Random seed for shuffling before selection.

    Returns:
        List of formatted samples.
    """
    items = _load_raw_mmlu_split(
        split=split,
        dataset_name=dataset_name,
        dataset_config=dataset_config,
        seed=seed,
    )

    if num_samples > 0:
        items = items[:num_samples]

    return [_format_mmlu_item(item) for item in items]


def _load_raw_mmlu_split(
    split: str,
    dataset_name: str,
    dataset_config: str,
    seed: int,
) -> List[dict]:
    try:
        from datasets import load_dataset

        ds = load_dataset(dataset_name, dataset_config, split=split)
        items = list(ds)
    except Exception:
        items = _generate_synthetic_mmlu(100000, seed)

    rng = random.Random(seed)
    rng.shuffle(items)
    return items


def _format_mmlu_item(item: dict) -> dict:
    """Convert a raw MMLU item to our format."""
    question = item.get("question", item.get("input", ""))
    choices = item.get("choices", [])
    if not choices:
        choices = [item.get(f"choice_{c}", f"Option {c}") for c in CHOICE_LABELS]
    answer_idx = item.get("answer", 0)
    if isinstance(answer_idx, str):
        answer_idx = CHOICE_LABELS.index(answer_idx) if answer_idx in CHOICE_LABELS else 0

    prompt = format_mmlu_prompt(question, choices)
    return {
        "prompt": prompt,
        "answer": CHOICE_LABELS[answer_idx],
        "choices": choices,
        "answer_idx": int(answer_idx),
    }


def format_mmlu_prompt(question: str, choices: List[str]) -> str:
    lines = [question]
    for i, c in enumerate(choices):
        lines.append(f"{CHOICE_LABELS[i]}. {c}")
    lines.append("Answer:")
    return "\n".join(lines)


def _generate_synthetic_mmlu(n: int, seed: int = 42) -> List[dict]:
    """Generate synthetic MMLU-like data for testing without network access."""
    rng = random.Random(seed)
    items = []
    for i in range(n):
        items.append({
            "question": f"What is the answer to question {i}?",
            "choices": [f"Option {c} for q{i}" for c in CHOICE_LABELS],
            "answer": rng.randint(0, 3),
        })
    return items
