"""
Dataset subset builders.

Full BigEarthNet is well over 100 GB. You need none of it. These builders
stream from the Hugging Face Hub and stop after N samples, so a usable
training set lands in minutes on a Kaggle notebook instead of hours.

Everything is written to DATA_DIR as JSONL plus a folder of JPEGs, so:
  - the subset is reproducible from a seed
  - one person builds it once and pushes to HF Datasets
  - the other five pull identical data instead of five different subsets

Run:
    python -m data.subsets --dataset bigearthnet
    python -m data.subsets --all
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Iterator

import config
from contracts import TaskType
from data.prompts import build_training_prompt

# --------------------------------------------------------------------------
# Dataset registry
# --------------------------------------------------------------------------
# HF paths change. Verify each on the Hub before your first run and update
# here rather than scattering dataset IDs through the codebase.

SOURCES: dict[str, dict[str, Any]] = {
    # Primary. VRSBench covers captioning, visual grounding and VQA in one
    # collection - 29,614 images with human-verified captions, 52,472 object
    # references, 123,221 QA pairs, built on DOTA-v2 and DIOR (NeurIPS 2024).
    # One dataset spanning three tasks is what makes the router worth having.
    "vrsbench": {
        "hf_path": "xiang709/VRSBench",
        "split": "train",
        "task": TaskType.CAPTION,
        "note": "Captioning, grounding and VQA over remote sensing imagery.",
    },
    "bigearthnet": {
        "hf_path": "blanchon/BigEarthNet-V2",
        "split": "train",
        "task": TaskType.CLASSIFY,
        "note": "Sentinel-1/2 multi-label land cover.",
    },
    "rsvqa": {
        "hf_path": "flax-sentence-embeddings/RSVQA-LR",
        "split": "train",
        "task": TaskType.VQA,
        "note": "Question answering over low-res remote sensing tiles.",
    },
    "vrsbench": {
        "hf_path": "xiang709/VRSBench",
        "split": "train",
        "task": TaskType.CAPTION,
        "note": "Captioning, grounding and VQA over remote sensing imagery.",
    },
    "cdvqa": {
        "hf_path": "cdvqa/CDVQA",
        "split": "train",
        "task": TaskType.CHANGE,
        "note": "Change-detection VQA over image pairs.",
    },
}


# --------------------------------------------------------------------------
# Record shape - one JSONL line per training example
# --------------------------------------------------------------------------

def make_record(idx: int, task: TaskType, prompt: str, answer: str,
                images: list[str], meta: dict | None = None) -> dict:
    return {
        "id": idx,
        "task": task.value,
        "prompt": prompt,
        "answer": answer,
        "images": images,          # relative paths under the subset folder
        "meta": meta or {},
    }


# --------------------------------------------------------------------------
# Extraction - each dataset names its fields differently
# --------------------------------------------------------------------------

def _first_key(row: dict, candidates: list[str]) -> Any:
    for k in candidates:
        if k in row and row[k] is not None:
            return row[k]
    return None


def _as_answer(value: Any) -> str:
    if isinstance(value, (list, tuple)):
        return ", ".join(str(v) for v in value)
    return str(value)


def extract(name: str, row: dict) -> tuple[str, str, list[Any]] | None:
    """Row -> (prompt, answer, [PIL images]). None means skip the row."""
    task = SOURCES[name]["task"]

    img = _first_key(row, ["image", "img", "image_1", "before", "rgb"])
    img2 = _first_key(row, ["image_2", "after", "image_b"])
    images = [i for i in (img, img2) if i is not None]
    if not images:
        return None

    if task is TaskType.CLASSIFY:
        labels = _first_key(row, ["labels", "label", "classes", "multilabel"])
        if labels is None:
            return None
        return build_training_prompt(task), _as_answer(labels), images

    if task is TaskType.CAPTION:
        cap = _first_key(row, ["caption", "captions", "description", "text"])
        if cap is None:
            return None
        return build_training_prompt(task), _as_answer(cap), images

    # VQA and CHANGE both carry a question.
    q = _first_key(row, ["question", "query", "prompt"])
    a = _first_key(row, ["answer", "answers", "label", "response"])
    if q is None or a is None:
        return None
    return build_training_prompt(task, question=str(q)), _as_answer(a), images


# --------------------------------------------------------------------------
# Builder
# --------------------------------------------------------------------------

def stream_rows(hf_path: str, split: str) -> Iterator[dict]:
    try:
        from datasets import load_dataset
    except ImportError as exc:                       # pragma: no cover
        raise SystemExit(
            "`datasets` is not installed. Run: pip install -r requirements-train.txt"
        ) from exc

    ds = load_dataset(hf_path, split=split, streaming=True)
    for row in ds:
        yield row


def build(name: str, n: int | None = None, out_dir: Path | None = None,
          seed: int = config.SEED) -> Path:
    """Build one subset. Returns the folder it was written to."""
    if name not in SOURCES:
        raise KeyError(f"Unknown dataset '{name}'. Known: {list(SOURCES)}")

    src = SOURCES[name]
    n = n or config.SUBSET_SIZES.get(name, 1000)
    out = out_dir or (config.DATA_DIR / name)
    (out / "images").mkdir(parents=True, exist_ok=True)

    random.seed(seed)
    records: list[dict] = []
    kept = skipped = 0

    print(f"[{name}] streaming {src['hf_path']} -> {n} samples")
    for row in stream_rows(src["hf_path"], src["split"]):
        if kept >= n:
            break
        got = extract(name, row)
        if got is None:
            skipped += 1
            continue

        prompt, answer, images = got
        paths: list[str] = []
        for j, im in enumerate(images):
            rel = f"images/{kept:06d}_{j}.jpg"
            try:
                im.convert("RGB").save(out / rel, quality=90)
            except AttributeError:
                skipped += 1
                paths = []
                break
            paths.append(rel)
        if not paths:
            continue

        records.append(make_record(kept, src["task"], prompt, answer, paths))
        kept += 1
        if kept % 500 == 0:
            print(f"  {kept}/{n}")

    # Deterministic shuffle, then a held-out split you never train on.
    random.shuffle(records)
    cut = max(int(len(records) * 0.1), 1)
    splits = {"test": records[:cut], "train": records[cut:]}

    for split_name, rows in splits.items():
        path = out / f"{split_name}.jsonl"
        with path.open("w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print(f"[{name}] {split_name}: {len(rows)} -> {path}")

    print(f"[{name}] kept {kept}, skipped {skipped}")
    return out


def load_jsonl(path: Path) -> list[dict]:
    with Path(path).open() as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    ap = argparse.ArgumentParser(description="Build dataset subsets.")
    ap.add_argument("--dataset", choices=list(SOURCES))
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--n", type=int, default=None)
    args = ap.parse_args()

    if args.all:
        for name in SOURCES:
            build(name, args.n)
    elif args.dataset:
        build(args.dataset, args.n)
    else:
        ap.error("pass --dataset NAME or --all")


if __name__ == "__main__":
    main()
