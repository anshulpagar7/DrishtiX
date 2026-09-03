"""
Download VRSBench and reformat it into a unified instruction-tuning JSON
covering VQA (open + closed-ended) and referring-expression grounding.

Output format (one JSON list, each item):
{
    "image": "relative/path/to/image.png",
    "task_type": "vqa_open" | "vqa_closed" | "grounding",
    "instruction": "Where are the buildings?",
    "response": "The buildings are concentrated in the northern half...",
    "bbox": [x, y, w, h] | null
}

This unified format is what train_lora.py consumes, regardless of which
VLM backbone is chosen in configs/model.yaml.

Usage:
    python scripts/prepare_dataset.py --output_dir data/vrsbench_processed --max_samples 4000
"""
import argparse
import json
import os
import random


def load_vrsbench(cache_dir: str):
    """
    Pulls VRSBench splits via the `datasets` library.
    VRSBench is hosted on Hugging Face: https://huggingface.co/datasets/xiang709/VRSBench
    (check the current dataset card for the exact repo id — HF dataset ids
    occasionally move; this is the one referenced in the VRSBench NeurIPS 2024 paper).
    """
    from datasets import load_dataset

    captioning = load_dataset("xiang709/VRSBench", "captioning", cache_dir=cache_dir)
    vqa = load_dataset("xiang709/VRSBench", "vqa", cache_dir=cache_dir)
    grounding = load_dataset("xiang709/VRSBench", "referring", cache_dir=cache_dir)
    return captioning, vqa, grounding


def is_closed_ended(question: str, answer: str) -> bool:
    """Heuristic: yes/no or short categorical answers count as closed-ended."""
    short_answer = len(answer.split()) <= 3
    yn_starts = ("is ", "are ", "does ", "do ", "was ", "were ", "can ", "has ")
    return short_answer or question.strip().lower().startswith(yn_starts)


def build_records(vqa_split, grounding_split, task_mixture: dict, max_samples: int):
    records = []

    for ex in vqa_split:
        task_type = "vqa_closed" if is_closed_ended(ex["question"], ex["answer"]) else "vqa_open"
        records.append({
            "image": ex["image_path"],
            "task_type": task_type,
            "instruction": ex["question"],
            "response": ex["answer"],
            "bbox": None,
        })

    for ex in grounding_split:
        records.append({
            "image": ex["image_path"],
            "task_type": "grounding",
            "instruction": ex["referring_expression"],
            "response": f"The region is located at the specified bounding box.",
            "bbox": ex["bbox"],  # expected as [x, y, w, h] in the source dataset
        })

    # Enforce the configured task mixture ratios, then cap to max_samples.
    by_type = {}
    for r in records:
        by_type.setdefault(r["task_type"], []).append(r)

    random.seed(42)
    for v in by_type.values():
        random.shuffle(v)

    total = min(max_samples, len(records))
    balanced = []
    for task_type, ratio in task_mixture.items():
        n = int(total * ratio)
        pool = by_type.get(task_type, [])
        balanced.extend(pool[:n])

    random.shuffle(balanced)
    return balanced[:max_samples]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--cache_dir", type=str, default="data/.hf_cache")
    parser.add_argument("--max_samples", type=int, default=4000)
    parser.add_argument("--val_split", type=float, default=0.05)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("Downloading VRSBench splits (this can take a while the first time)...")
    _, vqa, grounding = load_vrsbench(args.cache_dir)

    task_mixture = {"vqa_open": 0.4, "vqa_closed": 0.3, "grounding": 0.3}
    records = build_records(vqa["train"], grounding["train"], task_mixture, args.max_samples)

    random.shuffle(records)
    n_val = int(len(records) * args.val_split)
    val_records, train_records = records[:n_val], records[n_val:]

    with open(os.path.join(args.output_dir, "train.json"), "w") as f:
        json.dump(train_records, f, indent=2)
    with open(os.path.join(args.output_dir, "val.json"), "w") as f:
        json.dump(val_records, f, indent=2)

    print(f"Wrote {len(train_records)} train / {len(val_records)} val records to {args.output_dir}")
    print("Task type breakdown (train):")
    counts = {}
    for r in train_records:
        counts[r["task_type"]] = counts.get(r["task_type"], 0) + 1
    print(counts)


if __name__ == "__main__":
    main()
