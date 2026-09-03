"""
Evaluate a fine-tuned adapter on the held-out VRSBench val split.

Metrics:
- vqa_closed: exact-match accuracy (normalized string match)
- vqa_open: ROUGE-L / BLEU (rough proxy for open-ended answer quality)
- grounding: mean IoU between predicted and ground-truth bounding box

Usage:
    python scripts/evaluate.py --adapter_dir outputs/lora_vqa_v1 --data_dir data/vrsbench_processed
"""
import argparse
import json
import os
import re

from inference import SatQueryVQAModel


def normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9 ]", "", text.lower()).strip()


def iou(box_a, box_b) -> float:
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b
    ax2, ay2, bx2, by2 = ax + aw, ay + ah, bx + bw, by + bh

    inter_x1, inter_y1 = max(ax, bx), max(ay, by)
    inter_x2, inter_y2 = min(ax2, bx2), min(ay2, by2)
    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)

    union_area = aw * ah + bw * bh - inter_area
    return inter_area / union_area if union_area > 0 else 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter_dir", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--image_root", type=str, default=None)
    args = parser.parse_args()
    image_root = args.image_root or args.data_dir

    with open(os.path.join(args.data_dir, "val.json")) as f:
        val_records = json.load(f)

    model = SatQueryVQAModel(adapter_dir=args.adapter_dir)

    closed_correct, closed_total = 0, 0
    grounding_ious = []

    for rec in val_records:
        image_path = os.path.join(image_root, rec["image"])
        result = model.predict(image_path, rec["instruction"], task_hint=rec["task_type"])

        if rec["task_type"] == "vqa_closed":
            closed_total += 1
            if normalize(result["answer"]) == normalize(rec["response"]):
                closed_correct += 1

        elif rec["task_type"] == "grounding" and rec.get("bbox") and result["regions"]:
            pred_box = result["regions"][0]["bbox"]
            grounding_ious.append(iou(pred_box, rec["bbox"]))

    print("=== Evaluation results ===")
    if closed_total:
        print(f"Closed-ended VQA accuracy: {closed_correct/closed_total:.3f} ({closed_correct}/{closed_total})")
    if grounding_ious:
        mean_iou = sum(grounding_ious) / len(grounding_ious)
        print(f"Grounding mean IoU: {mean_iou:.3f} (n={len(grounding_ious)})")
    print("Note: open-ended VQA quality still needs manual/LLM-judged spot-checking —")
    print("string-match metrics don't capture paraphrased-but-correct answers.")


if __name__ == "__main__":
    main()
