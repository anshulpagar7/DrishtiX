"""
Change localisation evaluation.

Answer accuracy tells you whether the text is right. This tells you whether
the *box* is right, which is the thing on screen during the demo. A system
that says "vegetation cleared in the south-west" while highlighting the
north-east is worse than one that says nothing.

Synthetic scenes are used because they are the only pairs where ground truth
is exact. Report these numbers as what they are - localisation on controlled
scenes - and never imply they are Sentinel benchmark results.

Run:  python -m eval.change_eval
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config                                               # noqa: E402
from contracts import EvidenceKind                          # noqa: E402
from demo.samples import SCENES, build                      # noqa: E402
from eval.metrics import markdown_table                     # noqa: E402
from pipeline import SatQueryPipeline                       # noqa: E402

QUERY = "What changed between these two images?"


def iou(a: tuple[float, float, float, float],
        b: tuple[float, float, float, float]) -> float:
    """Intersection over union of two normalised boxes."""
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1])
    ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    union = area_a + area_b - inter
    return inter / union if union else 0.0


def hit_rate(pred_boxes: list, gt: tuple) -> float:
    """Fraction of predicted boxes whose centre falls inside ground truth.

    Centre-in-box rather than IoU, because the detector reports 8x8 grid
    cells while ground truth is a free-form region - IoU would punish a
    correct cell for being small. Both are reported; read them together.
    """
    if not pred_boxes:
        return 0.0
    inside = 0
    for x0, y0, x1, y1 in pred_boxes:
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        if gt[0] <= cx <= gt[2] and gt[1] <= cy <= gt[3]:
            inside += 1
    return inside / len(pred_boxes)


def evaluate(size: int = 256) -> dict:
    pipe = SatQueryPipeline()
    rows: list[dict] = []

    for name in SCENES:
        before, after, gt, desc = build(name, size)
        ans = pipe.ask(QUERY, [before, after])

        boxes = [e.bbox for e in ans.evidence
                 if e.kind is EvidenceKind.REGION and e.bbox]
        has_heat = any(e.kind is EvidenceKind.HEATMAP and e.grid
                       for e in ans.evidence)

        if gt is None:
            # Negative cases: the correct behaviour is to report nothing.
            correct = len(boxes) == 0
            rows.append({
                "scene": name,
                "type": "negative",
                "boxes": len(boxes),
                "best_iou": "-",
                "hit_rate": "-",
                "heatmap": has_heat,
                "correct": correct,
            })
            continue

        best = max((iou(b, gt) for b in boxes), default=0.0)
        hr = hit_rate(boxes, gt)
        rows.append({
            "scene": name,
            "type": "positive",
            "boxes": len(boxes),
            "best_iou": round(best, 3),
            "hit_rate": round(hr, 3),
            "heatmap": has_heat,
            "correct": hr >= 0.5,
        })

    positives = [r for r in rows if r["type"] == "positive"]
    negatives = [r for r in rows if r["type"] == "negative"]
    summary = {
        "scenes": len(rows),
        "positive_correct": sum(r["correct"] for r in positives),
        "positive_total": len(positives),
        "negative_correct": sum(r["correct"] for r in negatives),
        "negative_total": len(negatives),
        "mean_hit_rate": round(
            sum(r["hit_rate"] for r in positives) / max(len(positives), 1), 3),
        "mean_best_iou": round(
            sum(r["best_iou"] for r in positives) / max(len(positives), 1), 3),
    }
    return {"rows": rows, "summary": summary}


def main() -> None:
    report = evaluate()
    s = report["summary"]

    print("\nChange localisation (synthetic scenes)")
    print("=" * 68)
    print(markdown_table(
        report["rows"],
        ["scene", "type", "boxes", "best_iou", "hit_rate", "heatmap", "correct"],
    ))
    print("=" * 68)
    print(f"positive scenes  {s['positive_correct']}/{s['positive_total']} "
          f"localised (hit rate >= 0.5)")
    print(f"negative scenes  {s['negative_correct']}/{s['negative_total']} "
          f"correctly reported no change")
    print(f"mean hit rate    {s['mean_hit_rate']:.1%}")
    print(f"mean best IoU    {s['mean_best_iou']:.1%}")

    config.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out = config.REPORT_DIR / "change_report.json"
    out.write_text(json.dumps(report, indent=2))
    print(f"\nreport -> {out}")


if __name__ == "__main__":
    main()
