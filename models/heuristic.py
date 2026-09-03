"""
Week-1 baseline models. CPU-only, no weights, no downloads, no accounts.

These are NOT placeholders that fake an answer. They are real spectral and
pixel-difference baselines. That matters for two reasons:

  1. The demo works today, on a laptop, offline.
  2. In week 2 you report "fine-tuned VLM vs heuristic baseline" instead of
     "fine-tuned VLM vs nothing". Every serious evaluation needs a floor,
     and a naive baseline is the honest floor.

Keep these registered forever at priority 0. When a real model fails to load
mid-demo, the router falls through to these and the demo survives.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from contracts import Evidence, ModelOutput, TaskSpec, TaskType
from models.base import SpecialistModel

# --------------------------------------------------------------------------
# Shared spectral helpers
# --------------------------------------------------------------------------

_GRID = 8          # change detection cell grid, 8x8 = 64 cells
_TOP_CELLS = 3     # how many changed cells to report as evidence
_MIN_ABS_CHANGE = 0.02   # below this a "change" is sensor noise


def _to_array(image: Any) -> np.ndarray:
    """Accept a PIL image or an array. Return float RGB in 0-1, HxWx3."""
    arr = np.asarray(image, dtype=np.float32)
    if arr.ndim == 2:                       # greyscale -> 3 channels
        arr = np.stack([arr] * 3, axis=-1)
    if arr.shape[-1] == 4:                  # drop alpha
        arr = arr[..., :3]
    if arr.max() > 1.5:                     # 0-255 -> 0-1
        arr = arr / 255.0
    return np.clip(arr, 0.0, 1.0)


def _cover_scores(arr: np.ndarray) -> dict[str, float]:
    """Crude land-cover fractions from RGB statistics.

    No NIR band in an RGB upload, so this uses a visible-band greenness index
    as an NDVI stand-in. State that limitation in the pitch - naming what your
    baseline cannot do reads as rigour, not weakness.
    """
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    eps = 1e-6

    # Visible-band greenness (ExG-style), stands in for NDVI.
    greenness = (2 * g - r - b) / (r + g + b + eps)
    brightness = arr.mean(axis=-1)
    saturation = arr.max(axis=-1) - arr.min(axis=-1)

    vegetation = float((greenness > 0.05).mean())
    water = float(((b > r) & (b > g) & (brightness < 0.45)).mean())
    urban = float(((saturation < 0.12) & (brightness > 0.35) & (brightness < 0.8)).mean())
    bare = float(((r > g) & (g > b) & (brightness > 0.4)).mean())
    bright = float((brightness > 0.85).mean())      # snow, cloud, sand

    scores = {
        "vegetation": vegetation,
        "water": water,
        "urban / built-up": urban,
        "bare soil": bare,
        "snow, cloud or sand": bright,
    }
    total = sum(scores.values()) or 1.0
    return {k: round(v / total, 3) for k, v in scores.items()}


def _dominant(scores: dict[str, float]) -> tuple[str, float]:
    label = max(scores, key=scores.get)
    return label, scores[label]


# --------------------------------------------------------------------------
# Classification
# --------------------------------------------------------------------------

class HeuristicClassifier(SpecialistModel):
    """Multi-label land cover from visible-band statistics."""

    model_id = "heuristic-cover-v1"
    supported_tasks = {TaskType.CLASSIFY}
    priority = 0

    def _run(self, spec: TaskSpec, images: list[Any]) -> ModelOutput:
        arr = _to_array(images[0])
        scores = _cover_scores(arr)
        present = {k: v for k, v in scores.items() if v >= 0.15}
        if not present:
            present = dict([max(scores.items(), key=lambda kv: kv[1])])

        listed = ", ".join(f"{k} ({v:.0%})" for k, v in
                           sorted(present.items(), key=lambda kv: -kv[1]))
        return ModelOutput(
            answer=f"Land cover present: {listed}.",
            confidence=round(float(max(scores.values())), 3),
            model_id=self.model_id,
            scores=scores,
            evidence=[Evidence(image_index=0, note="whole-tile spectral statistics")],
        )


# --------------------------------------------------------------------------
# Captioning
# --------------------------------------------------------------------------

class HeuristicCaptioner(SpecialistModel):
    """One-sentence scene description built from cover fractions."""

    model_id = "heuristic-caption-v1"
    supported_tasks = {TaskType.CAPTION}
    priority = 0

    def _run(self, spec: TaskSpec, images: list[Any]) -> ModelOutput:
        arr = _to_array(images[0])
        scores = _cover_scores(arr)
        ranked = sorted(scores.items(), key=lambda kv: -kv[1])
        first, second = ranked[0], ranked[1]

        h, w = arr.shape[:2]
        text = (f"A {w}x{h} tile dominated by {first[0]} ({first[1]:.0%}), "
                f"with {second[0]} making up {second[1]:.0%} of the scene.")
        return ModelOutput(
            answer=text,
            confidence=round(float(first[1]), 3),
            model_id=self.model_id,
            scores=scores,
            evidence=[Evidence(image_index=0, note="whole-tile spectral statistics")],
        )


# --------------------------------------------------------------------------
# Grounding
# --------------------------------------------------------------------------

class HeuristicGrounder(SpecialistModel):
    """Locate a named cover class by finding the grid cell richest in it."""

    model_id = "heuristic-ground-v1"
    supported_tasks = {TaskType.GROUND}
    priority = 0

    _ALIASES = {
        "water": "water", "river": "water", "lake": "water", "sea": "water",
        "ocean": "water", "flood": "water", "flooded": "water",
        "forest": "vegetation", "trees": "vegetation", "vegetation": "vegetation",
        "farmland": "vegetation", "cropland": "vegetation", "field": "vegetation",
        "grassland": "vegetation", "pasture": "vegetation", "agriculture": "vegetation",
        "urban": "urban / built-up", "city": "urban / built-up",
        "buildings": "urban / built-up", "settlement": "urban / built-up",
        "residential": "urban / built-up", "industrial": "urban / built-up",
        "snow": "snow, cloud or sand", "ice": "snow, cloud or sand",
        "glacier": "snow, cloud or sand", "sand": "snow, cloud or sand",
        "beach": "snow, cloud or sand", "desert": "bare soil",
        "bare soil": "bare soil", "rock": "bare soil",
    }

    def can_handle(self, spec: TaskSpec) -> bool:
        return spec.task_type is TaskType.GROUND and spec.target_class is not None

    def _run(self, spec: TaskSpec, images: list[Any]) -> ModelOutput:
        arr = _to_array(images[0])
        target = self._ALIASES.get((spec.target_class or "").lower())

        if target is None:
            return ModelOutput(
                answer=f"No detector available for '{spec.target_class}'.",
                confidence=0.0,
                model_id=self.model_id,
            )

        h, w = arr.shape[:2]
        ch, cw = max(h // _GRID, 1), max(w // _GRID, 1)
        best, best_score = (0, 0), -1.0

        for gy in range(_GRID):
            for gx in range(_GRID):
                cell = arr[gy * ch:(gy + 1) * ch, gx * cw:(gx + 1) * cw]
                if cell.size == 0:
                    continue
                score = _cover_scores(cell).get(target, 0.0)
                if score > best_score:
                    best, best_score = (gx, gy), score

        gx, gy = best
        bbox = (gx / _GRID, gy / _GRID, (gx + 1) / _GRID, (gy + 1) / _GRID)
        return ModelOutput(
            answer=(f"Strongest {spec.target_class} signal is in the "
                    f"{'top' if gy < _GRID / 2 else 'bottom'}-"
                    f"{'left' if gx < _GRID / 2 else 'right'} of the tile."),
            confidence=round(float(best_score), 3),
            model_id=self.model_id,
            evidence=[Evidence(image_index=0, bbox=bbox,
                               note=f"highest {target} fraction of any cell")],
        )


# --------------------------------------------------------------------------
# Change detection - the demo centrepiece
# --------------------------------------------------------------------------

class PixelChangeDetector(SpecialistModel):
    """Grid-cell difference between two co-registered tiles.

    Genuinely useful, genuinely cheap, and it produces the visual that makes
    evaluators lean forward: highlighted boxes over what moved.
    """

    model_id = "pixel-change-v1"
    supported_tasks = {TaskType.CHANGE}
    priority = 0

    def can_handle(self, spec: TaskSpec) -> bool:
        return spec.task_type is TaskType.CHANGE

    def _run(self, spec: TaskSpec, images: list[Any]) -> ModelOutput:
        a = _to_array(images[0])
        b = _to_array(images[1])

        # Match shapes by cropping to the common region.
        h = min(a.shape[0], b.shape[0])
        w = min(a.shape[1], b.shape[1])
        a, b = a[:h, :w], b[:h, :w]

        diff = np.abs(a - b).mean(axis=-1)
        ch, cw = max(h // _GRID, 1), max(w // _GRID, 1)

        cells: list[tuple[float, int, int]] = []
        for gy in range(_GRID):
            for gx in range(_GRID):
                cell = diff[gy * ch:(gy + 1) * ch, gx * cw:(gx + 1) * cw]
                if cell.size:
                    cells.append((float(cell.mean()), gx, gy))

        cells.sort(reverse=True)
        overall = float(diff.mean())
        # Relative threshold alone fires on pure sensor noise, where some cell
        # is always 1.5x the mean of nothing. The absolute floor is what makes
        # "no change" actually report no change.
        top = [c for c in cells[:_TOP_CELLS]
               if c[0] > overall * 1.5 and c[0] > _MIN_ABS_CHANGE]

        # Which direction did cover shift?
        sa, sb = _cover_scores(a), _cover_scores(b)
        shifts = {k: round(sb[k] - sa[k], 3) for k in sa}
        gained = max(shifts, key=shifts.get)
        lost = min(shifts, key=shifts.get)

        if not top:
            text = (f"No localised change stands out. Mean pixel difference is "
                    f"{overall:.1%} across the tile.")
        else:
            text = (f"Change concentrated in {len(top)} region(s). "
                    f"{gained} increased by {shifts[gained]:+.0%} and "
                    f"{lost} decreased by {shifts[lost]:+.0%}.")

        evidence = [
            Evidence(
                image_index=1,
                bbox=(gx / _GRID, gy / _GRID, (gx + 1) / _GRID, (gy + 1) / _GRID),
                note=f"mean difference {score:.1%}",
            )
            for score, gx, gy in top
        ]

        return ModelOutput(
            answer=text,
            confidence=round(min(overall * 4, 1.0), 3),
            model_id=self.model_id,
            scores=shifts,
            evidence=evidence,
        )


# --------------------------------------------------------------------------
# VQA fallback
# --------------------------------------------------------------------------

class HeuristicVQA(SpecialistModel):
    """Answers only the narrow questions cover statistics can actually support.

    Refuses everything else rather than guessing. That refusal is the point.
    """

    model_id = "heuristic-vqa-v1"
    supported_tasks = {TaskType.VQA}
    priority = 0

    def _run(self, spec: TaskSpec, images: list[Any]) -> ModelOutput:
        arr = _to_array(images[0])
        scores = _cover_scores(arr)
        q = spec.raw_query.lower()
        target = spec.target_class

        # "is there X" / "how much X"
        if target:
            from models.heuristic import HeuristicGrounder
            key = HeuristicGrounder._ALIASES.get(target.lower())
            if key:
                frac = scores[key]
                present = frac >= 0.15
                if "how much" in q or "what fraction" in q or "percentage" in q:
                    text = f"About {frac:.0%} of the tile reads as {target}."
                else:
                    text = (f"Yes - {target} covers roughly {frac:.0%} of the tile."
                            if present else
                            f"No clear {target} signal; it would be under 15% of the tile.")
                return ModelOutput(
                    answer=text,
                    confidence=round(float(frac if present else 1 - frac), 3),
                    model_id=self.model_id,
                    scores=scores,
                    evidence=[Evidence(image_index=0,
                                       note=f"{key} fraction from spectral statistics")],
                )

        label, frac = _dominant(scores)
        return ModelOutput(
            answer=(f"This baseline only reports land cover. The dominant class here "
                    f"is {label} at {frac:.0%}. A fine-tuned VLM is needed for "
                    f"open-ended questions like this one."),
            confidence=0.1,
            model_id=self.model_id,
            scores=scores,
            evidence=[Evidence(image_index=0, note="whole-tile spectral statistics")],
        )


def default_models() -> list[SpecialistModel]:
    """Everything registered by default in week 1."""
    return [
        HeuristicClassifier(),
        HeuristicCaptioner(),
        HeuristicGrounder(),
        PixelChangeDetector(),
        HeuristicVQA(),
    ]
