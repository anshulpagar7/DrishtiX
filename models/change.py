"""
Semantic change detection.

Week 1 shipped PixelChangeDetector: it tells you *where* pixels moved. That is
half an answer. An analyst does not want "cell (3,5) changed by 12%", they want
"vegetation became built-up in the north-east".

This module adds the other half:

  - per-cell land cover BEFORE and AFTER, so change gets a type, not just a
    magnitude
  - a change heatmap as structured evidence, rendered as an overlay
  - a co-registration sanity check, because two tiles of different places will
    happily produce a confident, completely meaningless change map

That last one matters. Pixel differencing cannot tell "the city grew" from
"you uploaded two different cities". Detecting that and saying so is the
same honesty principle as the input validator, applied one layer deeper.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from contracts import Evidence, EvidenceKind, ModelOutput, TaskSpec, TaskType
from models.base import SpecialistModel
from models.heuristic import _cover_scores, _to_array

GRID = 8

# Co-registration check.
#
# First attempt used luminance correlation alone. It false-positived on the
# deforestation scene: when a THIRD of the tile genuinely changes, correlation
# drops below any useful threshold and real change gets reported as "different
# places". Caught in demo rehearsal, which is the point of rehearsing.
#
# The robust signal is the fraction of cells that stayed put. Two images of the
# same place keep most cells stable even under dramatic change; two unrelated
# tiles keep almost none.
#
# Correlation was originally a secondary veto signal alongside it, but a scene
# with two opposing local changes (a river channel drying in one place while a
# new one appears elsewhere) proved it unreliable in a way worth recording:
# whole-image Pearson correlation is dominated by whichever pixels carry
# variance, so two small anti-correlated changes can drag it sharply negative
# even with ~80% of the frame pixel-identical. It stayed at "weak secondary
# signal" in this comment while actually gating equally in the code below,
# which is what let it veto a same-place pair that the primary signal called
# fine. It is now reported for diagnostics only and never gates alone.
MIN_STABLE_FRACTION = 0.35      # at least this share of cells must be stable
STABLE_CELL_DIFF = 0.08         # a cell differing by less than this is stable

# A cell must exceed this multiple of the tile's mean difference to count.
CELL_THRESHOLD = 1.5

# Cover transitions worth naming explicitly, keyed (from, to).
TRANSITIONS = {
    ("vegetation", "urban / built-up"): "vegetation cleared for development",
    ("vegetation", "bare soil"): "vegetation loss / clearing",
    ("bare soil", "urban / built-up"): "new construction on open ground",
    ("bare soil", "vegetation"): "revegetation or new planting",
    ("water", "bare soil"): "water body receded",
    ("bare soil", "water"): "flooding or water expansion",
    ("vegetation", "water"): "inundation of vegetated land",
    ("water", "vegetation"): "wetland drying or reclamation",
    ("urban / built-up", "bare soil"): "demolition or clearance",
    ("snow, cloud or sand", "bare soil"): "snow or ice retreat",
    ("bare soil", "snow, cloud or sand"): "snow accumulation",
}


def _structural_correlation(a: np.ndarray, b: np.ndarray) -> float:
    """Luminance correlation. Weak secondary co-registration signal."""
    x = a.mean(axis=-1).ravel()
    y = b.mean(axis=-1).ravel()
    if x.std() < 1e-6 or y.std() < 1e-6:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def _stable_fraction(diff: np.ndarray, grid: int = GRID) -> float:
    """Share of grid cells that barely differ between the two tiles.

    This is the primary co-registration signal. Robust to large genuine
    change in a way that whole-image correlation is not.
    """
    h, w = diff.shape
    ch, cw = max(h // grid, 1), max(w // grid, 1)
    stable = total = 0
    for gy in range(grid):
        for gx in range(grid):
            cell = diff[gy * ch:(gy + 1) * ch, gx * cw:(gx + 1) * cw]
            if not cell.size:
                continue
            total += 1
            if float(cell.mean()) < STABLE_CELL_DIFF:
                stable += 1
    return stable / total if total else 0.0


def _cell_cover(arr: np.ndarray, gy: int, gx: int, ch: int, cw: int) -> str:
    cell = arr[gy * ch:(gy + 1) * ch, gx * cw:(gx + 1) * cw]
    if cell.size == 0:
        return "unknown"
    s = _cover_scores(cell)
    return max(s, key=s.get)


class SemanticChangeDetector(SpecialistModel):
    """Types the change, not just its magnitude. Emits a heatmap."""

    model_id = "semantic-change-v1"
    supported_tasks = {TaskType.CHANGE}
    priority = 10                  # above the pixel baseline, below any VLM

    def can_handle(self, spec: TaskSpec) -> bool:
        return spec.task_type is TaskType.CHANGE

    def _run(self, spec: TaskSpec, images: list[Any]) -> ModelOutput:
        if len(images) < 2:
            return ModelOutput(
                answer="Change detection needs two images.",
                confidence=0.0,
                model_id=self.model_id,
            )

        a, b = _to_array(images[0]), _to_array(images[1])
        h = min(a.shape[0], b.shape[0])
        w = min(a.shape[1], b.shape[1])
        a, b = a[:h, :w], b[:h, :w]

        diff = np.abs(a - b).mean(axis=-1)
        ch, cw = max(h // GRID, 1), max(w // GRID, 1)
        overall = float(diff.mean())

        # --- co-registration sanity check ---------------------------------
        stable = _stable_fraction(diff)
        corr = _structural_correlation(a, b)          # diagnostic only, see note above
        if stable < MIN_STABLE_FRACTION:
            return ModelOutput(
                answer=(f"These two tiles do not appear to show the same place "
                        f"(only {stable:.0%} of the scene is stable). Any change "
                        f"map would be meaningless, so none was produced."),
                confidence=round(1.0 - stable, 3),
                model_id=self.model_id,
                veto=True,      # other models' change maps are meaningless here
                scores={"stable_fraction": round(stable, 3),
                        "structural_correlation": round(corr, 3)},
                evidence=[Evidence(image_index=0, kind=EvidenceKind.WHOLE,
                                   note="co-registration check failed")],
            )

        heat: list[list[float]] = []
        changed: list[tuple[float, int, int, str, str]] = []

        for gy in range(GRID):
            row: list[float] = []
            for gx in range(GRID):
                cell = diff[gy * ch:(gy + 1) * ch, gx * cw:(gx + 1) * cw]
                mag = float(cell.mean()) if cell.size else 0.0
                row.append(mag)
                if cell.size and mag > overall * CELL_THRESHOLD:
                    before = _cell_cover(a, gy, gx, ch, cw)
                    after = _cell_cover(b, gy, gx, ch, cw)
                    if before != after:
                        changed.append((mag, gx, gy, before, after))
            heat.append(row)

        # Normalise the heatmap to 0-1 for rendering.
        peak = max((v for row in heat for v in row), default=0.0) or 1.0
        heat = [[round(v / peak, 3) for v in row] for row in heat]

        changed.sort(reverse=True)

        # Collapse repeats: several adjacent cells usually share one
        # transition in one compass sector. Reporting it three times reads
        # as a bug, because it is one.
        top: list[tuple[float, int, int, str, str]] = []
        seen: set[tuple[str, str, str]] = set()
        for mag, gx, gy, before, after in changed:
            sector = (f"{'north' if gy < GRID / 2 else 'south'}-"
                      f"{'west' if gx < GRID / 2 else 'east'}")
            key = (before, after, sector)
            if key in seen:
                continue
            seen.add(key)
            top.append((mag, gx, gy, before, after))
            if len(top) >= 3:
                break

        # --- global cover shift -------------------------------------------
        sa, sb = _cover_scores(a), _cover_scores(b)
        shifts = {k: round(sb[k] - sa[k], 3) for k in sa}
        gained = max(shifts, key=shifts.get)
        lost = min(shifts, key=shifts.get)

        # --- narrate -------------------------------------------------------
        if not top:
            text = (f"No typed land-cover transition stands out. Mean pixel "
                    f"difference is {overall:.1%}, and cover fractions are "
                    f"broadly stable.")
        else:
            named = []
            for mag, gx, gy, before, after in top:
                label = TRANSITIONS.get((before, after), f"{before} to {after}")
                where = (f"{'north' if gy < GRID / 2 else 'south'}-"
                         f"{'west' if gx < GRID / 2 else 'east'}")
                named.append(f"{label} in the {where}")
            text = (f"{len(top)} typed transition(s) detected: "
                    f"{'; '.join(named)}. "
                    f"Overall {gained} rose {shifts[gained]:+.0%} while "
                    f"{lost} fell {shifts[lost]:+.0%}.")

        evidence: list[Evidence] = [
            Evidence(
                image_index=1,
                kind=EvidenceKind.HEATMAP,
                grid=heat,
                score=round(overall, 4),
                note="change intensity per cell",
            )
        ]
        for mag, gx, gy, before, after in top:
            evidence.append(Evidence(
                image_index=1,
                kind=EvidenceKind.REGION,
                bbox=(gx / GRID, gy / GRID, (gx + 1) / GRID, (gy + 1) / GRID),
                score=round(mag, 4),
                note=TRANSITIONS.get((before, after), f"{before} to {after}"),
            ))

        return ModelOutput(
            answer=text,
            confidence=round(min(overall * 6, 1.0) * min(stable * 1.5, 1.0), 3),
            model_id=self.model_id,
            scores={**shifts, "stable_fraction": round(stable, 3),
                    "structural_correlation": round(corr, 3)},
            evidence=evidence,
        )
