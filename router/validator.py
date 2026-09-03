"""
Input validation: can the supplied images actually answer this question?

This is the credibility feature. When an evaluator asks "how do we know it
isn't making things up?", the answer is this file plus the evidence overlay.

Every refusal says what is wrong AND what to do about it. A refusal that
leaves the user stuck is a bug.
"""

from __future__ import annotations

from contracts import Modality, TaskSpec, TaskType, ValidationResult

# Images smaller than this are almost certainly thumbnails or screenshots,
# not usable satellite tiles.
MIN_SIDE_PX = 64

# Above this ratio, two images are too different in shape to be the same
# scene at two dates.
MAX_ASPECT_MISMATCH = 1.15


def _aspect(size: tuple[int, int]) -> float:
    w, h = size
    return w / h if h else 0.0


def validate(spec: TaskSpec, image_sizes: list[tuple[int, int]]) -> ValidationResult:
    """Check a parsed query against the images the user actually supplied.

    image_sizes is a list of (width, height) so this stays dependency-free
    and unit-testable without loading real imagery.
    """
    n = len(image_sizes)

    # --- nothing to look at -----------------------------------------------
    if n == 0:
        return ValidationResult(
            ok=False,
            reason="No image supplied.",
            fix_hint="Upload at least one satellite image to ask about.",
        )

    # --- unusable imagery --------------------------------------------------
    for i, (w, h) in enumerate(image_sizes):
        if min(w, h) < MIN_SIDE_PX:
            return ValidationResult(
                ok=False,
                reason=f"Image {i + 1} is {w}x{h}, too small to analyse.",
                fix_hint=f"Supply a tile at least {MIN_SIDE_PX}x{MIN_SIDE_PX} pixels.",
            )

    # --- the headline check: change queries need two images ----------------
    if spec.needs_pair and n < 2:
        return ValidationResult(
            ok=False,
            reason="This asks what changed, but only one image was supplied.",
            fix_hint="Add a second image of the same area from a different date.",
        )

    if spec.needs_pair and n >= 2:
        a0, a1 = _aspect(image_sizes[0]), _aspect(image_sizes[1])
        if a0 and a1 and max(a0 / a1, a1 / a0) > MAX_ASPECT_MISMATCH:
            return ValidationResult(
                ok=False,
                reason="The two images have different shapes, so they are probably "
                       "not the same area.",
                fix_hint="Supply co-registered tiles covering the same footprint.",
            )

    # --- single-image tasks given a pair -----------------------------------
    if not spec.needs_pair and n > 1 and spec.task_type in {TaskType.CAPTION, TaskType.GROUND}:
        return ValidationResult(
            ok=True,
            reason="Several images supplied; answering about the first one.",
        )

    # --- the parser could not tell what was being asked --------------------
    if spec.task_type is TaskType.UNKNOWN:
        return ValidationResult(
            ok=False,
            reason="Could not tell what this question is asking for.",
            fix_hint="Try naming the task: describe, classify, locate, or compare.",
        )

    # --- modality the registry may not cover -------------------------------
    if spec.modality is Modality.SAR:
        return ValidationResult(
            ok=True,
            reason="Reading this as SAR imagery; optical-trained models may be "
                   "less reliable here.",
        )

    return ValidationResult(ok=True)
