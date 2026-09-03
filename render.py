"""
Evidence rendering.

The overlay is the feature the pitch rests on. When an evaluator asks "how do
we know it isn't making this up", you do not explain - you point at the screen.
So this file gets more visual care than anything else in the repo.

Everything here is pure PIL. No matplotlib, no extra dependency, and it works
identically in the Streamlit app and in a headless script that dumps PNGs for
the deck.

Palette: a single warm accent against neutral imagery. One accent, used
consistently, reads as a system. Three accents read as a demo.
"""

from __future__ import annotations

from typing import Any

from PIL import Image, ImageDraw

from contracts import Answer, Evidence, EvidenceKind

# --------------------------------------------------------------------------
# Palette - one accent, three weights
# --------------------------------------------------------------------------

ACCENT = (255, 92, 0)             # region outlines, strongest evidence
ACCENT_DIM = (255, 158, 92)       # weaker evidence
INK = (24, 24, 27)                # label backgrounds
PAPER = (250, 250, 249)           # label text

HEAT_LOW = (255, 92, 0, 0)        # transparent where nothing changed
HEAT_HIGH = (255, 92, 0, 105)     # translucent - imagery must stay readable


def _as_pil(image: Any) -> Image.Image:
    """Accept PIL or a 0-1 / 0-255 array."""
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    import numpy as np

    arr = np.asarray(image)
    if arr.dtype != np.uint8:
        arr = (np.clip(arr, 0, 1) * 255).astype("uint8")
    return Image.fromarray(arr).convert("RGB")


def _ascii(text: str) -> str:
    """PIL's default bitmap font has no em-dash and renders it as a box.

    Every label goes through here. Learned the hard way from a screenshot
    with tofu glyphs in it.
    """
    return (text.replace("\u2014", "-").replace("\u2013", "-")
                .replace("\u2019", "'").replace("\u201c", '"')
                .replace("\u201d", '"').encode("ascii", "replace").decode())


def _overlaps(a: tuple[float, float, float, float],
              b: tuple[float, float, float, float]) -> bool:
    return not (a[2] <= b[0] or b[2] <= a[0] or a[3] <= b[1] or b[3] <= a[1])


def _label(draw: ImageDraw.ImageDraw, xy: tuple[float, float], text: str,
           scale: int = 1, bounds: tuple[int, int] | None = None,
           taken: list | None = None) -> bool:
    """Text on a solid chip. Clipped to the canvas, and skipped entirely if
    it would collide with a chip already drawn.

    Returns whether it was drawn. Overlapping chips are unreadable, and an
    unreadable label is worse than no label - the box already shows where.
    """
    if not text:
        return False
    text = _ascii(text)
    x, y = xy
    pad = 3 * scale
    try:
        tb = draw.textbbox((0, 0), text)
        tw, th = tb[2] - tb[0], tb[3] - tb[1]
    except Exception:
        tw, th = len(text) * 6 * scale, 11 * scale

    w_chip, h_chip = tw + pad * 2, th + pad * 2
    if bounds:
        x = max(0, min(x, bounds[0] - w_chip))
        y = max(0, min(y, bounds[1] - h_chip))

    chip = (x, y, x + w_chip, y + h_chip)
    if taken is not None:
        if any(_overlaps(chip, t) for t in taken):
            return False
        taken.append(chip)

    draw.rectangle(list(chip), fill=INK)
    draw.text((x + pad, y + pad), text, fill=PAPER)
    return True


# --------------------------------------------------------------------------
# Overlays
# --------------------------------------------------------------------------

def draw_regions(image: Any, evidence: list[Evidence],
                 show_labels: bool = True, max_labels: int = 3) -> Image.Image:
    """Outline every REGION evidence box. Strongest gets the full accent."""
    canvas = _as_pil(image).copy()
    draw = ImageDraw.Draw(canvas)
    w, h = canvas.size
    scale = max(1, w // 400)

    regions = [e for e in evidence if e.kind is EvidenceKind.REGION and e.bbox]
    if not regions:
        return canvas

    strongest = max((e.score for e in regions), default=0.0)
    labelled: set[str] = set()
    taken: list = []
    drawn = 0

    # Strongest first, so if several boxes share a note the label lands on
    # the most significant one, and weaker boxes lose the collision.
    for ev in sorted(regions, key=lambda e: -e.score):
        x0, y0, x1, y1 = ev.bbox
        box = (x0 * w, y0 * h, x1 * w, y1 * h)
        primary = ev.score >= strongest and strongest > 0
        draw.rectangle(box, outline=ACCENT if primary else ACCENT_DIM,
                       width=max(2, w // 200))

        if (show_labels and ev.note and ev.note not in labelled
                and drawn < max_labels):
            if _label(draw, (box[0] + 4, box[1] + 4), ev.note, scale,
                      canvas.size, taken):
                labelled.add(ev.note)
                drawn += 1

    return canvas


def draw_heatmap(image: Any, evidence: list[Evidence],
                 opacity: float = 1.0) -> Image.Image:
    """Blend the first HEATMAP grid over the image.

    The grid is small (8x8) and gets resampled up bicubically, which gives a
    soft field rather than visible blocks - reads as an analysis product
    instead of a debug view.
    """
    canvas = _as_pil(image).convert("RGBA")
    w, h = canvas.size

    heat = next((e for e in evidence
                 if e.kind is EvidenceKind.HEATMAP and e.grid), None)
    if heat is None:
        return canvas.convert("RGB")

    rows = len(heat.grid)
    cols = len(heat.grid[0]) if rows else 0
    if not rows or not cols:
        return canvas.convert("RGB")

    layer = Image.new("RGBA", (cols, rows))
    px = layer.load()
    for y in range(rows):
        for x in range(cols):
            v = max(0.0, min(1.0, float(heat.grid[y][x]))) * opacity
            px[x, y] = (
                int(HEAT_LOW[0] + (HEAT_HIGH[0] - HEAT_LOW[0]) * v),
                int(HEAT_LOW[1] + (HEAT_HIGH[1] - HEAT_LOW[1]) * v),
                int(HEAT_LOW[2] + (HEAT_HIGH[2] - HEAT_LOW[2]) * v),
                int(HEAT_LOW[3] + (HEAT_HIGH[3] - HEAT_LOW[3]) * v),
            )

    layer = layer.resize((w, h), Image.BICUBIC)
    return Image.alpha_composite(canvas, layer).convert("RGB")


def draw_change(before: Any, after: Any, evidence: list[Evidence],
                gap: int = 12) -> Image.Image:
    """Before | after-with-heatmap-and-boxes, side by side.

    This single image is the week-3 demo. Put it in the deck as a static
    screenshot too, in case the live demo misbehaves.
    """
    left = _as_pil(before)
    right = draw_heatmap(after, evidence)
    right = draw_regions(right, evidence)

    h = max(left.size[1], right.size[1])
    left = left.resize((int(left.size[0] * h / left.size[1]), h))
    right = right.resize((int(right.size[0] * h / right.size[1]), h))

    total_w = left.size[0] + gap + right.size[0]
    strip = Image.new("RGB", (total_w, h + 22), PAPER)
    strip.paste(left, (0, 22))
    strip.paste(right, (left.size[0] + gap, 22))

    draw = ImageDraw.Draw(strip)
    _label(draw, (2, 2), "before")
    _label(draw, (left.size[0] + gap + 2, 2), "after — change highlighted")
    return strip


# --------------------------------------------------------------------------
# Entry point used by the app
# --------------------------------------------------------------------------

def render_answer(images: list[Any], answer: Answer,
                  mode: str = "auto") -> dict[str, Image.Image]:
    """Build every overlay this answer supports.

    Returns {caption: image} so the UI can lay them out without knowing which
    evidence kinds were produced.

    mode: "auto" | "regions" | "heatmap" | "change"
    """
    out: dict[str, Image.Image] = {}
    ev = answer.evidence
    if not images or not ev:
        return out

    has_heat = any(e.kind is EvidenceKind.HEATMAP and e.grid for e in ev)
    is_pair = len(images) >= 2 and any(e.image_index == 1 for e in ev)

    if mode == "change" or (mode == "auto" and is_pair):
        out["Before and after"] = draw_change(images[0], images[1], ev)
        return out

    for idx in sorted({e.image_index for e in ev if e.image_index < len(images)}):
        img = images[idx]
        subset = [e for e in ev if e.image_index == idx]
        if mode in ("heatmap", "auto") and has_heat:
            img = draw_heatmap(img, subset)
        if mode in ("regions", "auto", "heatmap"):
            img = draw_regions(img, subset)
        out[f"Image {idx + 1}"] = _as_pil(img)

    return out
