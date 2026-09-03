"""
Synthetic sample scenes.

Two reasons this exists, and neither is laziness:

  1. The demo must work with no network, no dataset, no Kaggle. Someone will
     open your laptop on stage and the venue wifi will be broken.
  2. These are the only image pairs where you know the ground truth exactly,
     so they are how you test whether the change detector localises correctly
     rather than eyeballing a Sentinel tile and hoping.

Real Sentinel imagery goes in the pitch. These go in the test suite and in
the fallback demo.

Run:  python -m demo.samples --out demo/scenes
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

RNG = np.random.default_rng(7)

# Base cover colours in 0-1 RGB, tuned so _cover_scores classifies them
# the way a human would label them.
COVER = {
    "vegetation": (0.18, 0.52, 0.20),
    "water": (0.10, 0.16, 0.55),
    "urban": (0.58, 0.58, 0.60),
    "bare": (0.62, 0.50, 0.34),
    "snow": (0.92, 0.93, 0.95),
    "scar": (0.55, 0.40, 0.28),      # scorched ground - clears the bare-soil test
}


def _texture(shape: tuple[int, int], strength: float = 0.05) -> np.ndarray:
    """Low-frequency noise so tiles look like imagery, not swatches."""
    small = RNG.normal(0, strength, (max(shape[0] // 16, 2),
                                     max(shape[1] // 16, 2), 3))
    # np.ptp(x), not x.ptp() - the method was removed in numpy 2.0.
    img = Image.fromarray(((small - small.min()) /
                           (np.ptp(small) + 1e-6) * 255).astype("uint8"))
    img = img.resize((shape[1], shape[0]), Image.BICUBIC)
    return (np.asarray(img).astype(np.float32) / 255.0 - 0.5) * strength * 2


def tile(size: int = 256, base: str = "vegetation") -> np.ndarray:
    arr = np.zeros((size, size, 3), dtype=np.float32)
    arr[:] = COVER[base]
    arr += _texture((size, size))
    return np.clip(arr, 0, 1)


def patch(arr: np.ndarray, cover: str, box: tuple[float, float, float, float]
          ) -> np.ndarray:
    """Paint a normalised box with a cover type."""
    out = arr.copy()
    h, w = out.shape[:2]
    x0, y0, x1, y1 = (int(box[0] * w), int(box[1] * h),
                      int(box[2] * w), int(box[3] * h))
    region = np.zeros((y1 - y0, x1 - x0, 3), dtype=np.float32)
    region[:] = COVER[cover]
    region += _texture((y1 - y0, x1 - x0), 0.04)
    out[y0:y1, x0:x1] = np.clip(region, 0, 1)
    return out


def band(arr: np.ndarray, cover: str, y0_frac: float, y1_frac: float,
         drift: float = 0.0, width_px: int | None = None,
         x0_frac: float = 0.0, x1_frac: float = 1.0) -> np.ndarray:
    """Paint a diagonal band — a road or a river channel, not a blob.

    y0_frac/y1_frac give the vertical position where the band ENTERS its
    x-range, normalised. drift shifts that position linearly across the
    range, so the band can run at an angle instead of dead horizontal —
    every other scene here is an axis-aligned box, and a real linear feature
    almost never is.

    x0_frac/x1_frac confine the band to part of the tile's width. Left at
    the default (0, 1) it spans edge to edge, which - for two DIFFERENT
    bands drawn across the full width, as a channel relocating does - can
    leave so much of the tile touched that the co-registration check reads
    it as two unrelated images. A bounded reach is also more realistic: a
    real river avulsion does not usually redraw itself across an entire
    scene edge to edge.

    width_px must be wide enough to dominate at least part of a change-
    detection grid cell, or the pixel-difference layer sees the feature
    while the semantic layer's per-cell dominant-class check never flips.
    """
    out = arr.copy()
    h, w = out.shape[:2]
    wpx = width_px or max(int(h * (y1_frac - y0_frac)), 4)
    colour = np.array(COVER[cover], dtype=np.float32)
    xa, xb = int(x0_frac * w), int(x1_frac * w)

    for x in range(max(xa, 0), min(xb, w)):
        t = (x - xa) / max(xb - xa - 1, 1)
        centre = (y0_frac + drift * t) * h
        y0 = max(int(centre - wpx / 2), 0)
        y1 = min(int(centre + wpx / 2), h)
        if y1 <= y0:
            continue
        seg = np.tile(colour, (y1 - y0, 1))
        seg += _texture((y1 - y0, 1), 0.03).reshape(-1, 1, 3).repeat(1, axis=1)[:, 0]
        out[y0:y1, x] = np.clip(seg, 0, 1)
    return out


# --------------------------------------------------------------------------
# Scenes - each returns (before, after, ground_truth_box, description)
# --------------------------------------------------------------------------

def scene_urban_growth(size: int = 256):
    before = tile(size, "vegetation")
    before = patch(before, "urban", (0.05, 0.05, 0.30, 0.30))
    after = patch(before, "urban", (0.55, 0.55, 0.90, 0.90))
    return before, after, (0.55, 0.55, 0.90, 0.90), "new development in the south-east"


def scene_deforestation(size: int = 256):
    before = tile(size, "vegetation")
    after = patch(before, "bare", (0.10, 0.55, 0.45, 0.92))
    return before, after, (0.10, 0.55, 0.45, 0.92), "forest cleared in the south-west"


def scene_flood(size: int = 256):
    before = tile(size, "bare")
    before = patch(before, "vegetation", (0.0, 0.0, 1.0, 0.35))
    after = patch(before, "water", (0.15, 0.40, 0.85, 0.80))
    return before, after, (0.15, 0.40, 0.85, 0.80), "flooding across the centre"


def scene_glacier_retreat(size: int = 256):
    before = tile(size, "snow")
    before = patch(before, "bare", (0.0, 0.80, 1.0, 1.0))
    after = patch(before, "bare", (0.0, 0.55, 1.0, 1.0))
    return before, after, (0.0, 0.55, 1.0, 0.80), "ice margin retreated northward"


def scene_no_change(size: int = 256):
    before = tile(size, "vegetation")
    before = patch(before, "water", (0.30, 0.30, 0.70, 0.70))
    after = before + _texture((size, size), 0.015)      # sensor noise only
    return before, np.clip(after, 0, 1), None, "no real change, noise only"


def scene_different_places(size: int = 256):
    """The trap case. Two unrelated tiles - the detector must refuse."""
    before = tile(size, "vegetation")
    before = patch(before, "water", (0.1, 0.1, 0.4, 0.4))
    after = tile(size, "urban")
    after = patch(after, "bare", (0.6, 0.6, 0.95, 0.95))
    return before, after, None, "two different places, should be refused"


def scene_coastal_erosion(size: int = 256):
    """The shoreline retreats: land becomes water along one edge."""
    before = tile(size, "bare")
    before = patch(before, "vegetation", (0.0, 0.0, 1.0, 0.55))
    before = patch(before, "water", (0.0, 0.80, 1.0, 1.0))
    after = patch(before, "water", (0.0, 0.62, 1.0, 1.0))
    return before, after, (0.0, 0.62, 1.0, 0.85), "coastline receded in the south"


def scene_wildfire_scar(size: int = 256):
    """A burn scar appears in standing forest."""
    before = tile(size, "vegetation")
    after = patch(before, "scar", (0.30, 0.20, 0.78, 0.68))
    return before, after, (0.30, 0.20, 0.78, 0.68), "burn scar in the centre"


def scene_port_expansion(size: int = 256):
    """Reclaimed land: a harbour extends out into open water."""
    before = tile(size, "water")
    before = patch(before, "urban", (0.0, 0.0, 1.0, 0.30))
    after = patch(before, "urban", (0.0, 0.0, 1.0, 0.52))
    return before, after, (0.0, 0.30, 1.0, 0.52), "port reclaimed land from the harbour"


def scene_drought_reservoir(size: int = 256):
    """A reservoir shrinks from within, exposing lakebed - not an edge crop."""
    before = tile(size, "bare")
    before = patch(before, "water", (0.15, 0.15, 0.85, 0.85))
    after = patch(before, "bare", (0.32, 0.32, 0.68, 0.68))
    return before, after, (0.32, 0.32, 0.68, 0.68), "reservoir drawn down, exposing lakebed"


def scene_new_highway(size: int = 256):
    """A linear feature, not a blob - tests localisation on a thin diagonal.

    Width is set relative to the change-detection grid, not just visually -
    a road narrower than roughly a third of a grid cell never dominates
    that cell's average and the semantic layer misses it even though the
    pixel-difference layer does not.
    """
    before = tile(size, "vegetation")
    after = band(before, "urban", y0_frac=0.18, y1_frac=0.18,
                drift=0.55, width_px=max(size // 9, 18))
    return before, after, (0.0, 0.10, 1.0, 0.82), "new highway cut across the tile"


def scene_river_meander(size: int = 256):
    """The channel relocates rather than growing or shrinking in place -
    change appears in two places at once: new water, and a drying old bed.

    Confined to a bounded reach (not edge to edge) for two reasons: it is
    what a real avulsion looks like, and two full-width bands here trip the
    co-registration check - between the drying bed and the new channel,
    enough of the frame changes that it reads as two different places.
    """
    before = tile(size, "vegetation")
    before = band(before, "water", y0_frac=0.32, y1_frac=0.32, drift=0.30,
                 width_px=max(size // 11, 9), x0_frac=0.12, x1_frac=0.72)
    after = tile(size, "vegetation")
    after = band(after, "bare", y0_frac=0.32, y1_frac=0.32, drift=0.30,       # old bed dries
                width_px=max(size // 15, 6), x0_frac=0.12, x1_frac=0.72)
    after = band(after, "water", y0_frac=0.56, y1_frac=0.56, drift=0.26,     # new channel
                width_px=max(size // 11, 9), x0_frac=0.20, x1_frac=0.80)
    return before, after, (0.12, 0.28, 0.80, 0.68), "river avulsed to a new channel south"


SCENES = {
    "urban_growth": scene_urban_growth,
    "deforestation": scene_deforestation,
    "flood": scene_flood,
    "glacier_retreat": scene_glacier_retreat,
    "coastal_erosion": scene_coastal_erosion,
    "wildfire_scar": scene_wildfire_scar,
    "port_expansion": scene_port_expansion,
    "drought_reservoir": scene_drought_reservoir,
    "new_highway": scene_new_highway,
    "river_meander": scene_river_meander,
    "no_change": scene_no_change,
    "different_places": scene_different_places,
}


def as_pil(arr: np.ndarray) -> Image.Image:
    return Image.fromarray((np.clip(arr, 0, 1) * 255).astype("uint8"))


def build(name: str, size: int = 256):
    """Returns (before_pil, after_pil, gt_box, description)."""
    if name not in SCENES:
        raise KeyError(f"Unknown scene '{name}'. Known: {list(SCENES)}")
    before, after, box, desc = SCENES[name](size)
    return as_pil(before), as_pil(after), box, desc


def main() -> None:
    ap = argparse.ArgumentParser(description="Write synthetic demo scenes.")
    ap.add_argument("--out", type=Path, default=Path("demo/scenes"))
    ap.add_argument("--size", type=int, default=256)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    for name in SCENES:
        b, a, box, desc = build(name, args.size)
        b.save(args.out / f"{name}_before.png")
        a.save(args.out / f"{name}_after.png")
        print(f"{name:<18} {desc}")
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
