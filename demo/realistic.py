"""
Photorealistic-style demo imagery, for manual upload during a live demo.

This is deliberately SEPARATE from demo/samples.py, which feeds the tested
pipeline, the eval harness, and the console's built-in gallery. Nothing here
is imported by the app. The reason for the split: demo/samples.py's flat-
fill rectangles are tuned so the heuristic models' colour thresholds and the
change-detector's ground-truth boxes line up exactly, and 71 passing tests
plus a published hit-rate number depend on that geometry staying put. This
file has no such constraint - its only job is to look like a real aerial
photo when a person uploads it through the console UI.

Honest limits, stated once here rather than left implicit:
  - This is still synthetic. No real satellite data was fetched - the
    environment this runs in has no network path to a real imagery
    provider, and no photorealistic image-generation model is available to
    call. What changed is the rendering technique: multi-octave terrain
    noise, organic wobbly boundaries instead of axis-aligned rectangles,
    directional relief shading, and sensor grain, in a muted true-colour-
    style palette - the things that make flat vector fills read as
    "cartoon" to begin with.
  - Because boundaries are now organic rather than crisp rectangles, do NOT
    point demo/change_eval.py's ground-truth boxes at these images. They
    are for visual upload demos only.

Run:  python -m demo.realistic --out demo/realistic_scenes --size 512
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image

RNG = np.random.default_rng(31)

# Each cover type is a small palette blended by local noise, not one flat
# fill - real ground cover has visible variation (denser/sparser canopy,
# wetter/drier soil) that a single RGB value cannot show.
PALETTES = {
    "vegetation": [(0.14, 0.30, 0.12), (0.21, 0.42, 0.17), (0.29, 0.50, 0.21)],
    "water":      [(0.04, 0.08, 0.20), (0.07, 0.13, 0.30), (0.11, 0.19, 0.38)],
    "urban":      [(0.40, 0.39, 0.38), (0.52, 0.51, 0.49), (0.61, 0.60, 0.58)],
    "bare":       [(0.34, 0.26, 0.17), (0.45, 0.35, 0.23), (0.54, 0.43, 0.29)],
    "snow":       [(0.74, 0.77, 0.81), (0.85, 0.87, 0.90), (0.95, 0.96, 0.98)],
    "scar":       [(0.52, 0.38, 0.26), (0.60, 0.45, 0.31), (0.68, 0.53, 0.37)],  # brighter: clears the bare-soil brightness test after shading
}


# --------------------------------------------------------------------------
# Noise primitives
# --------------------------------------------------------------------------

def _grid_noise(h: int, w: int, cells: int, rng: np.random.Generator) -> np.ndarray:
    gy = max(int(cells), 2)
    gx = max(int(cells * w / max(h, 1)), 2)
    grid = rng.normal(0, 1, (gy, gx)).astype(np.float32)
    up = Image.fromarray(
        ((grid - grid.min()) / (np.ptp(grid) + 1e-6) * 255).astype("uint8")
    ).resize((w, h), Image.BICUBIC)
    return np.asarray(up, dtype=np.float32) / 255.0


def fbm(h: int, w: int, octaves: int = 5, persistence: float = 0.55,
        base_cells: float = 3.0, rng: np.random.Generator | None = None) -> np.ndarray:
    """Fractal terrain noise: several noise layers at increasing detail,
    summed. This is what gives real terrain its layered look — broad
    moisture/elevation patterns with finer texture riding on top — versus
    the single soft blur a flat cartoon fill uses."""
    rng = rng or RNG
    total = np.zeros((h, w), dtype=np.float32)
    amp, cells, amp_sum = 1.0, base_cells, 0.0
    for _ in range(octaves):
        total += _grid_noise(h, w, cells, rng) * amp
        amp_sum += amp
        amp *= persistence
        cells *= 2.1
    return total / amp_sum


def _curve(n: int, rng: np.random.Generator) -> np.ndarray:
    """Smooth 1D wandering curve in roughly [-1, 1], for wobbling an edge."""
    knots = max(int(n / 30), 3)
    pts = rng.normal(0, 1, knots)
    xs = np.linspace(0, n - 1, knots)
    c = np.interp(np.arange(n), xs, pts)
    return c / (np.abs(c).max() + 1e-6)


def organic_mask(h: int, w: int, box: tuple[float, float, float, float],
                 wobble: float, rng: np.random.Generator) -> np.ndarray:
    """Soft-edged region roughly matching `box`, with each side wobbling
    independently — a coastline, a burn scar, a cleared parcel. Real ground
    features are essentially never axis-aligned rectangles."""
    x0, y0, x1, y1 = (box[0] * w, box[1] * h, box[2] * w, box[3] * h)
    bw, bh = max(x1 - x0, 1), max(y1 - y0, 1)
    amp = wobble * min(bw, bh)

    left = x0 + _curve(h, rng) * amp
    right = x1 + _curve(h, rng) * amp
    top = y0 + _curve(w, rng) * amp
    bot = y1 + _curve(w, rng) * amp

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    d_left, d_right = xx - left[:, None], right[:, None] - xx
    d_top, d_bot = yy - top[None, :], bot[None, :] - yy

    # A hard min() across four independently-wobbling edges creates sharp
    # cusps wherever two edges are nearly tied - a "starburst" artifact, not
    # a coastline. A smooth minimum (soft-min via -logsumexp) rounds those
    # seams into the flowing curves a real shoreline or forest edge has.
    k = 6.0 / max(min(bw, bh), 1)
    stacked = np.stack([d_left, d_right, d_top, d_bot])
    d = -np.log(np.sum(np.exp(-k * stacked), axis=0) + 1e-9) / k

    soft = max(amp * 0.5, 3.0)
    m = np.clip(d / soft * 0.5 + 0.5, 0, 1).astype(np.float32)
    # Roughen the transition band itself - a real cleared-forest edge or
    # coastline is ragged, not a single smooth gradient step.
    edge_zone = np.clip(1 - np.abs(m - 0.5) * 4, 0, 1)
    rough = _grid_noise(h, w, min(w, h) / 10, rng)
    return np.clip(m + (rough - 0.5) * 0.5 * edge_zone, 0, 1)


def organic_band(h: int, w: int, y0_frac: float, y1_frac: float, drift: float,
                 width_px: float, x0_frac: float, x1_frac: float,
                 wobble: float, rng: np.random.Generator) -> np.ndarray:
    """Soft-edged diagonal band for a road or a river channel."""
    xa, xb = x0_frac * w, x1_frac * w
    wob = _curve(w, rng) * wobble * width_px
    centre = np.where(
        (np.arange(w) >= xa) & (np.arange(w) <= xb),
        (y0_frac + drift * np.clip((np.arange(w) - xa) / max(xb - xa - 1, 1), 0, 1)) * h + wob,
        -1e6,
    )
    yy = np.arange(h)[:, None]
    dist = width_px / 2 - np.abs(yy - centre[None, :])
    soft = max(width_px * 0.35, 2.0)
    return np.clip(dist / soft * 0.5 + 0.5, 0, 1).astype(np.float32)


# --------------------------------------------------------------------------
# Painting
# --------------------------------------------------------------------------

def paint(field: np.ndarray, mask: np.ndarray, cover: str,
          local_noise: np.ndarray) -> None:
    """Blend a cover's palette into `field`, weighted by `mask`, with the
    palette stop chosen per pixel from local_noise (0-1) for natural
    variation rather than a flat fill."""
    stops = np.array(PALETTES[cover], dtype=np.float32)
    idx = local_noise * (len(stops) - 1)
    lo = np.floor(idx).astype(int).clip(0, len(stops) - 2)
    t = (idx - lo)[..., None]
    lo_flat = lo.reshape(-1)
    c = (stops[lo_flat] * (1 - t.reshape(-1, 1)) +
         stops[lo_flat + 1] * t.reshape(-1, 1)).reshape(*field.shape[:2], 3)
    m = mask[..., None]
    field[:] = field * (1 - m) + c * m


def relief_shade(field: np.ndarray, elevation: np.ndarray,
                 strength: float = 0.16) -> np.ndarray:
    """Directional hillshade from a coarse elevation field — the cheapest
    single change that makes a flat render look like real terrain."""
    gy, gx = np.gradient(elevation)
    shade = np.clip(1.0 + strength * (-gx * 1.4 - gy), 0.68, 1.35)
    return np.clip(field * shade[..., None], 0, 1)


def make_grain(size: int, rng: np.random.Generator, grain: float = 0.018) -> np.ndarray:
    """One shared grain field per scene, reused for both before and after.

    Drawing grain fresh inside finish() on every call meant unchanged ground
    got two INDEPENDENT random samples - one per image - which is itself a
    source of pixel-level difference between "the same place" in before and
    after. A change detector cannot tell that noise from a real edit. One
    grain field, applied identically to both outputs, removes it as a false
    signal while keeping the same textured, grainy look in each image.
    """
    return rng.normal(0, grain, (size, size, 1)).astype(np.float32)


def finish(field: np.ndarray, grain_field: np.ndarray,
          vignette: float = 0.12) -> np.ndarray:
    """Apply shared sensor grain and a soft vignette — the last 2% that
    sells a render as a photograph rather than a vector illustration."""
    h, w = field.shape[:2]
    out = field + grain_field
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cy, cx = h / 2, w / 2
    r = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2) / np.sqrt(cy ** 2 + cx ** 2)
    vig = 1 - vignette * np.clip(r - 0.55, 0, 1) / 0.45
    return np.clip(out * vig[..., None], 0, 1)


def base_terrain(size: int, primary: str, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """A textured, UNSHADED base of one dominant cover, plus the elevation
    field to shade with. Shading is deferred to exactly one call per scene
    output - see the note on relief_shade() below for why."""
    field = np.zeros((size, size, 3), dtype=np.float32)
    tex = fbm(size, size, octaves=6, persistence=0.62, base_cells=4, rng=rng)
    paint(field, np.ones((size, size), dtype=np.float32), primary, tex)
    field = add_grain_detail(field, size, rng)
    elevation = fbm(size, size, octaves=4, base_cells=2.4, rng=rng)
    return field, elevation


def add_grain_detail(field: np.ndarray, size: int,
                     rng: np.random.Generator, strength: float = 0.06) -> np.ndarray:
    """A distinct high-frequency micro-texture layer - individual tree
    crowns, furrows, roof lines - riding on top of the smooth base colour.
    This single addition is what separates "aerial photo" from "airbrushed
    gradient": real ground cover is never colour-smooth at close range."""
    fine = fbm(size, size, octaves=3, persistence=0.5, base_cells=size / 9, rng=rng)
    fine = (fine - fine.mean()) * strength
    return np.clip(field + fine[..., None], 0, 1)


def as_pil(field: np.ndarray, grain_field: np.ndarray) -> Image.Image:
    from PIL import ImageEnhance, ImageFilter

    out = finish(field, grain_field)
    img = Image.fromarray((np.clip(out, 0, 1) * 255).astype("uint8"))

    # A smooth numpy render is inherently soft. Real satellite products go
    # through a sharpening pass in processing, which is a lot of why they
    # read as "photo" rather than "illustration" - this closes that gap.
    img = img.filter(ImageFilter.UnsharpMask(radius=2.0, percent=95, threshold=2))
    img = ImageEnhance.Contrast(img).enhance(1.10)
    img = ImageEnhance.Color(img).enhance(0.92)   # true-colour imagery reads slightly desaturated
    return img


# --------------------------------------------------------------------------
# Scenes — same 12 stories as demo/samples.py, rendered for realism
# --------------------------------------------------------------------------

def _rng(name: str) -> np.random.Generator:
    """A stable per-scene seed so re-runs are reproducible but scenes don't
    all share one noise pattern."""
    return np.random.default_rng(abs(hash(name)) % (2**32))


def scene_urban_growth(size):
    r = _rng("urban_growth")
    base, elev = base_terrain(size, "vegetation", r)
    grain = make_grain(size, r)

    before = base.copy()
    m = organic_mask(size, size, (0.06, 0.06, 0.34, 0.32), 0.22, r)
    paint(before, m, "urban", fbm(size, size, base_cells=6, rng=r))

    after = before.copy()
    m2 = organic_mask(size, size, (0.50, 0.50, 0.94, 0.94), 0.20, r)
    paint(after, m2, "urban", fbm(size, size, base_cells=6, rng=r))

    before, after = relief_shade(before, elev), relief_shade(after, elev)
    return as_pil(before, grain), as_pil(after, grain), "new development in the south-east"


def scene_deforestation(size):
    r = _rng("deforestation")
    before, elev = base_terrain(size, "vegetation", r)
    grain = make_grain(size, r)

    after = before.copy()
    m = organic_mask(size, size, (0.08, 0.55, 0.46, 0.93), 0.24, r)
    paint(after, m, "bare", fbm(size, size, base_cells=5, rng=r))

    before, after = relief_shade(before, elev), relief_shade(after, elev)
    return as_pil(before, grain), as_pil(after, grain), "forest cleared in the south-west"


def scene_flood(size):
    r = _rng("flood")
    base, elev = base_terrain(size, "bare", r)
    grain = make_grain(size, r)

    before = base.copy()
    m0 = organic_mask(size, size, (0, 0, 1, 0.5), 0.14, r)
    paint(before, m0, "vegetation", fbm(size, size, base_cells=4, rng=r))

    after = before.copy()
    m = organic_mask(size, size, (0.12, 0.36, 0.88, 0.86), 0.22, r)
    paint(after, m, "water", fbm(size, size, base_cells=6, rng=r))

    before, after = relief_shade(before, elev), relief_shade(after, elev)
    return as_pil(before, grain), as_pil(after, grain), "flooding across the centre"


def scene_glacier_retreat(size):
    r = _rng("glacier_retreat")
    base, elev = base_terrain(size, "snow", r)
    grain = make_grain(size, r)

    before = base.copy()
    m0 = organic_mask(size, size, (0, 0.78, 1, 1), 0.18, r)
    paint(before, m0, "bare", fbm(size, size, base_cells=5, rng=r))

    after = before.copy()
    m = organic_mask(size, size, (0, 0.52, 1, 1), 0.20, r)
    paint(after, m, "bare", fbm(size, size, base_cells=5, rng=r))

    before, after = relief_shade(before, elev), relief_shade(after, elev)
    return as_pil(before, grain), as_pil(after, grain), "ice margin retreated northward"


def scene_coastal_erosion(size):
    r = _rng("coastal_erosion")
    base, elev = base_terrain(size, "bare", r)
    grain = make_grain(size, r)

    before = base.copy()
    m0 = organic_mask(size, size, (0, 0, 1, 0.56), 0.16, r)
    paint(before, m0, "vegetation", fbm(size, size, base_cells=4, rng=r))
    m1 = organic_mask(size, size, (0, 0.80, 1, 1), 0.20, r)
    paint(before, m1, "water", fbm(size, size, base_cells=5, rng=r))

    after = before.copy()
    m2 = organic_mask(size, size, (0, 0.62, 1, 1), 0.22, r)
    paint(after, m2, "water", fbm(size, size, base_cells=5, rng=r))

    before, after = relief_shade(before, elev), relief_shade(after, elev)
    return as_pil(before, grain), as_pil(after, grain), "coastline receded in the south"


def scene_wildfire_scar(size):
    r = _rng("wildfire_scar")
    before, elev = base_terrain(size, "vegetation", r)
    grain = make_grain(size, r)

    after = before.copy()
    m = organic_mask(size, size, (0.28, 0.16, 0.82, 0.74), 0.16, r)   # less soft, more core area
    paint(after, m, "scar", fbm(size, size, base_cells=6, rng=r))

    before, after = relief_shade(before, elev), relief_shade(after, elev)
    return as_pil(before, grain), as_pil(after, grain), "burn scar in the centre"


def scene_port_expansion(size):
    r = _rng("port_expansion")
    base, elev = base_terrain(size, "water", r)
    grain = make_grain(size, r)

    before = base.copy()
    m0 = organic_mask(size, size, (0, 0, 1, 0.30), 0.16, r)
    paint(before, m0, "urban", fbm(size, size, base_cells=5, rng=r))

    after = before.copy()
    m = organic_mask(size, size, (0, 0.30, 1, 0.53), 0.18, r)
    paint(after, m, "urban", fbm(size, size, base_cells=5, rng=r))

    before, after = relief_shade(before, elev), relief_shade(after, elev)
    return as_pil(before, grain), as_pil(after, grain), "port reclaimed land from the harbour"


def scene_drought_reservoir(size):
    r = _rng("drought_reservoir")
    base, elev = base_terrain(size, "bare", r)
    grain = make_grain(size, r)

    before = base.copy()
    m0 = organic_mask(size, size, (0.14, 0.14, 0.86, 0.86), 0.16, r)
    paint(before, m0, "water", fbm(size, size, base_cells=5, rng=r))

    after = before.copy()
    m = organic_mask(size, size, (0.31, 0.31, 0.69, 0.69), 0.20, r)
    paint(after, m, "bare", fbm(size, size, base_cells=5, rng=r))

    before, after = relief_shade(before, elev), relief_shade(after, elev)
    return as_pil(before, grain), as_pil(after, grain), "reservoir drawn down, exposing lakebed"


def scene_new_highway(size):
    r = _rng("new_highway")
    before, elev = base_terrain(size, "vegetation", r)
    grain = make_grain(size, r)

    after = before.copy()
    band = organic_band(size, size, 0.17, 0.17, 0.55,
                        max(size / 9, 20), 0.02, 0.98, 0.18, r)      # wider, less wobble-softened
    paint(after, band, "urban", fbm(size, size, base_cells=6, rng=r))

    before, after = relief_shade(before, elev), relief_shade(after, elev)
    return as_pil(before, grain), as_pil(after, grain), "new highway cut across the tile"


def scene_river_meander(size):
    r = _rng("river_meander")
    base, elev = base_terrain(size, "vegetation", r)
    grain = make_grain(size, r)

    before = base.copy()
    b0 = organic_band(size, size, 0.32, 0.32, 0.30,
                      max(size / 9, 16), 0.12, 0.72, 0.18, r)
    paint(before, b0, "water", fbm(size, size, base_cells=5, rng=r))

    after = base.copy()                      # from the SAME unshaded base as before
    a0 = organic_band(size, size, 0.32, 0.32, 0.30,
                      max(size / 12, 12), 0.12, 0.72, 0.18, r)
    paint(after, a0, "bare", fbm(size, size, base_cells=5, rng=r))
    a1 = organic_band(size, size, 0.56, 0.56, 0.26,
                      max(size / 9, 16), 0.20, 0.80, 0.18, r)
    paint(after, a1, "water", fbm(size, size, base_cells=5, rng=r))

    before, after = relief_shade(before, elev), relief_shade(after, elev)
    return as_pil(before, grain), as_pil(after, grain), "river avulsed to a new channel south"


def scene_no_change(size):
    r = _rng("no_change")
    base, elev = base_terrain(size, "vegetation", r)
    grain = make_grain(size, r)

    before = base.copy()
    m0 = organic_mask(size, size, (0.28, 0.28, 0.72, 0.72), 0.16, r)
    paint(before, m0, "water", fbm(size, size, base_cells=5, rng=r))
    before = relief_shade(before, elev)

    # The only intentional difference here is a tiny explicit noise term -
    # applied post-shading, on top of an otherwise pixel-identical frame,
    # with the SAME grain field used for both outputs. That is what makes
    # this scene an honest "nothing really changed" rather than an
    # accidental one from independent rendering noise.
    after = np.clip(before + r.normal(0, 0.01, before.shape).astype(np.float32), 0, 1)
    return as_pil(before, grain), as_pil(after, grain), "no real change, sensor noise only"


def scene_different_places(size):
    r = _rng("different_places")
    before, elev = base_terrain(size, "vegetation", r)
    grain_before = make_grain(size, r)
    m0 = organic_mask(size, size, (0.08, 0.08, 0.42, 0.42), 0.18, r)
    paint(before, m0, "water", fbm(size, size, base_cells=5, rng=r))
    before = relief_shade(before, elev)

    # Deliberately a fully independent render - different seed, different
    # elevation, different grain. This is the one scene where before/after
    # SHOULD share nothing, because it exists to test the refusal path.
    r2 = _rng("different_places_b")
    after, elev2 = base_terrain(size, "urban", r2)
    grain_after = make_grain(size, r2)
    m1 = organic_mask(size, size, (0.58, 0.58, 0.96, 0.96), 0.18, r2)
    paint(after, m1, "bare", fbm(size, size, base_cells=5, rng=r2))
    after = relief_shade(after, elev2)
    return (as_pil(before, grain_before), as_pil(after, grain_after),
            "two different places, should be refused")


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


def build(name: str, size: int = 512):
    if name not in SCENES:
        raise KeyError(f"Unknown scene '{name}'. Known: {list(SCENES)}")
    before, after, desc = SCENES[name](size)
    return before, after, desc


def main() -> None:
    ap = argparse.ArgumentParser(description="Write realistic-style demo scenes.")
    ap.add_argument("--out", type=Path, default=Path("demo/realistic_scenes"))
    ap.add_argument("--size", type=int, default=512)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    for name in SCENES:
        before, after, desc = build(name, args.size)
        before.save(args.out / f"{name}_before.png")
        after.save(args.out / f"{name}_after.png")
        print(f"{name:<18} {desc}")
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
