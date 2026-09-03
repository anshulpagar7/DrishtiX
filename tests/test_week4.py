"""
Week 4 tests: the integration seam and the HTTP layer.

The seam is the risky part of this build. Two modules were written by
different people against a JSON contract, and the failure mode is not a
crash — it is a bounding box drawn in the wrong place, or a demo that
silently answers with baselines while the deck claims a fine-tuned model.
Both are tested here.

Run:  python -m tests.test_week4
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PIL import Image                                        # noqa: E402

from contracts import EvidenceKind, TaskSpec, TaskType       # noqa: E402
from demo.samples import build                               # noqa: E402
from models.rsvqa import RSVQAModel                          # noqa: E402


# --------------------------------------------------------------------------
# Coordinate translation — the silent-failure case
# --------------------------------------------------------------------------

def test_pixel_bbox_becomes_normalised() -> None:
    """[x, y, w, h] pixels -> (x0, y0, x1, y1) in 0-1.

    Getting this wrong does not raise. It draws the box somewhere else,
    which is worse than an exception because the demo still 'works'.
    """
    regions = [{"bbox": [50, 100, 150, 50], "label": "buildings"}]
    ev = RSVQAModel._to_evidence(regions, (200, 200))
    assert len(ev) == 1
    x0, y0, x1, y1 = ev[0].bbox
    assert (x0, y0, x1, y1) == (0.25, 0.5, 1.0, 0.75)
    assert ev[0].note == "buildings"
    assert ev[0].kind is EvidenceKind.REGION
    print("  bbox pixels -> normalised       ok")


def test_bbox_is_clamped_to_frame() -> None:
    """A model can return a box that runs off the edge. Clamp, don't crash."""
    ev = RSVQAModel._to_evidence([{"bbox": [-20, -10, 400, 400]}], (200, 200))
    x0, y0, x1, y1 = ev[0].bbox
    assert x0 == 0.0 and y0 == 0.0
    assert x1 <= 1.0 and y1 <= 1.0
    print("  bbox clamped to frame           ok")


def test_malformed_regions_are_dropped() -> None:
    bad = [{"bbox": [1, 2]}, {"label": "no bbox"}, {}, {"bbox": None}]
    assert RSVQAModel._to_evidence(bad, (100, 100)) == []
    print("  malformed regions dropped       ok")


def test_zero_size_image_is_survivable() -> None:
    assert RSVQAModel._to_evidence([{"bbox": [0, 0, 1, 1]}], (0, 0)) == []
    print("  zero-size image survivable      ok")


# --------------------------------------------------------------------------
# Availability — the "is it actually loaded" case
# --------------------------------------------------------------------------

def test_missing_adapter_is_not_fatal() -> None:
    m = RSVQAModel(adapter_dir="does/not/exist")
    assert m.adapter_present is False
    assert m.ensure_loaded() is False
    assert m.is_loaded is False
    assert "No adapter" in m.load_error
    print("  missing adapter degrades        ok")


def test_unloaded_model_never_wins_a_route() -> None:
    """An absent model claiming a task and then failing is the worst case."""
    m = RSVQAModel(adapter_dir="does/not/exist")
    m.ensure_loaded()
    spec = TaskSpec(raw_query="Where is the water?", task_type=TaskType.GROUND)
    assert m.can_handle(spec) is False
    print("  unloaded model cannot route     ok")


def test_adapter_detection_requires_config() -> None:
    """A directory alone is not an adapter — adapter_config.json makes it one."""
    import tempfile

    with tempfile.TemporaryDirectory() as d:
        m = RSVQAModel(adapter_dir=d)
        assert m.adapter_present is False
        (Path(d) / "adapter_config.json").write_text("{}")
        assert RSVQAModel(adapter_dir=d).adapter_present is True
    print("  adapter detection is strict     ok")


def test_priority_outranks_baselines() -> None:
    from models.heuristic import default_models

    top = max(m.priority for m in default_models())
    assert RSVQAModel.priority > top
    print("  trained model outranks baseline ok")


def test_contract_shape_is_translated() -> None:
    """Feed the exact JSON shape training/scripts/inference.py returns and
    check it becomes a well-formed ModelOutput."""
    m = RSVQAModel(adapter_dir="does/not/exist")
    m._loaded = True                       # bypass the real model
    m._run_vqa = lambda image_path, query, adapter_dir: {
        "task": "grounding",
        "answer": "Found the requested region.",
        "confidence": 0.87,
        "regions": [{"bbox": [120, 45, 140, 135], "label": "building cluster"}],
        "mask_path": None,
        "metadata": {"model": "rs-llava-lora-v1", "task_type": "grounding"},
    }

    img = Image.new("RGB", (300, 300), "green")
    spec = TaskSpec(raw_query="Where are the buildings?", task_type=TaskType.GROUND)
    out = m.run(spec, [img])

    assert out.answer == "Found the requested region."
    assert out.confidence == 0.87
    assert out.model_id == "rs-llava-lora-v1"
    assert len(out.evidence) == 1
    x0, y0, x1, y1 = out.evidence[0].bbox
    assert abs(x0 - 0.4) < 1e-6 and abs(y0 - 0.15) < 1e-6
    assert all(0.0 <= v <= 1.0 for v in out.evidence[0].bbox)
    print("  full contract translation       ok")


def test_empty_regions_become_whole_image_evidence() -> None:
    m = RSVQAModel(adapter_dir="does/not/exist")
    m._loaded = True
    m._run_vqa = lambda **k: {
        "task": "vqa", "answer": "About 40% farmland.", "confidence": 0.6,
        "regions": [], "mask_path": None, "metadata": {},
    }
    out = m.run(TaskSpec(raw_query="How much farmland?", task_type=TaskType.VQA),
                [Image.new("RGB", (100, 100))])
    assert out.evidence and out.evidence[0].kind is EvidenceKind.WHOLE
    print("  no regions -> whole-image ev    ok")


def test_inference_exception_is_contained() -> None:
    """Their code raising must not take the whole answer down."""
    m = RSVQAModel(adapter_dir="does/not/exist")
    m._loaded = True

    def boom(**kwargs):
        raise RuntimeError("CUDA out of memory")

    m._run_vqa = boom
    out = m.run(TaskSpec(raw_query="anything", task_type=TaskType.VQA),
                [Image.new("RGB", (64, 64))])
    assert out.confidence == 0.0
    assert "CUDA out of memory" in out.answer
    print("  inference exception contained   ok")


# --------------------------------------------------------------------------
# HTTP layer
# --------------------------------------------------------------------------

def _client():
    from fastapi.testclient import TestClient

    from api.main import app
    return TestClient(app)


def test_health_reports_adapter_truthfully() -> None:
    """A demo that implies a trained model while running baselines is how
    teams get caught. Health must never overstate."""
    h = _client().get("/api/health").json()
    assert h["status"] == "ok"
    assert isinstance(h["rsvqa_adapter_present"], bool)
    if not h["rsvqa_adapter_present"]:
        assert "rs-vqa-lora-v1" not in h["models_loaded"]
    print("  health does not overstate       ok")


def test_ask_scene_answers() -> None:
    r = _client().post("/api/ask/scene", data={
        "query": "What changed between these two images?",
        "scene": "deforestation", "pair": "true"})
    d = r.json()
    assert r.status_code == 200
    assert d["answered"] is True
    assert d["task"] == "change"
    assert d["route"] and d["plan"]["models"]
    assert d["overlays"]
    print("  POST /api/ask/scene             ok")


def test_ask_scene_refuses_with_one_image() -> None:
    d = _client().post("/api/ask/scene", data={
        "query": "What changed here?", "scene": "flood",
        "pair": "false"}).json()
    assert d["answered"] is False
    assert d["refusal_reason"] and d["fix_hint"]
    assert d["route"] == []
    print("  refusal survives the API        ok")


def test_ask_upload_roundtrip() -> None:
    before, _, _, _ = build("urban_growth", 192)
    buf = io.BytesIO()
    before.save(buf, "PNG")
    buf.seek(0)
    d = _client().post("/api/ask", data={"query": "Where is the water?"},
                       files={"images": ("t.png", buf, "image/png")}).json()
    assert d["answered"] is True
    boxes = [e["bbox"] for e in d["evidence"] if e["bbox"]]
    assert boxes and all(0.0 <= v <= 1.0 for v in boxes[0])
    print("  POST /api/ask upload            ok")


def test_empty_query_rejected() -> None:
    r = _client().post("/api/ask/scene", data={
        "query": "   ", "scene": "flood"})
    assert r.status_code == 400
    print("  empty query rejected            ok")


def test_unknown_scene_404s() -> None:
    r = _client().post("/api/ask/scene", data={"query": "x", "scene": "atlantis"})
    assert r.status_code == 404
    print("  unknown scene 404               ok")


def test_bad_upload_rejected() -> None:
    r = _client().post("/api/ask", data={"query": "describe this"},
                       files={"images": ("x.png", io.BytesIO(b"not an image"),
                                         "image/png")})
    assert r.status_code == 400
    print("  non-image upload rejected       ok")


def test_pages_are_served() -> None:
    c = _client()
    for page in ["/", "/console.html", "/system.html", "/data.html",
                 "/status.html"]:
        assert c.get(page).status_code == 200, page
    print("  all 5 pages served              ok")


def test_page_route_cannot_escape_site_dir() -> None:
    """The page route takes a name from the URL. Containment is checked."""
    r = _client().get("/..%2f..%2fconfig.html")
    assert r.status_code == 404
    print("  page route is contained         ok")


def main() -> None:
    print("\nWeek 4 tests — integration seam + API")
    print("-" * 46)
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("-" * 46)
    print("all passed\n")


if __name__ == "__main__":
    main()
