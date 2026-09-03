"""
Contract tests. These guard the interfaces six people depend on.

Run:  python -m tests.test_contract
(or `pytest` once someone adds it - these are plain asserts either way.)

If a test here fails, someone changed a shared shape without telling the team.
That is the failure this file exists to catch.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np                                          # noqa: E402

from contracts import (Answer, Evidence, EvidenceKind,     # noqa: E402
                       ModelOutput, TaskType)
from models.heuristic import default_models                 # noqa: E402
from pipeline import SatQueryPipeline                       # noqa: E402
from router.parser import RuleParser                        # noqa: E402
from router.registry import ModelRegistry                   # noqa: E402


def _tile(kind: str = "mixed", size: int = 128) -> np.ndarray:
    """Synthetic tiles so tests need no dataset download."""
    rng = np.random.default_rng(0)
    a = np.zeros((size, size, 3), dtype=np.float32)
    if kind == "water":
        a[..., 2] = 0.6
        a[..., :2] = 0.15
    elif kind == "vegetation":
        a[..., 1] = 0.55
        a[..., 0] = 0.2
        a[..., 2] = 0.2
    else:
        half = size // 2
        a[:half, :, 2] = 0.6
        a[:half, :, :2] = 0.15
        a[half:, :, 1] = 0.55
        a[half:, :, 0] = 0.2
        a[half:, :, 2] = 0.2
    return np.clip(a + rng.normal(0, 0.02, a.shape), 0, 1).astype(np.float32)


def test_parser_task_types() -> None:
    p = RuleParser()
    assert p.parse("What changed between these images?").task_type is TaskType.CHANGE
    assert p.parse("Where is the water?").task_type is TaskType.GROUND
    assert p.parse("Describe this scene.").task_type is TaskType.CAPTION
    assert p.parse("What land cover is here?").task_type is TaskType.CLASSIFY
    assert p.parse("Is there a river?").task_type is TaskType.VQA
    print("  parser task types            ok")


def test_parser_sets_pair_flag() -> None:
    spec = RuleParser().parse("What changed between 2020 and 2024?")
    assert spec.needs_pair is True
    assert spec.temporal is True
    print("  parser pair/temporal flags   ok")


def test_registry_rejects_duplicates() -> None:
    reg = ModelRegistry()
    models = default_models()
    reg.register(models[0])
    try:
        reg.register(models[0])
    except ValueError:
        print("  registry duplicate guard     ok")
        return
    raise AssertionError("registry accepted a duplicate model_id")


def test_registry_prefers_priority() -> None:
    reg = ModelRegistry()
    for m in default_models():
        reg.register(m)
    spec = RuleParser().parse("What land cover is here?")
    chosen = reg.route(spec)
    assert chosen is not None and chosen.model_id == "heuristic-cover-v1"
    print("  registry routing             ok")


def test_refuses_change_query_with_one_image() -> None:
    ans: Answer = SatQueryPipeline().ask("What changed here?", [_tile()])
    assert ans.answered is False
    assert "only one image" in ans.text.lower()
    assert ans.validation is not None and ans.validation.fix_hint
    print("  refuses single-image change  ok")


def test_refuses_with_no_image() -> None:
    ans = SatQueryPipeline().ask("Describe this scene.", [])
    assert ans.answered is False
    print("  refuses empty input          ok")


def _regions(ans: Answer) -> list[Evidence]:
    """Week 3 added heatmap evidence, so evidence[0] is no longer always a
    box. Filter by kind rather than assuming an index."""
    return [e for e in ans.evidence
            if e.kind is EvidenceKind.REGION and e.bbox is not None]


def test_answers_carry_evidence() -> None:
    ans = SatQueryPipeline().ask("Where is the water?", [_tile("mixed")])
    assert ans.answered is True
    assert ans.evidence, "answer had no evidence"
    regions = _regions(ans)
    assert regions, "answer had no region evidence"
    for ev in regions:
        assert all(0.0 <= v <= 1.0 for v in ev.bbox), "bbox must be normalised"
    print("  evidence present + normalised ok")


def test_change_detection_finds_difference() -> None:
    before, after = _tile("vegetation"), _tile("vegetation").copy()
    after[:40, :40] = [0.7, 0.7, 0.7]          # a built patch appears
    ans = SatQueryPipeline().ask("What changed between these images?",
                                 [before, after])
    assert ans.answered is True
    regions = _regions(ans)
    assert regions, "change detector returned no region evidence"
    # Any reported region must fall in the top-left quadrant where the
    # synthetic change was painted.
    assert any(r.bbox[0] < 0.5 and r.bbox[1] < 0.5 for r in regions), \
        "change should localise to the top-left"
    print("  change localisation          ok")


def test_model_output_defaults() -> None:
    out = ModelOutput(answer="x")
    assert out.evidence == [] and out.scores == {}
    print("  ModelOutput defaults         ok")


def main() -> None:
    print("\nContract tests")
    print("-" * 40)
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("-" * 40)
    print("all passed\n")


if __name__ == "__main__":
    main()
