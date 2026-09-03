"""
Week 3 tests.

Covering the things week 3 added, and specifically the bugs rehearsal caught:

  - the co-registration check false-positiving on large genuine change
  - noise being reported as change
  - the same transition being narrated three times
  - a veto being ignored so nonsense got concatenated onto a refusal

Every one of those was a real regression found by running the demo, not by
reading the code. They stay here so they cannot come back.

Run:  python -m tests.test_week3
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from contracts import (Evidence, EvidenceKind, ExecutionPlan,   # noqa: E402
                       FusionStrategy, ModelOutput, TaskType)
from demo.samples import SCENES, build                          # noqa: E402
from demo.script import STEPS                                   # noqa: E402
from models.change import SemanticChangeDetector                # noqa: E402
from pipeline import SatQueryPipeline, build_registry           # noqa: E402
from render import draw_change, draw_heatmap, draw_regions, render_answer  # noqa: E402
from router.fusion import fuse                                  # noqa: E402
from router.parser import RuleParser                            # noqa: E402
from router.planner import Planner                              # noqa: E402

CHANGE_Q = "What changed between these two images?"


def _pipe() -> SatQueryPipeline:
    return SatQueryPipeline()


def _regions(ans) -> list[Evidence]:
    return [e for e in ans.evidence
            if e.kind is EvidenceKind.REGION and e.bbox is not None]


# --------------------------------------------------------------------------
# Planner
# --------------------------------------------------------------------------

def test_planner_single_model_for_simple_task() -> None:
    reg = build_registry()
    spec = RuleParser().parse("What land cover is in this image?")
    plan = Planner(reg).plan(spec)
    assert len(plan) == 1
    assert plan.fusion is FusionStrategy.BEST
    assert plan.rationale
    print("  planner: single model           ok")


def test_planner_multi_model_for_change() -> None:
    reg = build_registry()
    spec = RuleParser().parse(CHANGE_Q)
    plan = Planner(reg).plan(spec)
    assert len(plan) >= 2, "change should plan more than one model"
    assert plan.fusion is FusionStrategy.CONCAT
    print("  planner: multi model on change  ok")


def test_planner_respects_max_models() -> None:
    reg = build_registry()
    spec = RuleParser().parse(CHANGE_Q)
    plan = Planner(reg, max_models=1).plan(spec)
    assert len(plan) == 1
    print("  planner: max_models cap         ok")


def test_planner_empty_when_nothing_registered() -> None:
    from router.registry import ModelRegistry

    plan = Planner(ModelRegistry()).plan(RuleParser().parse("Describe this."))
    assert len(plan) == 0
    assert plan.rationale
    print("  planner: empty plan explained   ok")


# --------------------------------------------------------------------------
# Fusion
# --------------------------------------------------------------------------

def test_fuse_best_picks_highest_confidence() -> None:
    outs = [ModelOutput(answer="low", confidence=0.2, model_id="a"),
            ModelOutput(answer="high", confidence=0.9, model_id="b")]
    text, conf, _ = fuse(outs, FusionStrategy.BEST)
    assert text.startswith("high")
    assert conf == 0.9
    print("  fusion: best                    ok")


def test_fuse_concat_keeps_every_answer() -> None:
    outs = [ModelOutput(answer="first.", confidence=0.5, model_id="a"),
            ModelOutput(answer="second.", confidence=0.5, model_id="b")]
    text, _, _ = fuse(outs, FusionStrategy.CONCAT)
    assert "first." in text and "second." in text
    print("  fusion: concat                  ok")


def test_fuse_vote_reports_disagreement() -> None:
    outs = [ModelOutput(answer="water", confidence=0.8, model_id="a"),
            ModelOutput(answer="urban", confidence=0.7, model_id="b")]
    text, conf, _ = fuse(outs, FusionStrategy.VOTE)
    assert "disagree" in text.lower()
    assert conf < 0.8, "disagreement must lower confidence"
    print("  fusion: vote surfaces dissent   ok")


def test_veto_suppresses_other_outputs() -> None:
    """Regression: a veto used to be concatenated WITH the nonsense it vetoed."""
    outs = [ModelOutput(answer="inputs are unsound", confidence=0.9,
                        model_id="checker", veto=True),
            ModelOutput(answer="confident nonsense", confidence=0.8,
                        model_id="other")]
    text, _, _ = fuse(outs, FusionStrategy.CONCAT)
    assert "unsound" in text
    assert "nonsense" not in text, "vetoed output must not survive fusion"
    print("  fusion: veto suppresses others  ok")


def test_fuse_dedupes_identical_evidence() -> None:
    ev = Evidence(image_index=0, bbox=(0.1, 0.1, 0.2, 0.2))
    outs = [ModelOutput(answer="a", model_id="a", evidence=[ev]),
            ModelOutput(answer="b", model_id="b", evidence=[ev])]
    _, _, merged = fuse(outs, FusionStrategy.CONCAT)
    assert len(merged) == 1
    print("  fusion: evidence dedupe         ok")


# --------------------------------------------------------------------------
# Semantic change - the rehearsal bugs
# --------------------------------------------------------------------------

def test_large_genuine_change_is_not_vetoed() -> None:
    """Regression: deforestation changes ~35% of the tile, which collapsed
    luminance correlation and got misread as 'two different places'."""
    before, after, _, _ = build("deforestation")
    out = SemanticChangeDetector().run(RuleParser().parse(CHANGE_Q),
                                       [before, after])
    assert out.veto is False, "real change must not trigger the co-reg veto"
    assert "clearing" in out.answer.lower() or "vegetation" in out.answer.lower()
    print("  change: large change not vetoed ok")


def test_different_places_is_vetoed() -> None:
    before, after, _, _ = build("different_places")
    out = SemanticChangeDetector().run(RuleParser().parse(CHANGE_Q),
                                       [before, after])
    assert out.veto is True
    assert "same place" in out.answer.lower()
    print("  change: different places vetoed ok")


def test_opposing_local_changes_not_vetoed() -> None:
    """Regression: river_meander has two changes in opposite directions (a
    channel drying in one place, a new one appearing elsewhere). Whole-image
    luminance correlation swung sharply negative from that alone - correlation
    is dominated by whichever pixels carry variance, and two small opposing
    changes can outweigh a mostly-identical frame. That falsely vetoed a
    same-place pair even though the primary stable-fraction signal (~80%)
    was comfortably fine. Correlation is now diagnostic only; stable_fraction
    alone gates the veto."""
    before, after, _, _ = build("river_meander")
    out = SemanticChangeDetector().run(RuleParser().parse(CHANGE_Q),
                                       [before, after])
    assert out.veto is False, "opposing local changes must not read as a place mismatch"
    assert out.scores.get("stable_fraction", 0) > 0.5
    print("  change: opposing changes not vetoed ok")


def test_noise_is_not_reported_as_change() -> None:
    """Regression: a purely relative threshold fired on sensor noise."""
    ans = _pipe().ask(CHANGE_Q, list(build("no_change")[:2]))
    assert ans.answered is True
    assert not _regions(ans), "noise must not produce change regions"
    print("  change: noise reports nothing   ok")


def test_transitions_are_deduped() -> None:
    """Regression: one transition used to be narrated once per grid cell."""
    ans = _pipe().ask(CHANGE_Q, list(build("urban_growth")[:2]))
    text = ans.text.lower()
    assert text.count("vegetation cleared for development") <= 1, \
        "the same transition must be reported once"
    print("  change: transitions deduped     ok")


def test_change_emits_heatmap_and_regions() -> None:
    ans = _pipe().ask(CHANGE_Q, list(build("flood")[:2]))
    kinds = {e.kind for e in ans.evidence}
    assert EvidenceKind.HEATMAP in kinds
    assert EvidenceKind.REGION in kinds
    print("  change: heatmap + regions       ok")


def test_heatmap_grid_is_normalised() -> None:
    ans = _pipe().ask(CHANGE_Q, list(build("flood")[:2]))
    heat = next(e for e in ans.evidence if e.kind is EvidenceKind.HEATMAP)
    flat = [v for row in heat.grid for v in row]
    assert flat and all(0.0 <= v <= 1.0 for v in flat)
    assert max(flat) == 1.0, "heatmap should be normalised to its peak"
    print("  change: heatmap normalised      ok")


def test_change_localises_correctly_on_every_scene() -> None:
    from eval.change_eval import evaluate

    s = evaluate(size=192)["summary"]
    assert s["positive_correct"] == s["positive_total"], "missed a change"
    assert s["negative_correct"] == s["negative_total"], "false positive"
    print("  change: all scenes localise     ok")


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------

def test_render_regions_returns_same_size() -> None:
    before, after, _, _ = build("urban_growth", 192)
    ans = _pipe().ask(CHANGE_Q, [before, after])
    out = draw_regions(after, ans.evidence)
    assert out.size == after.size
    print("  render: regions preserve size   ok")


def test_render_heatmap_changes_pixels() -> None:
    before, after, _, _ = build("flood", 192)
    ans = _pipe().ask(CHANGE_Q, [before, after])
    out = draw_heatmap(after, ans.evidence)
    # np comparison avoids Pillow 14 getdata deprecation
    import numpy as _np
    assert not _np.array_equal(_np.asarray(out), _np.asarray(after.convert("RGB"))), \
        "heatmap overlay did not alter the image"
    print("  render: heatmap alters image    ok")


def test_render_change_strip_is_wider_than_one_image() -> None:
    before, after, _, _ = build("flood", 192)
    ans = _pipe().ask(CHANGE_Q, [before, after])
    strip = draw_change(before, after, ans.evidence)
    assert strip.size[0] > before.size[0] * 1.8
    print("  render: side-by-side strip      ok")


def test_render_handles_no_evidence() -> None:
    before, _, _, _ = build("urban_growth", 192)
    ans = _pipe().ask("What changed between these two images?", [before])
    assert ans.answered is False
    assert render_answer([before], ans) == {}
    print("  render: empty on refusal        ok")


def test_render_labels_are_ascii_safe() -> None:
    from render import _ascii

    assert "\u2014" not in _ascii("before \u2014 after")
    assert _ascii("caf\u00e9").isascii()
    print("  render: ascii label fallback    ok")


# --------------------------------------------------------------------------
# Demo integrity
# --------------------------------------------------------------------------

def test_every_demo_scene_builds() -> None:
    for name in SCENES:
        b, a, _, desc = build(name, 128)
        assert b.size == a.size == (128, 128)
        assert desc
    print(f"  demo: {len(SCENES)} scenes build       ok")


def test_demo_script_steps_behave_as_scripted() -> None:
    """The scripted demo must not contain a step that quietly misbehaves.

    Steps 3 and 6 are supposed to decline; everything else must answer.
    """
    pipe = _pipe()
    expect_refusal = {3, 6}
    for i, step in enumerate(STEPS, start=1):
        before, after, _, _ = build(step["scene"], 192)
        images = [before, after][:step["images"]]
        ans = pipe.ask(step["query"], images)
        if i in expect_refusal:
            declined = (not ans.answered) or any(o.veto for o in ans.outputs)
            assert declined, f"step {i} should decline, answered instead"
        else:
            assert ans.answered, f"step {i} should answer, refused instead"
            assert not any(o.veto for o in ans.outputs), \
                f"step {i} was vetoed unexpectedly"
    print(f"  demo: {len(STEPS)} scripted steps ok     ok")


def test_plan_is_attached_to_answers() -> None:
    ans = _pipe().ask(CHANGE_Q, list(build("flood")[:2]))
    assert isinstance(ans.plan, ExecutionPlan)
    assert ans.plan.model_ids and ans.plan.rationale
    assert ans.route, "route should record what actually ran"
    print("  plan attached to answer         ok")


def test_one_model_failing_does_not_sink_the_answer() -> None:
    """A model that raises must be reported, not propagated."""
    from models.base import SpecialistModel

    class Exploding(SpecialistModel):
        model_id = "exploding-v1"
        supported_tasks = {TaskType.CHANGE}
        priority = 99

        def _run(self, spec, images):
            raise RuntimeError("boom")

    reg = build_registry().register(Exploding())
    pipe = SatQueryPipeline(RuleParser(), reg, Planner(reg))
    ans = pipe.ask(CHANGE_Q, list(build("flood")[:2]))
    assert ans.answered is True, "one bad model should not kill the answer"
    assert "exploding-v1" in ans.text
    print("  one model failing is survivable ok")


def main() -> None:
    print("\nWeek 3 tests")
    print("-" * 44)
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
    print("-" * 44)
    print("all passed\n")


if __name__ == "__main__":
    main()
