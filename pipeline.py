"""
SatQuery pipeline: query + images -> grounded Answer.

Six stages, in order:
    1. parse      natural language      -> TaskSpec
    2. validate   TaskSpec + images     -> ValidationResult   (may refuse)
    3. plan       TaskSpec + registry   -> ExecutionPlan      (inspectable)
    4. execute    plan                  -> [ModelOutput]      (may be several)
    5. fuse       outputs               -> text + confidence
    6. ground     evidence              -> Answer

Week 3 split routing into plan-then-execute. The plan is produced before
anything runs and is attached to the Answer, so "why did it do that" has a
printed answer rather than a shrug. That separation is the agentic claim.

Refusal is a first-class outcome, not an error. Answer.answered says which
happened, and the UI renders each differently.
"""

from __future__ import annotations

import time
from typing import Any

from contracts import Answer, ExecutionPlan, ModelOutput, TaskSpec, ValidationResult
from models.change import SemanticChangeDetector
from models.heuristic import default_models
from router.fusion import fuse
from router.parser import Parser, RuleParser, build_parser
from router.planner import Planner
from router.registry import ModelRegistry
from router.validator import validate


def build_registry(with_vlm: bool = False) -> ModelRegistry:
    """Default registry.

    The heuristics are ALWAYS registered, at priority 0. They are the floor
    the router falls back to when the VLM is absent or fails to load - which
    is what keeps a live demo alive.
    """
    reg = ModelRegistry()
    for m in default_models():
        reg.register(m)

    reg.register(SemanticChangeDetector())      # priority 10, above pixel diff

    if with_vlm:
        # Person 1's trained RS-VQA adapter, if it exists. Highest priority.
        from models.rsvqa import RSVQAModel

        rsvqa = RSVQAModel()
        if rsvqa.ensure_loaded():
            reg.register(rsvqa)
        else:
            print(f"[registry] rs-vqa unavailable: {rsvqa.load_error}")

        # Generic VLM slot, for a checkpoint trained in this repo.
        from models.vlm import VLMModel

        vlm = VLMModel()
        if vlm.ensure_loaded():
            reg.register(vlm)
        else:
            print(f"[registry] vlm unavailable: {vlm.load_error}")

    return reg


def build_pipeline(with_vlm: bool = False,
                   prefer_llm: bool = True) -> "SatQueryPipeline":
    """One call the app and eval harness both use.

    Degrades cleanly: no LLM backend configured -> rule parser; no GPU or no
    adapter -> heuristics only. Neither case raises.
    """
    return SatQueryPipeline(build_parser(prefer_llm), build_registry(with_vlm))


class SatQueryPipeline:
    def __init__(self, parser: Parser | None = None,
                 registry: ModelRegistry | None = None,
                 planner: Planner | None = None) -> None:
        self.parser = parser or RuleParser()
        self.registry = registry or build_registry()
        self.planner = planner or Planner(self.registry)

    # ----------------------------------------------------------------------

    def ask(self, query: str, images: list[Any]) -> Answer:
        t0 = time.perf_counter()

        # 1. parse
        spec: TaskSpec = self.parser.parse(query, n_images=len(images))

        # 2. validate
        sizes = [self._size_of(im) for im in images]
        result: ValidationResult = validate(spec, sizes)
        if not result.ok:
            return self._refuse(spec, result, t0)

        # 3. plan
        plan: ExecutionPlan = self.planner.plan(spec)
        if not plan.model_ids:
            miss = ValidationResult(
                ok=False,
                reason=plan.rationale,
                fix_hint="Register a specialist for this task, or rephrase.",
            )
            return self._refuse(spec, miss, t0, plan)

        # 4. execute - one model failing must not sink the whole answer
        outputs: list[ModelOutput] = []
        ran: list[str] = []
        errors: list[str] = []

        by_id = {m.model_id: m for m in self.registry.models}
        for mid in plan.model_ids:
            model = by_id.get(mid)
            if model is None:
                continue
            try:
                outputs.append(model.run(spec, images))
                ran.append(mid)
            except Exception as exc:
                errors.append(f"{mid}: {type(exc).__name__}")

        if not outputs:
            fail = ValidationResult(
                ok=False,
                reason="Every planned model failed while running.",
                fix_hint="; ".join(errors)[:200],
            )
            return self._refuse(spec, fail, t0, plan)

        # 5. fuse
        text, confidence, evidence = fuse(outputs, plan.fusion)
        if errors:
            text += f"\n\n({len(errors)} model(s) failed: {', '.join(errors)})"

        # 6. ground
        answer = Answer(
            answered=True,
            text=text,
            spec=spec,
            validation=result,
            outputs=outputs,
            route=ran,
            plan=plan,
            total_latency_ms=(time.perf_counter() - t0) * 1000,
        )
        answer._fused_confidence = confidence     # type: ignore[attr-defined]
        answer._fused_evidence = evidence         # type: ignore[attr-defined]
        return answer

    # ----------------------------------------------------------------------

    def _refuse(self, spec: TaskSpec, result: ValidationResult,
                t0: float, plan: ExecutionPlan | None = None) -> Answer:
        return Answer(
            answered=False,
            text=f"{result.reason} {result.fix_hint}".strip(),
            spec=spec,
            validation=result,
            plan=plan,
            total_latency_ms=(time.perf_counter() - t0) * 1000,
        )

    @staticmethod
    def _size_of(image: Any) -> tuple[int, int]:
        """(width, height) from a PIL image or an HxWxC array."""
        if hasattr(image, "size") and isinstance(image.size, tuple):
            return image.size                          # PIL: (w, h)
        shape = getattr(image, "shape", None)
        if shape and len(shape) >= 2:
            return int(shape[1]), int(shape[0])        # numpy: (h, w, c)
        return (0, 0)
