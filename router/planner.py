"""
The planner: TaskSpec -> ExecutionPlan.

This is the file the pitch is actually about. The PS asks for an *agentic*
assistant, and the difference between agentic and "a wrapper around a VLM" is
exactly this: something decides what to run, says why, and can run more than
one thing and combine the results.

Three properties to point at when an evaluator asks what the contribution is:

  1. The plan is produced BEFORE anything executes, and is inspectable. The
     UI renders it. "Why did it do that" has a printed answer.
  2. Plans can be multi-model. A change query runs a localiser AND a
     describer, and the answers get fused.
  3. Nothing here hardcodes model names. It asks the registry what is
     available and plans against capability. Register a new model and plans
     start using it with no edit to this file.
"""

from __future__ import annotations

from contracts import ExecutionPlan, FusionStrategy, TaskSpec, TaskType
from router.registry import ModelRegistry


class Planner:
    """Decides the execution plan for a parsed query."""

    #: Tasks worth running more than one model on. Change is the obvious one:
    #: localising the change and describing it are different jobs.
    MULTI_MODEL_TASKS = {TaskType.CHANGE}

    #: Never dispatch more than this many models to one query. A plan that
    #: runs six models is slow on stage and nobody reads six answers.
    MAX_MODELS = 3

    def __init__(self, registry: ModelRegistry, max_models: int | None = None) -> None:
        self.registry = registry
        self.max_models = max_models or self.MAX_MODELS

    # ----------------------------------------------------------------------

    def plan(self, spec: TaskSpec) -> ExecutionPlan:
        candidates = self.registry.candidates(spec)

        if not candidates:
            return ExecutionPlan(
                model_ids=[],
                fusion=FusionStrategy.BEST,
                rationale=f"No model registered for a '{spec.task_type.value}' query.",
            )

        # Single-model tasks: take the highest-priority candidate.
        if spec.task_type not in self.MULTI_MODEL_TASKS or len(candidates) == 1:
            top = candidates[0]
            return ExecutionPlan(
                model_ids=[top.model_id],
                fusion=FusionStrategy.BEST,
                rationale=(f"{spec.task_type.value} query; {top.model_id} is the "
                           f"highest-priority model that handles it."),
            )

        # Multi-model tasks: run the top few and fuse.
        chosen = candidates[:self.max_models]
        ids = [m.model_id for m in chosen]
        return ExecutionPlan(
            model_ids=ids,
            fusion=FusionStrategy.CONCAT,
            rationale=(f"Change query; running {len(ids)} models so the answer "
                       f"carries both localisation and description: "
                       f"{', '.join(ids)}."),
        )

    # ----------------------------------------------------------------------

    def explain(self, spec: TaskSpec) -> str:
        """Human-readable plan, for the UI and for a screenshot in the deck."""
        p = self.plan(spec)
        if not p.model_ids:
            return p.rationale
        return (f"{p.rationale}\nFusion: {p.fusion.value}.")
