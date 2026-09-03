"""
Model registry and router.

Capability-based dispatch: models declare what they can handle, the router
picks the best available one. Adding a capability means registering a model,
never editing routing logic. That property is your scalability slide - say it
out loud in the pitch.
"""

from __future__ import annotations

from contracts import TaskSpec
from models.base import SpecialistModel


class ModelRegistry:
    """Holds every available specialist and routes TaskSpecs to them."""

    def __init__(self) -> None:
        self._models: list[SpecialistModel] = []

    def register(self, model: SpecialistModel) -> "ModelRegistry":
        """Add a model. Returns self so registrations can chain."""
        if any(m.model_id == model.model_id for m in self._models):
            raise ValueError(f"Duplicate model_id: {model.model_id}")
        self._models.append(model)
        return self

    @property
    def models(self) -> list[SpecialistModel]:
        return list(self._models)

    def candidates(self, spec: TaskSpec) -> list[SpecialistModel]:
        """Every model that can handle this spec, best first.

        Ranking: higher self-reported priority wins. Ties broken by
        registration order, so a real model registered after a stub
        should declare a higher priority.
        """
        able = [m for m in self._models if m.can_handle(spec)]
        return sorted(able, key=lambda m: -m.priority)

    def route(self, spec: TaskSpec) -> SpecialistModel | None:
        """Pick one model, or None if nothing can handle the spec."""
        c = self.candidates(spec)
        return c[0] if c else None

    def describe(self) -> list[dict[str, object]]:
        """Registry contents, for the UI sidebar and the architecture slide."""
        return [
            {
                "model_id": m.model_id,
                "tasks": sorted(t.value for t in m.supported_tasks),
                "priority": m.priority,
                "loaded": m.is_loaded,
            }
            for m in self._models
        ]
