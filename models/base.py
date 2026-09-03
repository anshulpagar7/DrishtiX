"""
The SpecialistModel interface.

Every model - stub, CLIP, fine-tuned VLM, change detector - implements this.
The router never knows which is which. That is what lets you swap a stub for
a real model in week 2 without touching the pipeline.

Contract for implementers:
  - can_handle() must be cheap. No model loading, no network.
  - Load weights lazily inside run(), never in __init__, so importing the
    registry stays fast and the UI starts instantly.
  - Always return evidence. An answer with no evidence is a hallucination
    with good manners.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any

from contracts import ModelOutput, TaskSpec, TaskType


class SpecialistModel(ABC):
    """Base class for anything the router can dispatch to."""

    #: Unique, stable, appears in the UI and in evaluation tables.
    model_id: str = "base"

    #: Which tasks this model claims to handle.
    supported_tasks: set[TaskType] = set()

    #: Higher wins when several models can handle the same spec.
    #: Stubs use 0. Real models use 10+. Fine-tuned models use 20+.
    priority: int = 0

    def __init__(self) -> None:
        self._loaded = False

    # -- capability --------------------------------------------------------

    def can_handle(self, spec: TaskSpec) -> bool:
        """Cheap check. Override to add modality or class constraints."""
        return spec.task_type in self.supported_tasks

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    # -- execution ---------------------------------------------------------

    @abstractmethod
    def _run(self, spec: TaskSpec, images: list[Any]) -> ModelOutput:
        """Do the actual work. Implement this, not run()."""

    def run(self, spec: TaskSpec, images: list[Any]) -> ModelOutput:
        """Timed wrapper. The pipeline calls this."""
        t0 = time.perf_counter()
        out = self._run(spec, images)
        out.latency_ms = (time.perf_counter() - t0) * 1000
        out.model_id = out.model_id or self.model_id
        return out

    def __repr__(self) -> str:  # pragma: no cover
        return f"<{type(self).__name__} {self.model_id}>"
