"""
API response shapes.

These mirror contracts.py rather than inventing a second vocabulary. When the
Answer dataclass gains a field, it gains one here too — one concept, one name,
across Python, HTTP and the front end.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class PlanOut(BaseModel):
    models: list[str] = Field(default_factory=list)
    fusion: str = "best"
    rationale: str = ""


class OutputOut(BaseModel):
    model_id: str
    answer: str
    confidence: float = 0.0
    latency_ms: float = 0.0
    veto: bool = False
    scores: dict[str, float] = Field(default_factory=dict)


class EvidenceOut(BaseModel):
    image_index: int = 0
    kind: str = "region"
    #: Normalised (x0, y0, x1, y1). Never pixels — see models/rsvqa.py.
    bbox: list[float] | None = None
    score: float = 0.0
    note: str = ""


class AnswerOut(BaseModel):
    answered: bool
    text: str
    task: str = "unknown"
    modality: str = "any"
    needs_pair: bool = False
    parser: str = "unknown"
    parser_confidence: float = 0.0
    #: Set only on a refusal, so the UI can style it differently.
    refusal_reason: str | None = None
    fix_hint: str | None = None
    plan: PlanOut | None = None
    route: list[str] = Field(default_factory=list)
    outputs: list[OutputOut] = Field(default_factory=list)
    evidence: list[EvidenceOut] = Field(default_factory=list)
    #: caption -> PNG data URI, rendered server-side so the client does no
    #: geometry. One source of truth for where a box goes.
    overlays: dict[str, str] = Field(default_factory=dict)
    latency_ms: float = 0.0


class ModelOut(BaseModel):
    model_id: str
    tasks: list[str]
    priority: int
    loaded: bool


class SceneOut(BaseModel):
    name: str
    description: str
    before: str
    after: str


class HealthOut(BaseModel):
    status: str
    parser: str
    llm_backend: str
    rsvqa_adapter_present: bool
    rsvqa_status: str
    models_loaded: list[str]
    models_registered: list[str]
    scenes: list[str]
