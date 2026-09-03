"""
SatQuery AI - the frozen contract.

THIS FILE IS THE TEAM AGREEMENT. Everything else plugs into these types.

Rules:
  1. Nobody changes this file alone. Changes need the whole team to agree,
     because every module depends on these shapes.
  2. Router produces a TaskSpec. Models consume a TaskSpec and produce
     ModelOutput. The pipeline turns those into an Answer.
  3. If you need a new field, add it with a default so old code keeps working.

Once this is frozen, six people can build in parallel without blocking.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any


# --------------------------------------------------------------------------
# Task taxonomy - the capabilities SatQuery can route to.
# --------------------------------------------------------------------------

class TaskType(str, Enum):
    """What the user is asking the system to do."""

    CLASSIFY = "classify"          # multi-label land cover  (BigEarthNet)
    VQA = "vqa"                    # open question about one image (RSVQA)
    CAPTION = "caption"            # describe the scene (VRSBench)
    GROUND = "ground"              # locate a named object (VRSBench)
    CHANGE = "change"              # what changed between two images (CDVQA)
    UNKNOWN = "unknown"            # parser could not decide


class Modality(str, Enum):
    """Sensor type of the imagery."""

    OPTICAL = "optical"            # Sentinel-2 / RGB
    SAR = "sar"                    # Sentinel-1
    ANY = "any"


# --------------------------------------------------------------------------
# Router output
# --------------------------------------------------------------------------

@dataclass
class TaskSpec:
    """Structured form of the user's natural-language question.

    Produced by router.parser. Consumed by router.validator, router.registry
    and every SpecialistModel.
    """

    raw_query: str
    task_type: TaskType = TaskType.UNKNOWN
    modality: Modality = Modality.ANY
    needs_pair: bool = False               # requires two co-registered images
    temporal: bool = False                 # question is about time
    target_class: str | None = None        # e.g. "water", "buildings"
    confidence: float = 0.0                # parser's confidence, 0-1
    parser_id: str = "unset"               # which parser produced this

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["task_type"] = self.task_type.value
        d["modality"] = self.modality.value
        return d


# --------------------------------------------------------------------------
# Validator output
# --------------------------------------------------------------------------

@dataclass
class ValidationResult:
    """Can the supplied images actually answer this question?

    This is a headline feature, not plumbing. A system that refuses cleanly
    reads as engineering. A system that always answers reads as a demo.
    """

    ok: bool
    reason: str = ""                       # user-facing, plain language
    fix_hint: str = ""                     # what the user should do instead


# --------------------------------------------------------------------------
# Model output
# --------------------------------------------------------------------------

class EvidenceKind(str, Enum):
    """How a piece of evidence should be drawn."""

    REGION = "region"          # a bounding box
    HEATMAP = "heatmap"        # a per-cell intensity grid
    WHOLE = "whole"            # applies to the entire image


@dataclass
class Evidence:
    """Where in the image the answer came from.

    bbox is (x0, y0, x1, y1) in normalised 0-1 coordinates so it survives
    resizing. image_index says which supplied image it refers to.

    Week 3 added kind, score and grid (all defaulted, so week-1 and week-2
    code that constructs Evidence(image_index=..., bbox=..., note=...) keeps
    working unchanged). grid is a row-major 2D list of 0-1 intensities used
    when kind is HEATMAP.
    """

    image_index: int = 0
    bbox: tuple[float, float, float, float] | None = None
    note: str = ""
    kind: EvidenceKind = EvidenceKind.REGION
    score: float = 0.0
    grid: list[list[float]] | None = None


@dataclass
class ModelOutput:
    """What a SpecialistModel returns. Never a bare string."""

    answer: str
    confidence: float = 0.0
    model_id: str = "unknown"
    evidence: list[Evidence] = field(default_factory=list)
    scores: dict[str, float] = field(default_factory=dict)   # per-class scores
    latency_ms: float = 0.0
    #: Week 3. When True this output invalidates the others in the same plan.
    #: Used when a model detects that the INPUTS are unsound (e.g. two tiles
    #: of different places), where every other model's confident output is
    #: meaningless and must not be shown alongside.
    veto: bool = False


# --------------------------------------------------------------------------
# Planner output - the agentic layer
# --------------------------------------------------------------------------

class FusionStrategy(str, Enum):
    """How several model outputs get combined into one answer."""

    BEST = "best"              # highest confidence wins, others kept as support
    CONCAT = "concat"          # narrate every output in order
    VOTE = "vote"              # agreement across models raises confidence


@dataclass
class ExecutionPlan:
    """What the router decided to do, before anything runs.

    Surfacing the plan separately from the result is what makes the system
    inspectable rather than a black box - the UI shows it, and an evaluator
    asking "why did it do that" gets an answer.
    """

    model_ids: list[str] = field(default_factory=list)
    fusion: FusionStrategy = FusionStrategy.BEST
    rationale: str = ""

    def __len__(self) -> int:
        return len(self.model_ids)


# --------------------------------------------------------------------------
# Pipeline output - what the UI renders
# --------------------------------------------------------------------------

@dataclass
class Answer:
    """Final grounded response. Either answered or refused, never both."""

    answered: bool
    text: str
    spec: TaskSpec | None = None
    validation: ValidationResult | None = None
    outputs: list[ModelOutput] = field(default_factory=list)
    route: list[str] = field(default_factory=list)   # model_ids that ran
    plan: ExecutionPlan | None = None                # what was decided, pre-run
    total_latency_ms: float = 0.0

    @property
    def evidence(self) -> list[Evidence]:
        """All evidence across every model that ran."""
        return [e for o in self.outputs for e in o.evidence]
