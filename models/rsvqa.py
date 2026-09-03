"""
Adapter for the trained remote-sensing VQA model.

This is the seam between two modules that were built separately:

  Person 1 (training/) exposes exactly one function, `run_vqa(image_path,
  query, adapter_dir)`, returning a fixed JSON shape. Their README is explicit
  that the agent side should treat it as a black box — swap RS-LLaVA for
  GeoChat, retrain, change ranks, and the signature does not move.

  This repo speaks SpecialistModel / TaskSpec / ModelOutput / Evidence.

So this file translates, and nothing else does. If the contract on either side
changes, exactly one file needs editing.

Two translation details that matter and are easy to get silently wrong:

  1. COORDINATES. run_vqa returns `bbox: [x, y, w, h]` in PIXELS. Evidence.bbox
     is `(x0, y0, x1, y1)` NORMALISED to 0-1. Getting this wrong does not
     raise — it draws a box in the wrong place, which is worse. Converted in
     `_to_evidence`, and unit-tested.

  2. FILE HANDOFF. run_vqa takes a PATH, not a PIL image. The pipeline passes
     images in memory, so anything not already on disk is written to a temp
     file and cleaned up after.

The model is not required. If the adapter directory is absent — which it is
until Person 1's first training run lands — `can_handle` returns False, the
router falls through to the classical baselines, and the system keeps working.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

import config
from contracts import Evidence, EvidenceKind, ModelOutput, TaskSpec, TaskType
from models.base import SpecialistModel

#: Where Person 1's training pipeline writes adapters.
DEFAULT_ADAPTER_DIR = Path(
    os.getenv("SATQUERY_RSVQA_ADAPTER", "training/outputs/lora_vqa_v1")
)

#: Their inference.py resolves configs/model.yaml relative to CWD, so calls
#: are made with the training directory as the working directory.
TRAINING_ROOT = Path(os.getenv("SATQUERY_TRAINING_ROOT", "training"))


class RSVQAModel(SpecialistModel):
    """Wraps `training/scripts/inference.py::run_vqa`."""

    model_id = "rs-vqa-lora-v1"
    supported_tasks = {TaskType.VQA, TaskType.GROUND, TaskType.CAPTION}
    priority = 30                     # outranks every classical baseline

    def __init__(self, adapter_dir: Path | str | None = None,
                 eager: bool = False) -> None:
        super().__init__()
        self.adapter_dir = Path(adapter_dir or DEFAULT_ADAPTER_DIR)
        self._run_vqa = None
        self._load_failed = False
        self.load_error: str = ""
        if eager:
            self.ensure_loaded()

    # -- availability ------------------------------------------------------

    @property
    def adapter_present(self) -> bool:
        """A LoRA directory needs an adapter_config.json to be real.

        Checked before importing torch, so a missing model costs nothing.
        """
        return (self.adapter_dir / "adapter_config.json").is_file()

    def ensure_loaded(self) -> bool:
        """Import and warm the model once. Returns whether it is usable."""
        if self._loaded:
            return True
        if self._load_failed:
            return False

        if not self.adapter_present:
            self._load_failed = True
            self.load_error = (
                f"No adapter at {self.adapter_dir}. Train one with "
                f"training/scripts/train_lora.py, or point "
                f"SATQUERY_RSVQA_ADAPTER at an existing directory."
            )
            return False

        try:
            import sys

            root = TRAINING_ROOT.resolve()
            if str(root) not in sys.path:
                sys.path.insert(0, str(root))
            from scripts.inference import run_vqa      # type: ignore

            self._run_vqa = run_vqa
            self._loaded = True
            return True
        except Exception as exc:
            self._load_failed = True
            self.load_error = f"{type(exc).__name__}: {exc}"[:300]
            return False

    def can_handle(self, spec: TaskSpec) -> bool:
        """Cheap, and never triggers a load — the router calls this per query."""
        return self._loaded and spec.task_type in self.supported_tasks

    @property
    def status(self) -> str:
        if self._loaded:
            return f"loaded ({self.adapter_dir})"
        if self._load_failed:
            return f"unavailable - {self.load_error}"
        return "not loaded"

    # -- translation -------------------------------------------------------

    @staticmethod
    def _to_evidence(regions: list[dict], size: tuple[int, int]) -> list[Evidence]:
        """[x, y, w, h] pixels  ->  (x0, y0, x1, y1) normalised.

        A silent failure here draws boxes in the wrong place rather than
        raising, so it is covered by tests.
        """
        w, h = size
        if not w or not h:
            return []

        out: list[Evidence] = []
        for r in regions or []:
            box = r.get("bbox")
            if not box or len(box) != 4:
                continue
            x, y, bw, bh = (float(v) for v in box)
            out.append(Evidence(
                image_index=0,
                kind=EvidenceKind.REGION,
                bbox=(
                    max(x / w, 0.0),
                    max(y / h, 0.0),
                    min((x + bw) / w, 1.0),
                    min((y + bh) / h, 1.0),
                ),
                score=float(r.get("score", 0.0)),
                note=str(r.get("label", "region")),
            ))
        return out

    @staticmethod
    def _image_path(image: Any) -> tuple[str, bool]:
        """(path, is_temporary). run_vqa needs a file, not a PIL object."""
        existing = getattr(image, "filename", None)
        if existing and Path(existing).is_file():
            return existing, False

        fd, path = tempfile.mkstemp(suffix=".png", prefix="satquery-")
        os.close(fd)
        image.convert("RGB").save(path)
        return path, True

    # -- execution ---------------------------------------------------------

    def _run(self, spec: TaskSpec, images: list[Any]) -> ModelOutput:
        if not self.ensure_loaded():
            return ModelOutput(
                answer=f"Model unavailable. {self.load_error}",
                confidence=0.0,
                model_id=self.model_id,
            )

        image = images[0]
        path, temporary = self._image_path(image)

        try:
            result = self._run_vqa(
                image_path=path,
                query=spec.raw_query,
                adapter_dir=str(self.adapter_dir),
            )
        except Exception as exc:
            return ModelOutput(
                answer=f"Inference failed: {type(exc).__name__}: {exc}"[:240],
                confidence=0.0,
                model_id=self.model_id,
            )
        finally:
            if temporary:
                try:
                    os.unlink(path)
                except OSError:
                    pass

        size = getattr(image, "size", (0, 0))
        evidence = self._to_evidence(result.get("regions", []), size)
        if not evidence:
            evidence = [Evidence(image_index=0, kind=EvidenceKind.WHOLE,
                                 note="whole image")]

        meta = result.get("metadata") or {}
        return ModelOutput(
            answer=str(result.get("answer", "")).strip(),
            confidence=float(result.get("confidence", 0.0)),
            model_id=str(meta.get("model", self.model_id)),
            evidence=evidence,
            scores={"reported_task": 1.0} if result.get("task") else {},
        )
