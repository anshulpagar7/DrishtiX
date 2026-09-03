"""
The vision-language model adapter.

Design rules, all of which exist to protect the live demo:

  1. Lazy loading. Weights load on first use, never at import, so the UI
     starts instantly whether or not a GPU exists.
  2. can_handle() returns False until the weights are genuinely resident.
     An absent model must never win a route and then fail.
  3. A failed load is remembered. It is attempted once, not on every query.
  4. The heuristic baselines stay registered at priority 0. When this model
     is unavailable the router falls through and the demo survives.

Bring it online with:
    export SATQUERY_VLM_ADAPTER=.checkpoints/qlora/adapter
"""

from __future__ import annotations

from typing import Any

import config
from contracts import Evidence, ModelOutput, TaskSpec, TaskType
from data.prompts import build_prompt, chat_messages
from models.base import SpecialistModel


class VLMModel(SpecialistModel):
    """Fine-tuned remote-sensing VLM behind the standard interface."""

    model_id = "vlm-v1"
    supported_tasks = {TaskType.VQA, TaskType.CAPTION, TaskType.CLASSIFY,
                       TaskType.CHANGE}
    priority = 20                       # beats every heuristic when available

    def __init__(self, checkpoint: str | None = None,
                 adapter: str | None = None,
                 load_in_4bit: bool | None = None,
                 eager: bool = False) -> None:
        super().__init__()
        self.checkpoint = checkpoint or config.VLM_CHECKPOINT
        self.adapter = adapter if adapter is not None else config.VLM_ADAPTER
        self.load_in_4bit = (config.VLM_LOAD_4BIT if load_in_4bit is None
                             else load_in_4bit)
        self._model = None
        self._processor = None
        self._load_failed = False
        self.load_error: str = ""
        if eager:
            self.ensure_loaded()

    # -- loading -----------------------------------------------------------

    def ensure_loaded(self) -> bool:
        """Attempt the load once. Returns whether the model is usable."""
        if self._loaded:
            return True
        if self._load_failed:
            return False

        try:
            import torch
            from transformers import AutoModelForVision2Seq, AutoProcessor

            kwargs: dict[str, Any] = {"device_map": "auto"}
            if self.load_in_4bit:
                from transformers import BitsAndBytesConfig

                kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                )

            src = self.adapter or self.checkpoint
            self._processor = AutoProcessor.from_pretrained(src)
            self._model = AutoModelForVision2Seq.from_pretrained(
                self.checkpoint, **kwargs
            )

            if self.adapter:
                from peft import PeftModel

                self._model = PeftModel.from_pretrained(self._model, self.adapter)
                self.model_id = "vlm-tuned-v1"

            self._model.eval()
            self._loaded = True
            return True

        except Exception as exc:
            self._load_failed = True
            self.load_error = f"{type(exc).__name__}: {exc}"[:300]
            return False

    # -- capability --------------------------------------------------------

    def can_handle(self, spec: TaskSpec) -> bool:
        """Only claim work once weights are genuinely resident.

        Deliberately does NOT trigger a load. Call ensure_loaded() explicitly
        at startup if you want the model in play - a route decision is the
        wrong place to spend 40 seconds.
        """
        if spec.task_type not in self.supported_tasks:
            return False
        if spec.task_type is TaskType.CHANGE and not spec.needs_pair:
            return False
        return self._loaded

    @property
    def status(self) -> str:
        if self._loaded:
            return f"loaded ({'tuned' if self.adapter else 'base'})"
        if self._load_failed:
            return f"unavailable - {self.load_error}"
        return "not loaded"

    # -- execution ---------------------------------------------------------

    def _run(self, spec: TaskSpec, images: list[Any]) -> ModelOutput:
        if not self.ensure_loaded():
            return ModelOutput(
                answer=f"Model unavailable. {self.load_error}",
                confidence=0.0,
                model_id=self.model_id,
            )

        import torch

        n = 2 if spec.needs_pair and len(images) >= 2 else 1
        used = images[:n]

        prompt = build_prompt(spec)
        messages = chat_messages(prompt, n_images=n)
        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self._processor(text=[text], images=[used], return_tensors="pt")
        inputs = {k: v.to(self._model.device) for k, v in inputs.items()}

        with torch.no_grad():
            out = self._model.generate(
                **inputs,
                max_new_tokens=config.VLM_MAX_NEW_TOKENS,
                do_sample=False,
                return_dict_in_generate=True,
                output_scores=True,
            )

        seq = out.sequences[0][inputs["input_ids"].shape[1]:]
        answer = self._processor.decode(seq, skip_special_tokens=True).strip()

        # Mean top-token probability as a rough confidence. NOT calibrated -
        # say so if asked, and do not put it on a slide as if it were.
        confidence = 0.0
        try:
            probs = [torch.softmax(s[0], dim=-1).max().item()
                     for s in out.scores[:len(seq)]]
            confidence = round(sum(probs) / max(len(probs), 1), 3)
        except Exception:
            pass

        return ModelOutput(
            answer=answer,
            confidence=confidence,
            model_id=self.model_id,
            evidence=[
                Evidence(image_index=i,
                         note=f"whole image, prompt: {spec.task_type.value}")
                for i in range(n)
            ],
        )
