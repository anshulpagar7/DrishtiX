"""
Inference interface — THE file Person 2 imports into the agent's tool layer.

Public contract:

    from scripts.inference import run_vqa

    result = run_vqa(image_path="scene.png", query="Where are the buildings?")

Returns exactly:
{
    "task": "vqa",
    "answer": "...",
    "confidence": 0.0-1.0,
    "regions": [{"bbox": [x, y, w, h], "label": "..."}],
    "mask_path": null,
    "metadata": {"model": "...", "task_type": "vqa" | "grounding"}
}

Person 2 should never need to know what's inside SatQueryVQAModel — if the
backbone changes (RS-LLaVA -> GeoChat) or the adapter is retrained, this
function signature and output shape stay fixed.
"""
import argparse
import re
from typing import Optional

import torch
import yaml
from PIL import Image
from peft import PeftModel
from transformers import AutoModelForVision2Seq, AutoProcessor

_MODEL_CACHE = {}


class SatQueryVQAModel:
    """Loads the base backbone + LoRA adapter once, reuses across calls."""

    def __init__(self, adapter_dir: str, model_config_path: str = "configs/model.yaml"):
        with open(model_config_path) as f:
            cfg = yaml.safe_load(f)
        base_name = cfg["base_model"]["name"]

        self.processor = AutoProcessor.from_pretrained(adapter_dir)
        base_model = AutoModelForVision2Seq.from_pretrained(
            base_name, device_map="auto", torch_dtype=torch.bfloat16, trust_remote_code=True
        )
        self.model = PeftModel.from_pretrained(base_model, adapter_dir)
        self.model.eval()

    def _is_grounding_query(self, query: str) -> bool:
        grounding_terms = ("where is", "where are", "locate", "find the", "identify the location of")
        return any(term in query.lower() for term in grounding_terms)

    def _parse_bbox(self, text: str):
        match = re.search(r"\[?\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]?", text)
        if not match:
            return None
        return [int(v) for v in match.groups()]

    def predict(self, image_path: str, query: str, task_hint: Optional[str] = None) -> dict:
        image = Image.open(image_path).convert("RGB")
        is_grounding = task_hint == "grounding" or (task_hint is None and self._is_grounding_query(query))

        if is_grounding:
            prompt = (
                f"<image>\nLocate the following in the image and return its bounding box "
                f"as [x, y, width, height]: {query}"
            )
        else:
            prompt = f"<image>\n{query}"

        inputs = self.processor(text=prompt, images=image, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False,
                output_scores=True,
                return_dict_in_generate=True,
            )

        decoded = self.processor.batch_decode(output_ids.sequences, skip_special_tokens=True)[0]
        # Strip the echoed prompt if the model includes it in output.
        answer_text = decoded.split(query)[-1].strip(" :\n")

        confidence = self._estimate_confidence(output_ids)

        regions = []
        if is_grounding:
            bbox = self._parse_bbox(answer_text)
            if bbox:
                regions.append({"bbox": bbox, "label": query})
                answer_text = f"Found the requested region — see highlighted bounding box."

        return {
            "answer": answer_text,
            "confidence": confidence,
            "regions": regions,
            "task_type": "grounding" if is_grounding else "vqa",
        }

    def _estimate_confidence(self, generation_output) -> float:
        """Rough confidence proxy from generation scores (mean token probability)."""
        try:
            scores = generation_output.scores
            probs = [torch.softmax(s, dim=-1).max().item() for s in scores]
            return round(sum(probs) / len(probs), 3) if probs else 0.5
        except Exception:
            return 0.5


def run_vqa(image_path: str, query: str, adapter_dir: str = "outputs/lora_vqa_v1") -> dict:
    """Person-2-facing entrypoint. Loads/caches the model, returns the shared JSON contract."""
    if adapter_dir not in _MODEL_CACHE:
        _MODEL_CACHE[adapter_dir] = SatQueryVQAModel(adapter_dir=adapter_dir)
    model = _MODEL_CACHE[adapter_dir]

    result = model.predict(image_path, query)

    return {
        "task": result["task_type"],
        "answer": result["answer"],
        "confidence": result["confidence"],
        "regions": result["regions"],
        "mask_path": None,
        "metadata": {"model": "rs-vqa-lora-v1", "task_type": result["task_type"]},
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sanity-check inference from the command line")
    parser.add_argument("--adapter_dir", type=str, required=True)
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--query", type=str, required=True)
    args = parser.parse_args()

    result = run_vqa(args.image, args.query, adapter_dir=args.adapter_dir)
    import json
    print(json.dumps(result, indent=2))
