# SatQuery AI — Person 1: Model Training Module

This module owns the remote-sensing VQA + grounding model: fine-tuning, evaluation,
and the inference interface that Person 2's agent calls.

## Scope for MVP (Phase 2 vertical slice)

We are fine-tuning a single backbone to cover **both**:
- Visual Question Answering (open + closed-ended)
- Text-guided region grounding (bounding boxes)

using LoRA adapters on top of an open-source RS-vision-language checkpoint,
fine-tuned on **VRSBench** (captioning + VQA + referring-expression grounding,
all on the same images — ideal since it lets one dataset drive two capabilities).

Change detection and Optical+SAR fusion are separate, later modules (see
`docs/change_detection.md` and `docs/optical_sar.md` once you get to Phase 4/5).
Don't start those until VQA is solid end-to-end with Person 2 + Person 3.

## Why this backbone

- **RS-LLaVA** (LLaVA architecture, pretrained/fine-tuned for RS captioning+VQA)
  is the recommended starting checkpoint — it's already domain-adapted, so LoRA
  fine-tuning on VRSBench should converge fast with limited compute.
- **GeoChat** is a stronger alternative if your GPU budget allows — it natively
  supports region-grounding, so you get grounding "for free" without a second
  fine-tuning run. Try RS-LLaVA first; swap to GeoChat if grounding quality is
  poor and you have more GPU time.

Both are swapped via `configs/model.yaml` — the training/inference code doesn't
hardcode a specific checkpoint.

## Directory layout

```
satquery-training/
├── configs/
│   ├── model.yaml          # which base checkpoint + LoRA hyperparams
│   └── training.yaml       # batch size, epochs, lr, etc.
├── scripts/
│   ├── prepare_dataset.py  # download + reformat VRSBench into instruction format
│   ├── train_lora.py       # LoRA/PEFT fine-tuning entrypoint
│   ├── evaluate.py         # run eval metrics (accuracy for closed VQA, IoU for grounding)
│   └── inference.py        # THE FILE Person 2 imports — clean function interface
├── data/                   # (gitignored) downloaded + processed dataset lives here
├── outputs/                # (gitignored) LoRA adapter checkpoints land here
└── requirements.txt
```

## Setup (Colab / Kaggle / local GPU box)

```bash
pip install -r requirements.txt
```

You'll need a Hugging Face account + token for gated checkpoints:
```bash
huggingface-cli login
```

## Pipeline

```bash
# 1. Pull and reformat VRSBench into instruction-tuning JSON
python scripts/prepare_dataset.py --output_dir data/vrsbench_processed

# 2. LoRA fine-tune
python scripts/train_lora.py \
    --config configs/model.yaml \
    --train_config configs/training.yaml \
    --data_dir data/vrsbench_processed \
    --output_dir outputs/lora_vqa_v1

# 3. Evaluate
python scripts/evaluate.py --adapter_dir outputs/lora_vqa_v1 --data_dir data/vrsbench_processed

# 4. Sanity check inference locally
python scripts/inference.py --adapter_dir outputs/lora_vqa_v1 \
    --image path/to/test.png --query "Where are the buildings?"
```

## The contract with Person 2

`scripts/inference.py` exposes a single function:

```python
from scripts.inference import run_vqa

result = run_vqa(image_path="scene.png", query="Where are the buildings?")
```

returning exactly the JSON shape Person 2's agent expects:

```json
{
  "task": "vqa",
  "answer": "The buildings are concentrated in the northern half of the image, along the main road.",
  "confidence": 0.87,
  "regions": [
    {"bbox": [120, 45, 260, 180], "label": "building cluster"}
  ],
  "mask_path": null,
  "metadata": {"model": "rs-llava-lora-v1", "task_type": "vqa+grounding"}
}
```

Person 2 should treat this function as a black box — if we swap RS-LLaVA for
GeoChat later, or retrain with more data, the function signature and output
shape don't change. That's the whole point of the modular contract in the
project doc (Section 9).

## GPU budget notes

- LoRA fine-tuning a 7B-class VLM: fits on a single T4 (16GB) with 4-bit
  quantization (QLoRA) + gradient checkpointing, or comfortably on an A100.
- Full VRSBench is large — for a hackathon timeline, start with a **filtered
  subset** (`--max_samples` flag in `prepare_dataset.py`) of a few thousand
  examples to get a working adapter fast, then scale up if time allows.
- Budget for 2-4 training runs, not one. First run is almost always about
  finding the right LoRA rank / learning rate, not the final model.
