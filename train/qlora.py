"""
QLoRA fine-tuning for the SatQuery VLM.

Designed for a single free Kaggle T4. Every default here exists because of a
constraint, not a preference:

  batch_size 1 + grad_accum 8   a T4 has 15 GB; effective batch 8 without OOM
  4-bit base + LoRA adapters    full fine-tuning of even a 2B VLM will not fit
  save every epoch              Kaggle kills sessions at ~9-12 hrs, silently
  --resume                      so a killed session costs one epoch, not all

Run:
    python -m train.qlora --datasets bigearthnet rsvqa
    python -m train.qlora --datasets bigearthnet --resume .checkpoints/epoch1
    python -m train.qlora --dry-run          # no GPU needed, checks the wiring

The adapter it writes is what models/vlm.py loads at inference. Point
SATQUERY_VLM_ADAPTER at it and the router starts preferring the tuned model.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import config
from data.prompts import SYSTEM_PREFIX
from data.subsets import load_jsonl


# --------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------

class JsonlVLMDataset:
    """Reads the JSONL that data.subsets writes. Lazy image loading."""

    def __init__(self, records: list[dict], root: Path) -> None:
        self.records = records
        self.root = Path(root)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, i: int) -> dict:
        from PIL import Image

        r = self.records[i]
        images = [Image.open(self.root / p).convert("RGB") for p in r["images"]]
        return {"prompt": r["prompt"], "answer": r["answer"],
                "images": images, "task": r["task"]}


def load_datasets(names: list[str], split: str = "train") -> tuple[list[dict], list[Path]]:
    """Merge several subsets into one training pool."""
    records: list[dict] = []
    roots: list[Path] = []
    for name in names:
        root = config.DATA_DIR / name
        path = root / f"{split}.jsonl"
        if not path.exists():
            raise SystemExit(
                f"Missing {path}. Build it first:\n"
                f"    python -m data.subsets --dataset {name}"
            )
        rows = load_jsonl(path)
        for r in rows:
            r["_root"] = str(root)
        records.extend(rows)
        roots.append(root)
        print(f"  {name}: {len(rows)} {split} records")
    return records, roots


# --------------------------------------------------------------------------
# Collation
# --------------------------------------------------------------------------

def make_collator(processor: Any, max_len: int) -> Any:
    """Builds chat-formatted batches and masks the prompt out of the loss.

    Masking matters: without it the model spends capacity learning to
    reproduce your instruction text instead of the answer.
    """
    from PIL import Image

    def collate(batch: list[dict]) -> dict:
        texts, image_lists = [], []
        for ex in batch:
            root = Path(ex["_root"])
            images = [Image.open(root / p).convert("RGB") for p in ex["images"]]
            content = [{"type": "image"} for _ in images]
            content.append({"type": "text",
                            "text": f"{SYSTEM_PREFIX}\n\n{ex['prompt']}"})
            messages = [
                {"role": "user", "content": content},
                {"role": "assistant",
                 "content": [{"type": "text", "text": ex["answer"]}]},
            ]
            texts.append(processor.apply_chat_template(messages, tokenize=False))
            image_lists.append(images)

        enc = processor(
            text=texts,
            images=image_lists,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_len,
        )
        labels = enc["input_ids"].clone()
        pad_id = getattr(processor.tokenizer, "pad_token_id", None)
        if pad_id is not None:
            labels[labels == pad_id] = -100
        enc["labels"] = labels
        return enc

    return collate


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------

def load_model_for_training(checkpoint: str, load_4bit: bool = True,
                            resume: str | None = None):
    """4-bit base + LoRA adapters. Returns (model, processor)."""
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig

    kwargs: dict[str, Any] = {"device_map": "auto"}
    if load_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

    processor = AutoProcessor.from_pretrained(checkpoint)
    model = AutoModelForVision2Seq.from_pretrained(checkpoint, **kwargs)

    if load_4bit:
        model = prepare_model_for_kbit_training(model)

    if resume:
        print(f"resuming adapters from {resume}")
        model = PeftModel.from_pretrained(model, resume, is_trainable=True)
    else:
        lora = LoraConfig(
            r=config.TRAIN["lora_r"],
            lora_alpha=config.TRAIN["lora_alpha"],
            lora_dropout=config.TRAIN["lora_dropout"],
            bias="none",
            task_type="CAUSAL_LM",
            # Attention projections only. Touching the vision tower as well
            # roughly doubles trainable params for little gain at this scale.
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        )
        model = get_peft_model(model, lora)

    model.print_trainable_parameters()
    return model, processor


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------

def train(datasets: list[str], out_dir: Path, epochs: int | None = None,
          resume: str | None = None) -> Path:
    from torch.utils.data import DataLoader
    from transformers import Trainer, TrainingArguments

    epochs = epochs or config.TRAIN["epochs"]
    out_dir.mkdir(parents=True, exist_ok=True)

    print("config:", json.dumps(config.summary(), indent=2, default=str))
    print("loading data")
    records, _ = load_datasets(datasets, "train")
    print(f"  total: {len(records)} records")

    model, processor = load_model_for_training(
        config.VLM_CHECKPOINT, config.VLM_LOAD_4BIT, resume
    )

    args = TrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=epochs,
        per_device_train_batch_size=config.TRAIN["batch_size"],
        gradient_accumulation_steps=config.TRAIN["grad_accum"],
        learning_rate=config.TRAIN["lr"],
        warmup_ratio=config.TRAIN["warmup_ratio"],
        fp16=True,
        gradient_checkpointing=True,
        logging_steps=25,
        save_strategy="epoch",            # never turn this off - sessions die
        save_total_limit=2,
        report_to=[],
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=records,
        data_collator=make_collator(processor, config.TRAIN["max_len"]),
    )

    t0 = time.time()
    trainer.train(resume_from_checkpoint=bool(resume))
    mins = (time.time() - t0) / 60

    final = out_dir / "adapter"
    model.save_pretrained(final)
    processor.save_pretrained(final)

    (out_dir / "run.json").write_text(json.dumps({
        "datasets": datasets,
        "records": len(records),
        "epochs": epochs,
        "minutes": round(mins, 1),
        "base": config.VLM_CHECKPOINT,
        "lora": {k: v for k, v in config.TRAIN.items() if k.startswith("lora")},
    }, indent=2))

    print(f"\ndone in {mins:.1f} min")
    print(f"adapter -> {final}")
    print(f"\nNext:\n    export SATQUERY_VLM_ADAPTER={final}\n"
          f"    python -m eval.run_eval --with-vlm")
    return final


# --------------------------------------------------------------------------
# Dry run - verifies the wiring without a GPU
# --------------------------------------------------------------------------

def dry_run(datasets: list[str]) -> None:
    """Checks data exists and prompts build. No model, no GPU, no downloads."""
    print("dry run - checking data and prompt construction only\n")
    try:
        records, _ = load_datasets(datasets, "train")
    except SystemExit as exc:
        print(exc)
        return

    print(f"\n{len(records)} records loaded")
    by_task: dict[str, int] = {}
    for r in records:
        by_task[r["task"]] = by_task.get(r["task"], 0) + 1
    for task, n in sorted(by_task.items()):
        print(f"  {task:<10} {n}")

    print("\nsample prompt:")
    print("-" * 60)
    r = records[0]
    print(f"{SYSTEM_PREFIX}\n\n{r['prompt']}")
    print("-" * 60)
    print(f"target: {r['answer']}")
    print(f"images: {r['images']}")


def main() -> None:
    ap = argparse.ArgumentParser(description="QLoRA fine-tune the SatQuery VLM.")
    ap.add_argument("--datasets", nargs="+", default=["vrsbench", "rsvqa"])
    ap.add_argument("--out", type=Path, default=config.CKPT_DIR / "qlora")
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--resume", type=str, default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if args.dry_run:
        dry_run(args.datasets)
    else:
        train(args.datasets, args.out, args.epochs, args.resume)


if __name__ == "__main__":
    main()
