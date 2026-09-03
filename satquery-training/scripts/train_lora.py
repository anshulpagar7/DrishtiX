"""
LoRA/QLoRA fine-tuning entrypoint for the RS VQA + grounding backbone.

Usage:
    python scripts/train_lora.py \
        --config configs/model.yaml \
        --train_config configs/training.yaml \
        --data_dir data/vrsbench_processed \
        --output_dir outputs/lora_vqa_v1
"""
import argparse
import json
import os

import torch
import yaml
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from PIL import Image
from transformers import (
    AutoProcessor,
    AutoModelForVision2Seq,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)


def load_yaml(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def build_prompt(record: dict) -> str:
    """Turn a unified-format record into an instruction-tuning prompt."""
    if record["task_type"] == "grounding":
        return (
            f"<image>\nLocate the following in the image and return its bounding box "
            f"as [x, y, width, height]: {record['instruction']}"
        )
    return f"<image>\n{record['instruction']}"


def build_target(record: dict) -> str:
    if record["task_type"] == "grounding" and record.get("bbox"):
        x, y, w, h = record["bbox"]
        return f"[{x}, {y}, {w}, {h}]"
    return record["response"]


def load_records(data_dir: str, split: str) -> list:
    with open(os.path.join(data_dir, f"{split}.json")) as f:
        return json.load(f)


def make_dataset(records: list, image_root: str, processor):
    def _map(example):
        prompt = build_prompt(example)
        target = build_target(example)
        image = Image.open(os.path.join(image_root, example["image"])).convert("RGB")
        inputs = processor(text=prompt, images=image, return_tensors="pt")
        labels = processor.tokenizer(target, return_tensors="pt").input_ids
        return {
            "input_ids": inputs["input_ids"][0],
            "attention_mask": inputs["attention_mask"][0],
            "pixel_values": inputs["pixel_values"][0],
            "labels": labels[0],
        }

    ds = Dataset.from_list(records)
    return ds.map(_map, remove_columns=ds.column_names)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--train_config", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--image_root", type=str, default=None,
                         help="Defaults to --data_dir if not set")
    parser.add_argument("--output_dir", type=str, required=True)
    args = parser.parse_args()

    model_cfg = load_yaml(args.config)
    train_cfg = load_yaml(args.train_config)["training"]
    data_cfg = load_yaml(args.train_config)["data"]
    image_root = args.image_root or args.data_dir

    base = model_cfg["base_model"]
    quant_config = BitsAndBytesConfig(
        load_in_4bit=base["load_in_4bit"],
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    ) if base["load_in_4bit"] else None

    print(f"Loading base model: {base['name']}")
    processor = AutoProcessor.from_pretrained(base["name"], trust_remote_code=base["trust_remote_code"])
    model = AutoModelForVision2Seq.from_pretrained(
        base["name"],
        quantization_config=quant_config,
        device_map="auto",
        trust_remote_code=base["trust_remote_code"],
    )

    if base["load_in_4bit"]:
        model = prepare_model_for_kbit_training(model)

    lora_cfg = model_cfg["lora"]
    peft_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        lora_dropout=lora_cfg["lora_dropout"],
        target_modules=lora_cfg["target_modules"],
        bias=lora_cfg["bias"],
        task_type=lora_cfg["task_type"],
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()

    print("Building datasets...")
    train_records = load_records(args.data_dir, "train")[: data_cfg["max_samples"]]
    val_records = load_records(args.data_dir, "val")

    train_ds = make_dataset(train_records, image_root, processor)
    val_ds = make_dataset(val_records, image_root, processor)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=train_cfg["num_train_epochs"],
        per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
        per_device_eval_batch_size=train_cfg["per_device_eval_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        learning_rate=train_cfg["learning_rate"],
        lr_scheduler_type=train_cfg["lr_scheduler_type"],
        warmup_ratio=train_cfg["warmup_ratio"],
        weight_decay=train_cfg["weight_decay"],
        logging_steps=train_cfg["logging_steps"],
        save_steps=train_cfg["save_steps"],
        eval_steps=train_cfg["eval_steps"],
        save_total_limit=train_cfg["save_total_limit"],
        gradient_checkpointing=train_cfg["gradient_checkpointing"],
        bf16=train_cfg["bf16"],
        eval_strategy="steps",
        report_to="none",
        seed=train_cfg["seed"],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
    )

    print("Starting training...")
    trainer.train()

    print(f"Saving LoRA adapter to {args.output_dir}")
    model.save_pretrained(args.output_dir)
    processor.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
