"""
Central configuration. Everything tunable lives here, nothing is hardcoded
across the codebase.

Secrets come from environment variables only. Never commit a key. On Kaggle
use Add-ons -> Secrets; locally use a .env you have gitignored.
"""

from __future__ import annotations

import os
from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent
CACHE_DIR = Path(os.getenv("SATQUERY_CACHE", ROOT / ".cache"))
DATA_DIR = Path(os.getenv("SATQUERY_DATA", ROOT / ".data"))
CKPT_DIR = Path(os.getenv("SATQUERY_CKPT", ROOT / ".checkpoints"))
REPORT_DIR = ROOT / "eval" / "reports"

for _d in (CACHE_DIR, DATA_DIR, CKPT_DIR, REPORT_DIR):
    _d.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------
# LLM parser backend
# --------------------------------------------------------------------------
# Set SATQUERY_LLM_BACKEND to one of: offline | openai_compat | gemini
# offline is the default so nothing ever depends on a network call by accident.

LLM_BACKEND = os.getenv("SATQUERY_LLM_BACKEND", "offline")

# OpenAI-compatible endpoints (Groq, OpenRouter, together, local vLLM...).
# Groq free tier:      https://api.groq.com/openai/v1
OPENAI_COMPAT_BASE = os.getenv("SATQUERY_LLM_BASE", "https://api.groq.com/openai/v1")
OPENAI_COMPAT_MODEL = os.getenv("SATQUERY_LLM_MODEL", "llama-3.1-8b-instant")
OPENAI_COMPAT_KEY = os.getenv("SATQUERY_LLM_KEY", "")

# Google AI Studio free tier.
GEMINI_MODEL = os.getenv("SATQUERY_GEMINI_MODEL", "gemini-2.0-flash")
GEMINI_KEY = os.getenv("SATQUERY_GEMINI_KEY", "")

LLM_TIMEOUT_S = float(os.getenv("SATQUERY_LLM_TIMEOUT", "8"))

# Cache every LLM parse to disk. This is what stops a rate limit from killing
# a live demo: rehearsed queries are already on disk before you walk on stage.
LLM_CACHE_PATH = CACHE_DIR / "llm_parse_cache.json"


# --------------------------------------------------------------------------
# Vision-language model
# --------------------------------------------------------------------------
# Verify a 4-bit load of whichever checkpoint you pick actually fits a Kaggle
# T4 BEFORE planning around it. Search the HF Hub for remote-sensing adapted
# checkpoints first.

VLM_CHECKPOINT = os.getenv("SATQUERY_VLM", "Qwen/Qwen2-VL-2B-Instruct")
VLM_ADAPTER = os.getenv("SATQUERY_VLM_ADAPTER", "")   # path to trained LoRA
VLM_LOAD_4BIT = os.getenv("SATQUERY_VLM_4BIT", "1") == "1"
VLM_MAX_NEW_TOKENS = int(os.getenv("SATQUERY_VLM_TOKENS", "96"))


# --------------------------------------------------------------------------
# Dataset subsets
# --------------------------------------------------------------------------
# Small on purpose. Full BigEarthNet is >100 GB; you need none of that for a
# hackathon fine-tune or a demo.

SUBSET_SIZES = {
    "bigearthnet": int(os.getenv("SATQUERY_N_BEN", "4000")),
    "rsvqa": int(os.getenv("SATQUERY_N_RSVQA", "3000")),
    "vrsbench": int(os.getenv("SATQUERY_N_VRSB", "1500")),
    "cdvqa": int(os.getenv("SATQUERY_N_CDVQA", "1000")),
}

SEED = 42


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------

TRAIN = {
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "lr": 1e-4,
    "epochs": 2,
    "batch_size": 1,
    "grad_accum": 8,          # effective batch 8 on a single T4
    "max_len": 512,
    "warmup_ratio": 0.03,
    "save_every_epoch": True,  # Kaggle sessions die. Never turn this off.
}


def summary() -> dict[str, object]:
    """Printed at the top of training and eval runs so every log is traceable."""
    return {
        "llm_backend": LLM_BACKEND,
        "vlm_checkpoint": VLM_CHECKPOINT,
        "vlm_adapter": VLM_ADAPTER or "(none)",
        "vlm_4bit": VLM_LOAD_4BIT,
        "subset_sizes": SUBSET_SIZES,
        "seed": SEED,
    }
