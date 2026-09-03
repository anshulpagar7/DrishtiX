"""
Instruction templates.

The single most common fine-tuning bug in projects like this is a mismatch
between the prompt used at training time and the prompt used at inference.
The model learns one format and gets asked in another, accuracy collapses,
and the team blames the LoRA rank.

So both sides import from here. Never inline a prompt string anywhere else.
"""

from __future__ import annotations

from contracts import TaskSpec, TaskType

# --------------------------------------------------------------------------
# Task instructions
# --------------------------------------------------------------------------

INSTRUCTIONS: dict[TaskType, str] = {
    TaskType.CLASSIFY: (
        "List the land cover classes visible in this satellite image. "
        "Answer with a comma-separated list and nothing else."
    ),
    TaskType.CAPTION: (
        "Describe this satellite image in one sentence."
    ),
    TaskType.GROUND: (
        "Locate {target} in this satellite image. Answer with the region "
        "and nothing else."
    ),
    TaskType.CHANGE: (
        "These are two satellite images of the same area at different times. "
        "Describe what changed between them in one sentence."
    ),
    TaskType.VQA: (
        "Answer the question about this satellite image concisely.\n"
        "Question: {question}"
    ),
}

# Prepended to every prompt so the model knows its domain.
SYSTEM_PREFIX = (
    "You are a remote sensing analyst. You answer only from what is visible "
    "in the imagery. If the image does not show enough to answer, say so."
)


def build_prompt(spec: TaskSpec) -> str:
    """TaskSpec -> the exact string the model sees. Used by train AND infer."""
    template = INSTRUCTIONS.get(spec.task_type, INSTRUCTIONS[TaskType.VQA])
    return template.format(
        target=spec.target_class or "the target",
        question=spec.raw_query,
    )


def build_training_prompt(task_type: TaskType, question: str = "",
                          target: str = "") -> str:
    """Same templates, reachable without constructing a TaskSpec."""
    template = INSTRUCTIONS.get(task_type, INSTRUCTIONS[TaskType.VQA])
    return template.format(target=target or "the target", question=question)


def chat_messages(prompt: str, n_images: int = 1) -> list[dict]:
    """Standard HF chat format with image placeholders.

    Kept here so training and inference build messages identically.
    """
    content: list[dict] = [{"type": "image"} for _ in range(max(n_images, 1))]
    content.append({"type": "text", "text": f"{SYSTEM_PREFIX}\n\n{prompt}"})
    return [{"role": "user", "content": content}]
