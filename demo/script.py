"""
Scripted demo runner.

Two jobs:

  1. Rehearsal. Run the exact sequence you will run on stage, so you find out
     something is broken on a Tuesday and not in front of an evaluator.
  2. Insurance. It writes every overlay to PNG. If the live demo dies - laptop,
     wifi, projector, nerves - you have the exact same images in the deck and
     you keep talking. Teams that lose the room are teams with nothing to show
     when the app crashes.

Run:
    python -m demo.script                    # all steps, writes demo/output
    python -m demo.script --step 3           # rehearse one step
    python -m demo.script --no-images        # text only, fast
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from demo.samples import build as build_scene
from pipeline import SatQueryPipeline
from render import render_answer

# --------------------------------------------------------------------------
# The script. Order is the argument you are making, not a feature tour.
# --------------------------------------------------------------------------

STEPS: list[dict] = [
    {
        "title": "It understands the question",
        "scene": "urban_growth",
        "images": 1,
        "query": "What land cover is in this image?",
        "say": "Plain English in. The router parses it into a task type, "
               "picks a specialist, and the answer names which model ran.",
    },
    {
        "title": "It points at its evidence",
        "scene": "flood",
        "images": 1,
        "query": "Where is the water?",
        "say": "Not just an answer - a region. This is how you know it "
               "isn't making it up.",
    },
    {
        "title": "It refuses when it cannot answer",
        "scene": "deforestation",
        "images": 1,
        "query": "What changed between these two images?",
        "say": "One image, a change question. It declines and says what to "
               "supply instead. Systems that always answer are demos.",
    },
    {
        "title": "Change detection, typed and localised",
        "scene": "deforestation",
        "images": 2,
        "query": "What changed between these two images?",
        "say": "Two dates. It names the transition - not 'pixels moved' but "
               "'vegetation cleared' - and highlights where.",
    },
    {
        "title": "Multiple models, one fused answer",
        "scene": "glacier_retreat",
        "images": 2,
        "query": "Compare these two images and tell me what is different.",
        "say": "The plan shows two models running. Localisation and "
               "description are different jobs; the answer combines them. "
               "(Note: 'has the ice retreated?' needs the LLM parser - the "
               "rule parser reads it as a single-image question. That gap is "
               "measured in eval, not hidden.)",
    },
    {
        "title": "It catches bad inputs, not just bad questions",
        "scene": "different_places",
        "images": 2,
        "query": "What changed between these two images?",
        "say": "Two unrelated tiles. Pixel differencing would happily produce "
               "a confident change map. It vetoes instead.",
    },
]


def run(step: dict, pipe: SatQueryPipeline, out_dir: Path | None,
        index: int, size: int = 384) -> dict:
    before, after, _, desc = build_scene(step["scene"], size)
    images = [before, after][:step["images"]]

    answer = pipe.ask(step["query"], images)

    print(f"\n{'=' * 72}")
    print(f"{index}. {step['title']}")
    print(f"{'=' * 72}")
    print(f"  scene   {step['scene']} ({desc}), {step['images']} image(s)")
    print(f"  ask     \"{step['query']}\"")
    if answer.plan and answer.plan.model_ids:
        print(f"  plan    {' -> '.join(answer.plan.model_ids)} "
              f"[{answer.plan.fusion.value}]")
    print(f"  result  {'ANSWER' if answer.answered else 'REFUSED'} "
          f"in {answer.total_latency_ms:.0f} ms")
    print(f"\n  {answer.text}")
    print(f"\n  say: {step['say']}")

    written: list[str] = []
    if out_dir is not None:
        rendered = render_answer(images, answer)
        for k, img in rendered.items():
            slug = k.lower().replace(" ", "_")
            path = out_dir / f"{index:02d}_{step['scene']}_{slug}.png"
            img.save(path)
            written.append(path.name)
        if written:
            print(f"  saved: {', '.join(written)}")

    return {
        "step": index,
        "title": step["title"],
        "query": step["query"],
        "answered": answer.answered,
        "plan": answer.plan.model_ids if answer.plan else [],
        "fusion": answer.plan.fusion.value if answer.plan else None,
        "latency_ms": round(answer.total_latency_ms, 1),
        "text": answer.text,
        "images": written,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the scripted demo.")
    ap.add_argument("--step", type=int, default=None,
                    help="run a single step (1-based)")
    ap.add_argument("--out", type=Path, default=Path("demo/output"))
    ap.add_argument("--no-images", action="store_true")
    ap.add_argument("--size", type=int, default=384)
    args = ap.parse_args()

    out_dir = None
    if not args.no_images:
        args.out.mkdir(parents=True, exist_ok=True)
        out_dir = args.out

    pipe = SatQueryPipeline()
    steps = ([STEPS[args.step - 1]] if args.step else STEPS)
    offset = args.step or 1

    results = [run(s, pipe, out_dir, offset + i, args.size)
               for i, s in enumerate(steps)]

    print(f"\n{'=' * 72}")
    ok = sum(1 for r in results if r["answered"])
    print(f"{len(results)} step(s): {ok} answered, {len(results) - ok} refused")
    print("Refusals here are CORRECT - steps 3 and 6 are supposed to decline.")

    if out_dir is not None:
        (out_dir / "run.json").write_text(json.dumps(results, indent=2))
        print(f"\nassets -> {out_dir}")
        print("Put these in the deck. If the live demo dies, you keep talking.")


if __name__ == "__main__":
    main()
