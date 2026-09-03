"""
Evaluation harness.

Its output is the most important slide in your deck. Almost nobody at internal
screening brings measured results; you will.

Three suites, each independently runnable:

  router     rule parser vs LLM parser on labelled questions.   No GPU, no data.
  validator  refusal correctness on synthetic input shapes.     No GPU, no data.
  answers    baseline vs fine-tuned VLM on held-out subsets.    Needs both.

Run:
    python -m eval.run_eval                      # router + validator only
    python -m eval.run_eval --with-llm           # adds LLM parser comparison
    python -m eval.run_eval --answers bigearthnet rsvqa
    python -m eval.run_eval --answers rsvqa --with-vlm
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import config                                               # noqa: E402
from contracts import TaskType                              # noqa: E402
from eval.metrics import TaskScore, markdown_table          # noqa: E402
from router.parser import LLMParser, RuleParser             # noqa: E402
from router.validator import validate                       # noqa: E402

# --------------------------------------------------------------------------
# Router test set. Grow this toward ~200 by hand - it costs an evening and it
# is the only labelled data you own outright. Split the writing across the
# team so no one person's phrasing habits bias it.
# --------------------------------------------------------------------------

ROUTER_CASES: list[tuple[str, TaskType]] = [
    # classify
    ("What land cover is in this image?", TaskType.CLASSIFY),
    ("Classify the terrain in this tile", TaskType.CLASSIFY),
    ("What kind of land is this?", TaskType.CLASSIFY),
    ("Give me the land use categories here", TaskType.CLASSIFY),
    # caption
    ("Describe this scene.", TaskType.CAPTION),
    ("Tell me about this image", TaskType.CAPTION),
    ("Summarise what you see", TaskType.CAPTION),
    ("Caption this satellite tile", TaskType.CAPTION),
    # ground
    ("Where is the water?", TaskType.GROUND),
    ("Locate the airport runway", TaskType.GROUND),
    ("Find the forest in this image", TaskType.GROUND),
    ("Show me the urban area", TaskType.GROUND),
    ("Which part of the tile is farmland?", TaskType.GROUND),
    # change
    ("What changed between these two images?", TaskType.CHANGE),
    ("Compare the before and after", TaskType.CHANGE),
    ("How much has the city grown since 2019?", TaskType.CHANGE),
    ("Show deforestation between 2020 and 2024", TaskType.CHANGE),
    ("What is different in the second image?", TaskType.CHANGE),
    # vqa
    ("How much of this tile is urban?", TaskType.VQA),
    ("Is there a river in this image?", TaskType.VQA),
    ("Are there any buildings near the coast?", TaskType.VQA),
    ("Does this area contain cropland?", TaskType.VQA),

    # --- hard cases -------------------------------------------------------
    # Known rule-parser failures, kept in the set on purpose. A benchmark you
    # score 100% on measures nothing. Closing this gap is the week-2 result.
    ("Has the glacier retreated?", TaskType.CHANGE),
    ("How many new buildings appeared?", TaskType.CHANGE),
    ("2019 vs 2023", TaskType.CHANGE),
    ("Is the lake smaller now?", TaskType.CHANGE),
    ("What is the crop type in the north-east corner?", TaskType.VQA),
    ("Anything unusual here?", TaskType.CAPTION),
    ("Give me a rundown of this area", TaskType.CAPTION),
    ("Pinpoint every settlement", TaskType.GROUND),
]

VALIDATOR_CASES: list[tuple[str, list[tuple[int, int]], bool]] = [
    ("What changed between these two images?", [(512, 512)], False),
    ("What changed between these two images?", [(512, 512), (512, 512)], True),
    ("What changed here?", [(512, 512), (900, 300)], False),
    ("Describe this scene.", [], False),
    ("Describe this scene.", [(512, 512)], True),
    ("Describe this scene.", [(32, 32)], False),
    ("asdf qwerty", [(512, 512)], False),
]


# --------------------------------------------------------------------------
# Suites
# --------------------------------------------------------------------------

def eval_router(parser, label: str) -> dict:
    correct = 0
    confusion: Counter[str] = Counter()
    failures: list[dict] = []
    t0 = time.perf_counter()

    for query, expected in ROUTER_CASES:
        got = parser.parse(query).task_type
        if got is expected:
            correct += 1
        else:
            confusion[f"{expected.value}->{got.value}"] += 1
            failures.append({"query": query, "expected": expected.value,
                             "got": got.value})

    n = len(ROUTER_CASES)
    return {
        "parser": label,
        "n": n,
        "correct": correct,
        "accuracy": round(correct / n, 4),
        "mean_latency_ms": round((time.perf_counter() - t0) * 1000 / n, 1),
        "confusion": dict(confusion),
        "failures": failures,
    }


def eval_validator() -> dict:
    parser = RuleParser()
    correct = 0
    failures: list[dict] = []

    for query, sizes, should_pass in VALIDATOR_CASES:
        spec = parser.parse(query, n_images=len(sizes))
        got = validate(spec, sizes).ok
        if got == should_pass:
            correct += 1
        else:
            failures.append({"query": query, "sizes": sizes,
                             "expected": should_pass, "got": got})

    n = len(VALIDATOR_CASES)
    return {"n": n, "correct": correct, "accuracy": round(correct / n, 4),
            "failures": failures}


def eval_answers(datasets: list[str], with_vlm: bool, limit: int) -> dict:
    """Baseline vs fine-tuned on held-out subsets. This is THE table."""
    from PIL import Image

    from data.subsets import load_jsonl
    from pipeline import SatQueryPipeline, build_registry
    from router.parser import RuleParser

    systems: dict[str, object] = {}

    base_reg = build_registry()
    systems["heuristic baseline"] = SatQueryPipeline(RuleParser(), base_reg)

    if with_vlm:
        from models.vlm import VLMModel

        vlm = VLMModel()
        if vlm.ensure_loaded():
            reg = build_registry().register(vlm)
            name = "fine-tuned VLM" if vlm.adapter else "base VLM"
            systems[name] = SatQueryPipeline(RuleParser(), reg)
        else:
            print(f"  VLM unavailable, baseline only: {vlm.load_error}")

    rows: list[dict] = []
    detail: dict[str, dict] = {}

    for sys_name, pipe in systems.items():
        for ds in datasets:
            path = config.DATA_DIR / ds / "test.jsonl"
            if not path.exists():
                print(f"  missing {path} - run: python -m data.subsets --dataset {ds}")
                continue

            records = load_jsonl(path)[:limit]
            root = config.DATA_DIR / ds
            score = TaskScore(task=f"{sys_name} / {ds}")
            multilabel = ds == "bigearthnet"

            for i, r in enumerate(records):
                images = [Image.open(root / p).convert("RGB") for p in r["images"]]
                ans = pipe.ask(r["prompt"], images)
                score.add(ans.text, r["answer"], ans.total_latency_ms,
                          multilabel=multilabel, keep_example=(i < 5))

            s = score.summary()
            s["system"] = sys_name
            s["dataset"] = ds
            rows.append(s)
            detail[f"{sys_name}/{ds}"] = score.examples
            print(f"  {sys_name:<20} {ds:<14} n={s['n']:<5} "
                  f"exact={s['exact_match']:.1%} f1={s['f1']:.1%}")

    return {"rows": rows, "examples": detail}


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------

def write_report(report: dict) -> Path:
    config.REPORT_DIR.mkdir(parents=True, exist_ok=True)
    (config.REPORT_DIR / "report.json").write_text(json.dumps(report, indent=2))

    md = ["# SatQuery evaluation", "", "## Router", ""]
    md.append(markdown_table(
        [{"parser": r["parser"], "n": r["n"], "accuracy": r["accuracy"],
          "mean_latency_ms": r["mean_latency_ms"]}
         for r in report.get("router", [])],
        ["parser", "n", "accuracy", "mean_latency_ms"],
    ))

    v = report.get("validator")
    if v:
        md += ["", "## Validator", "",
               f"{v['correct']}/{v['n']} refusal decisions correct "
               f"({v['accuracy']:.1%})."]

    ans = report.get("answers", {}).get("rows")
    if ans:
        md += ["", "## Answer accuracy", ""]
        md.append(markdown_table(
            ans, ["system", "dataset", "n", "exact_match", "f1",
                  "mean_latency_ms"]))

    fails = [f for r in report.get("router", []) for f in r["failures"]]
    if fails:
        md += ["", "## Remaining router failures", ""]
        md += [f"- `{f['query']}` - expected **{f['expected']}**, "
               f"got *{f['got']}*" for f in fails]

    path = config.REPORT_DIR / "report.md"
    path.write_text("\n".join(md) + "\n")
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate SatQuery.")
    ap.add_argument("--with-llm", action="store_true",
                    help="also evaluate the LLM parser (needs a configured backend)")
    ap.add_argument("--answers", nargs="*", default=None,
                    help="dataset names to score answers on")
    ap.add_argument("--with-vlm", action="store_true",
                    help="load the VLM and compare against the baseline")
    ap.add_argument("--limit", type=int, default=200)
    args = ap.parse_args()

    report: dict = {"config": config.summary()}

    print("\nRouter")
    report["router"] = [eval_router(RuleParser(), "rule-v1")]
    print(f"  rule-v1      {report['router'][0]['correct']}/"
          f"{report['router'][0]['n']}  "
          f"{report['router'][0]['accuracy']:.1%}")

    if args.with_llm:
        from router.llm_backends import get_backend

        backend = get_backend()
        if not backend.available():
            print(f"  llm-v1       skipped - backend '{backend.backend_id}' "
                  f"not configured (set SATQUERY_LLM_BACKEND and a key)")
        else:
            r = eval_router(LLMParser(backend=backend), "llm-v1")
            report["router"].append(r)
            print(f"  llm-v1       {r['correct']}/{r['n']}  {r['accuracy']:.1%}")

    print("\nValidator")
    report["validator"] = eval_validator()
    print(f"  {report['validator']['correct']}/{report['validator']['n']}  "
          f"{report['validator']['accuracy']:.1%}")

    if args.answers is not None:
        datasets = args.answers or ["bigearthnet", "rsvqa"]
        print("\nAnswers")
        report["answers"] = eval_answers(datasets, args.with_vlm, args.limit)

    path = write_report(report)
    print(f"\nreport -> {path}")

    for r in report["router"]:
        if r["failures"]:
            print(f"\n{r['parser']} failures:")
            for f in r["failures"]:
                print(f"  {f['query']}  expected {f['expected']}, got {f['got']}")


if __name__ == "__main__":
    main()
