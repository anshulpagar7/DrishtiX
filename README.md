# DrishtiX

SIH 2026 · **SIH26167** · ISRO · Software · Idea deadline 20 September 2026

An interactive vision-language assistant for satellite imagery. Ask a question
in plain English; the system works out what is being asked, checks the supplied
images can actually answer it, plans which specialists to run, and returns a
fused answer that points at its own evidence.

---

## Run it — backend + site

```bash
pip install -r requirements-api.txt
uvicorn api.main:app --reload --port 8000
```

Open <http://localhost:8000>. Five pages, one origin, no CORS to configure.
No GPU required — the API answers with classical baselines until a trained
adapter exists.

| Page | What it is |
|---|---|
| `/` | Launch sequence. Scroll drives flight state. |
| `/console.html` | **Live.** Upload or pick a scene, ask, get a grounded answer. |
| `/system.html` | Six stages, live model registry, the integration contract. |
| `/data.html` | VRSBench, task mixture, honest note on subset scale. |
| `/status.html` | Live health, measured results, what is not built yet. |

## Run it — standalone

```bash
git clone <your-repo-url> && cd satquery
pip install -r requirements.txt
streamlit run app.py
```

Or with make:

```bash
make install     # runtime deps only
make run         # launch the app
make check       # everything CI runs: tests, benchmarks, demo rehearsal
```

Three dependencies. No GPU, no accounts, no dataset downloads, works offline.
Pick **Sample scene** in the app and the whole change-detection demo runs with
no network at all.

```bash
python -m tests.test_contract     # week 1 - interfaces          (9)
python -m tests.test_week2        # week 2 - fallback, metrics  (16)
python -m tests.test_week3        # week 3 - plan, fuse, render (25)

python -m eval.run_eval           # router + validator benchmark
python -m eval.change_eval        # change localisation on synthetic scenes
python -m demo.script             # rehearse the demo, write deck assets
```

GPU work (Kaggle / Colab only) — see `train/kaggle_setup.md`:

```bash
pip install -r requirements-train.txt
python -m data.subsets --dataset bigearthnet --n 4000
python -m train.qlora --dry-run --datasets bigearthnet
python -m train.qlora --datasets bigearthnet rsvqa
python -m eval.run_eval --answers bigearthnet rsvqa --with-vlm
```

---

## Repository layout

```
satquery/
├── README.md                 you are here
├── CONTRIBUTING.md           team working agreement - read before pushing
├── LICENSE                   MIT
├── Makefile                  make install / run / test / eval / demo / check
├── pyproject.toml            package metadata + ruff config
├── requirements.txt          runtime: streamlit, pillow, numpy. That is all.
├── requirements-train.txt    GPU only: torch, transformers, peft, bitsandbytes
├── .env.example              copy to .env, fill in keys, never commit
├── .github/workflows/        CI: all suites + benchmarks + demo rehearsal
│
├── ├── config.py                 all tunables + env vars. No secrets in the repo.
├── contracts.py              THE FROZEN CONTRACT - do not edit alone
├── pipeline.py               parse → validate → plan → execute → fuse → ground
├── render.py                 evidence overlays: boxes, heatmaps, before/after
├── app.py                    Streamlit UI
│
├── router/
│   ├── parser.py             rule parser + LLM parser (cached, fallback-safe)
│   ├── llm_backends.py       Groq / Gemini / offline, stdlib urllib only
│   ├── validator.py          can these images answer this question?
│   ├── planner.py            TaskSpec → ExecutionPlan  ← the agentic layer
│   ├── fusion.py             several outputs → one answer, veto-aware
│   └── registry.py           capability-based model lookup
│
├── models/
│   ├── base.py               SpecialistModel interface
│   ├── heuristic.py          CPU baselines - real spectral + pixel-diff
│   ├── change.py             typed transitions + heatmap + co-registration
│   ├── rsvqa.py              ← the seam: wraps training/'s run_vqa()
│   └── vlm.py                4-bit VLM + LoRA, lazy load, never crashes
│
├── data/
│   ├── prompts.py            instruction templates - train AND infer
│   └── subsets.py            stream HF datasets into small reproducible sets
│
├── train/
│   ├── qlora.py              QLoRA fine-tune, resumable, epoch checkpoints
│   └── kaggle_setup.md       step-by-step for the free tier
│
├── eval/
│   ├── metrics.py            exact match, multi-label F1, documented norms
│   ├── run_eval.py           router / validator / answer suites
│   └── change_eval.py        change localisation on synthetic scenes
│
├── api/
│   ├── main.py               FastAPI: /api/ask, /health, /models, /scenes
│   └── schemas.py            response shapes, mirroring contracts.py
│
├── training/                 ← Person 1's module, vendored unchanged
│   ├── configs/              model.yaml (backbone + LoRA), training.yaml
│   ├── scripts/              prepare_dataset, train_lora, evaluate, inference
│   └── outputs/              (gitignored) adapters land here
│
├── site/
│   ├── index.html            launch page
│   ├── console.html          live console, talks to the API
│   ├── system.html           architecture + live registry
│   ├── data.html             VRSBench
│   ├── status.html           live health + honest accounting
│   └── assets/               shared css + nav/api client
│
├── demo/
│   ├── samples.py            6 synthetic scenes, no network needed
│   └── script.py             rehearsal + deck-ready PNG assets
│
├── site/
│   └── index.html            scroll-driven launch page. Open it directly.
│
└── tests/
    ├── test_contract.py      week 1 - interfaces           (9)
    ├── test_week2.py         week 2 - fallback, metrics   (16)
    └── test_week3.py         week 3 - plan, fuse, render  (25)
```

Nothing gitignored is precious: `.data/`, `.checkpoints/`, `eval/reports/` and
`demo/output/` are all regenerable with a make target.

## Six stages

```
query ─▶ parse ─▶ validate ─▶ plan ─▶ execute ─▶ fuse ─▶ ground ─▶ Answer
         │         │           │        │          │
      TaskSpec  may REFUSE  ExecPlan  [outputs]  veto-aware
```

The plan is produced **before** anything runs and is attached to the Answer.
"Why did it do that" has a printed answer, and the UI renders it.

---

## The four claims this project makes

**1. The contribution is the orchestration layer, not the models.**
`router/planner.py` decides what runs and says why; `router/fusion.py` combines
results. Adding a capability means registering a model — routing logic never
changes.

**2. Every answer is grounded.** Answers carry `Evidence` with normalised
boxes or a change heatmap. When an evaluator asks "how do we know it isn't
making this up", you point at the screen.

**3. It refuses honestly — at two levels.** The validator rejects a *question*
the inputs cannot answer. The co-registration check vetoes *inputs* that are
unsound: hand it two tiles of different places and it says so instead of
producing a confident, meaningless change map.

**4. Disagreement is surfaced, not hidden.** Vote fusion reports when models
conflict and lowers confidence accordingly.

---

## Integrating the trained model

`training/` is Person 1's module, vendored unchanged. It exposes exactly one
function:

```python
run_vqa(image_path, query, adapter_dir) -> {task, answer, confidence,
                                            regions, mask_path, metadata}
```

`models/rsvqa.py` is the **only** file in this repo that knows about it.
Swap RS-LLaVA for GeoChat, retrain, change LoRA rank — nothing else moves.

Two translations that fail silently if wrong, so both are unit-tested:

| Their side | Our side | Note |
|---|---|---|
| `bbox [x, y, w, h]` **pixels** | `bbox (x0,y0,x1,y1)` **normalised** | wrong = box in the wrong place, no exception |
| `image_path` (a file) | `PIL.Image` (in memory) | written to a temp file, cleaned up after |

To bring it online:

```bash
cd training
pip install -r requirements.txt          # GPU box only
python scripts/prepare_dataset.py --output_dir data/vrsbench_processed
python scripts/train_lora.py --config configs/model.yaml \
    --train_config configs/training.yaml \
    --data_dir data/vrsbench_processed --output_dir outputs/lora_vqa_v1
```

The API picks it up on next start — `/api/health` flips
`rsvqa_adapter_present` to true and the console banner stops saying
"classical baselines". **Until then it says so on every page**, which is
deliberate: a demo that implies a fine-tuned VLM while running heuristics is
how teams get caught in questions.

Point it elsewhere with `SATQUERY_RSVQA_ADAPTER=/path/to/adapter`.

## Current numbers

```
router (rule-v1)           23/30    76.7%
validator                   7/7    100.0%
change localisation         4/4    positives localised (hit rate 100%)
                            2/2    negatives correctly reported no change
tests                     70/70    passing
```

Mean best IoU on change is 9.2% — low **by construction**, because the
detector reports 8×8 grid cells while ground truth is a large free-form
region. Hit rate is the metric that means something here. Both are reported;
do not quote one without the other.

The router set deliberately contains cases the rule parser fails
("Has the glacier retreated?", "2019 vs 2023"). Closing that gap is what the
LLM parser is measured on:

```bash
export SATQUERY_LLM_BACKEND=openai_compat
export SATQUERY_LLM_KEY=...            # Groq free tier
python -m eval.run_eval --with-llm
```

---

## Bugs the demo rehearsal caught

Kept here because they are the argument for rehearsing, and all four are now
locked by tests in `tests/test_week3.py`.

| Bug | Why it happened | Fix |
|---|---|---|
| Real deforestation reported as "two different places" | Luminance correlation collapses when ~⅓ of a tile genuinely changes | Primary co-reg signal is now the *stable cell fraction*, robust to large real change |
| Sensor noise reported as change | Threshold was purely relative — some cell is always 1.5× the mean of nothing | Added an absolute floor |
| One transition narrated three times | One transition spans several adjacent grid cells | Collapse by (transition, compass sector) |
| A veto concatenated with the nonsense it vetoed | Fusion treated all outputs equally | `ModelOutput.veto` suppresses siblings |

Plus rendering: labels overlapped and clipped, and an em-dash rendered as a
tofu box in PIL's default font. Both fixed; overlap avoidance is in
`render.py`, ASCII fallback is `_ascii()`.

---

## Design decisions worth defending

**Heuristics stay registered forever, at priority 0.** They are the floor the
router falls back to when the VLM is absent or fails to load. Keeps the demo
alive, and means every result is reported against a real baseline.

**Fusion never invents content.** It selects, orders and joins text the models
actually produced. A fuser that paraphrases adds a hallucination surface with
no model behind it.

**`needs_pair` is set by the contract, not the model.** If the LLM says
"change" but claims no pair is needed, the contract wins.

**Prompts live in exactly one file.** Training and inference both import
`data/prompts.py`. Mismatched prompts are the most common fine-tuning bug in
projects like this.

**Confidence from token probability is not calibrated.** Rough signal only.
Say so if asked; do not put it on a slide as if it were a probability.

---

## Who owns what

| Area | Files | Week 3 job |
|---|---|---|
| Models / fine-tuning | `models/vlm.py`, `train/` | QLoRA run, adapter to HF |
| Router | `router/parser.py`, `planner.py`, `fusion.py` | Close the 7-case gap |
| UI + render | `app.py`, `render.py` | Overlay polish on real Sentinel tiles |
| Evaluation | `eval/` | Grow `ROUTER_CASES` toward 200 |
| Data | `data/subsets.py` | Subsets built once, pushed to HF Datasets |
| Deck | `demo/script.py` | Rehearse all six steps until it is boring |

**Nobody edits `contracts.py` alone.** Week 3 extended it additively —
`EvidenceKind`, `ExecutionPlan`, `ModelOutput.veto`, `Evidence.grid` — all
defaulted, so nothing written earlier broke.

---

## Known limits — state these, do not hide them

- Heuristic baselines use a visible-band greenness index as an NDVI stand-in,
  because an RGB upload has no NIR band.
- Grounding and change report 8×8 grid cells, not segmentation masks.
- Synthetic scenes are for testing and fallback. Real Sentinel imagery goes in
  the pitch; never imply synthetic numbers are benchmark results.
- `normalise()` strips articles, so a label that *is* an article vanishes.
  Harmless for land cover; documented and tested so it stays visible.

Naming what your system cannot do reads as rigour. Pretending otherwise gets
you caught in the Q&A.

## Things that will bite you

- Kaggle sessions die at ~9–12 hrs. Checkpoint every epoch, no exceptions.
- `/kaggle/working` is wiped on session end. Push the adapter before closing.
- HF dataset IDs move. Update `data/subsets.py::SOURCES`, nowhere else.
- Scope creep. Two capabilities working with numbers beats five half-working.
