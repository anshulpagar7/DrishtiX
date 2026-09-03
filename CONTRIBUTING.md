# Working agreement

Six people, four weeks, one deadline. These rules exist so parallel work stays
parallel instead of turning into merge conflicts in week 4.

## The one hard rule

**Nobody edits `contracts.py` alone.**

`TaskSpec`, `ModelOutput`, `Evidence`, `ExecutionPlan` and `Answer` are what
every module depends on. Changing a shape breaks other people's work silently.

If you need a change:

1. Raise it with the team first.
2. Add fields **with defaults**, never rename or remove. Week 3 added
   `EvidenceKind`, `ExecutionPlan`, `ModelOutput.veto` and `Evidence.grid`
   this way, and nothing written in weeks 1–2 broke.
3. Run all three test suites before pushing.

`tests/test_contract.py` exists to catch someone quietly changing a shape.
If it fails, that is the system working.

## Ownership

| Area | Files | Owner |
|---|---|---|
| Models / fine-tuning | `models/vlm.py`, `train/` | |
| Router | `router/parser.py`, `planner.py`, `fusion.py` | |
| UI + rendering | `app.py`, `render.py` | |
| Evaluation | `eval/` | |
| Data | `data/` | |
| Deck + demo | `demo/script.py` | |

Fill the owner column in your first meeting. Two people editing one file is
how you lose an afternoon.

## Before you push

```bash
make check
```

Runs all three suites, both benchmarks, and the demo rehearsal. CI runs the
same thing, so a green local run means a green PR.

## Adding a model

Never edit routing logic. Implement `SpecialistModel`, register it, done.

```python
class MyModel(SpecialistModel):
    model_id = "my-model-v1"          # unique, appears in the UI and in evals
    supported_tasks = {TaskType.VQA}
    priority = 15                      # 0 heuristics, 10 classical, 20+ tuned

    def _run(self, spec, images):
        return ModelOutput(
            answer="...",
            confidence=0.8,
            model_id=self.model_id,
            evidence=[Evidence(image_index=0, bbox=(0.1, 0.1, 0.4, 0.4))],
        )
```

Then add it in `pipeline.build_registry()`.

Three requirements:

- **`can_handle()` must be cheap.** No model loading, no network. The router
  calls it on every query.
- **Load weights lazily**, inside `_run` or an `ensure_loaded()`, never in
  `__init__`. The UI must start instantly on a machine with no GPU.
- **Always return evidence.** An answer with no evidence is a hallucination
  with good manners.

## Adding a task type

1. Add it to `TaskType` in `contracts.py` (team decision — see above).
2. Add patterns to `RuleParser` and a line to the `LLMParser` system prompt.
3. Add an instruction template in `data/prompts.py`.
4. Add at least four labelled cases to `ROUTER_CASES` in `eval/run_eval.py`.
5. Register a model that handles it.

Step 4 is not optional. An untested task type will misroute on stage.

## Evaluation discipline

- **Never delete a failing test case to raise a number.** The seven cases the
  rule parser fails are in the benchmark on purpose. A benchmark you score
  100% on measures nothing.
- Report synthetic-scene results **as synthetic**. Never imply they are
  Sentinel benchmark numbers.
- Report hit rate and IoU together. Quoting one without the other is
  misleading, and an evaluator will ask.

## Demo discipline

Run `make demo` after every significant change. It caught four real bugs in
week 3 that code review did not — a false veto on genuine change, noise
reported as change, duplicated narration, and a veto being ignored.

Before demo day run `make assets` and put the PNGs in the deck. If the live
demo dies, you keep talking.

## Commits

Small, one concern each, present tense: `add semantic change detector`,
`fix label overlap in evidence overlay`. Reference the file area when it is
not obvious. Nobody needs a changelog, but somebody will need to find when a
number changed.
