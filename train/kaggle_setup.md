# Running training on Kaggle (free)

Every free GPU account you have is ~30 hrs/week. Six people is ~180 hrs/week.
You will not run out of compute. You will run out of care, so read this once
properly.

## One-time setup

1. **Verify your phone number.** Settings → Phone Verification. Unverified
   accounts get **no GPU**. People forget this and lose a week.
2. New Notebook → right panel → **Accelerator: GPU T4 x2** (or P100).
3. Right panel → **Internet: On** (needed to pull weights and datasets).

## Notebook cells

```python
# 1. Get the code
!git clone https://github.com/<your-org>/satquery.git
%cd satquery
!pip install -q -r requirements-train.txt
```

```python
# 2. Build the data subsets (once - then push to HF Datasets so the
#    rest of the team pulls identical data instead of five different subsets)
!python -m data.subsets --dataset bigearthnet --n 4000
!python -m data.subsets --dataset rsvqa --n 3000
```

```python
# 3. Check the wiring before spending GPU hours
!python -m train.qlora --dry-run --datasets bigearthnet rsvqa
```

```python
# 4. Train
!python -m train.qlora --datasets bigearthnet rsvqa --epochs 2
```

```python
# 5. Score it against the baseline - this is the slide
import os
os.environ["SATQUERY_VLM_ADAPTER"] = ".checkpoints/qlora/adapter"
!python -m eval.run_eval --answers bigearthnet rsvqa --with-vlm
```

## The five things that will actually go wrong

**Session dies mid-training.** Not "if". Kaggle kills sessions at roughly
9–12 hrs and sometimes sooner. `save_strategy="epoch"` is already set — never
turn it off. Resume with:

```bash
python -m train.qlora --datasets bigearthnet --resume .checkpoints/qlora/checkpoint-XXX
```

**OOM on the T4.** 15 GB goes fast. In order: lower `max_len` in
`config.TRAIN`, then raise `grad_accum` while keeping `batch_size` at 1, then
pick a smaller base checkpoint. Do not lower `lora_r` first — it costs quality
and saves almost no memory.

**Output disappears.** `/kaggle/working` is wiped when the session ends. Push
the adapter to the HF Hub or download it before you close the tab:

```python
from huggingface_hub import upload_folder
upload_folder(folder_path=".checkpoints/qlora/adapter",
              repo_id="<you>/satquery-adapter", repo_type="model")
```

The adapter is small — tens of MB, not gigabytes. There is no excuse for
losing one.

**Dataset paths 404.** HF dataset IDs move. Verify each on the Hub and update
`data/subsets.py::SOURCES` rather than scattering IDs through the code.

**Processor/model mismatch.** If `apply_chat_template` errors, your checkpoint
uses a different chat format. Fix it in `data/prompts.py` only — both training
and inference read from there, so fixing one place fixes both.

## Before you close any session

- [ ] Adapter pushed to HF or downloaded
- [ ] `eval/reports/report.md` saved somewhere permanent
- [ ] `run.json` kept — it records base model, LoRA config, record count
- [ ] Losses noted, even roughly. "It went down" is not a result.
