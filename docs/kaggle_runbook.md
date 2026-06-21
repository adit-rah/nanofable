# Kaggle runbook (GPU execution)

The notebook [`notebooks/kaggle_run.ipynb`](../notebooks/kaggle_run.ipynb) runs the GPU-side
of the study. This doc is the short companion: what runs, in what order, and the gotchas.

## What runs (in order)
1. **Build data** — tokenize TinyStories → `train.bin`/`val.bin` memmaps (CPU, minutes).
2. **Calibration (once, before the sweep)** — score good / mediocre / degenerate references
   through the frozen rubric + Qwen2.5-7B judge → `eval/calibration.md`. **Inspect it.**
3. **Sweep (Phase A)** — train the 16 runs (4 tiers × fp16/ternary × 2 seeds) to 500M tokens.
4. **Eval (Phase B)** — load the judge once, score the 200 frozen prefixes per run → `eval.json`.
5. **Plot** — `frontier.png` (coherence vs total bytes) + the smallest-capable headline.
6. **Writeup** — fill `docs/writeup.md`.

## Kaggle settings (right-hand panel)
- **Accelerator:** P100 (single 16GB) or T4. The harness uses one GPU.
- **Persistence:** *Files only* — keeps `/kaggle/working/runs` across sessions so the sweep
  resumes instead of restarting.
- **Internet:** On (pip + TinyStories + Qwen judge downloads).

## Resume / session budget
- Sessions cap at 12h; quota ~30 GPU-h/week. Full sweep ≈ 10–20 GPU-h (~2 sessions).
- The sweep is **idempotent**: each run checkpoints (`ckpt_latest.pt`) and finished runs carry
  a `DONE` marker that is skipped. If a session times out, just re-run the sweep cell next
  session — it continues from the last checkpoint. No run exceeds ~2h, so none is un-resumable.
- Cap a session by setting `ONLY` in the sweep cell to a subset of the matrix.

## Gotchas
- **Calibration is a gate, not a formality.** Before trusting any capability claim, confirm in
  `eval/calibration.md`: good refs clear ~4.0, garbage well below, and **intra-judge std <
  good−bad gap**. If the judge is too noisy, swap/upgrade it (an Anthropic-API pass via
  `AnthropicJudge` for borderline configs is permitted) before sweeping.
- **Frozen artifacts must predate the sweep.** The tokenizer, 200 prefixes, rubric, and judge
  prompt are committed in the repo; do not regenerate them on Kaggle.
- **OOM on the large tier?** Lower `micro_rows` in the sweep cell (smaller grad-accum
  micro-batch). fp16 autocast is already on for GPU runs.
- **Don't hand-edit results.** `metrics.csv` / `eval.json` are produced by the harness; the
  plot reads them.
