# tinychat — ternary vs fp16 emergence frontier on TinyStories

Research harness for the study specified in
[`specs/idea2_ternary_emergence_frontier_spec.md`](specs/idea2_ternary_emergence_frontier_spec.md):
how small (in **honest bytes**) can a coherent English language model get, and does ternary
(1.58-bit) weighting push that frontier down versus fp16? See
[`docs/frozen_config.md`](docs/frozen_config.md) for the frozen controlled variables and
[`docs/design_notes.md`](docs/design_notes.md) for the resolved design decisions.

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt   # torch, tokenizers, datasets, transformers, ...
```

Tests run offline with no GPU:

```bash
.venv/bin/python -m pytest      # full unit suite
```

## Pipeline (in order)

| Step | Command | Output | Needs |
|------|---------|--------|-------|
| 1. Tokenizer (frozen) | `python scripts/build_tokenizer.py` | `artifacts/tokenizer/tokenizer.json` | TinyStories text (CPU) |
| 2. Data | `python scripts/build_dataset.py` | `artifacts/data/{train,val}.bin` | tokenizer (CPU) |
| 3. Prefixes (frozen) | `python scripts/make_prefixes.py` | `eval/prefixes.jsonl` (200) | tokenizer + TinyStories (CPU) |
| 4. Calibration (frozen) | `python scripts/run_calibration.py` | `eval/calibration.md` | judge model (GPU) |
| 5. Sweep | `python scripts/run_sweep.py` | `runs/<tier>_<prec>_<seed>/` | data + judge (GPU) |
| 6. Plot | `python scripts/plot_frontier.py` | `docs/frontier.png` | run CSVs (CPU) |

Steps 1–3 are one-time **frozen artifacts** committed before the sweep (spec §8). Steps 4–5
need the Kaggle GPU (≤12-h sessions); the sweep is idempotent — re-run to resume, finished
runs (those with a `DONE` marker) are skipped. The 1–2 page finding goes in
[`docs/writeup.md`](docs/writeup.md) after the sweep (Definition of Done).

## Layout

- `src/tinychat/` — model (`model.py`, `bitlinear.py`, `rope.py`), byte accounting
  (`bytes.py`), data (`data.py`), training (`train.py`), sweep (`sweep.py`), plotting.
- `eval/` — frozen rubric, judge prompt, 200 prefixes; judge backends, capability gate,
  eval runner.
- `scripts/` — one-shot generators and the sweep / plot entrypoints.
- `tests/` — mirrors `src/` and `eval/`; offline, no GPU.
- `runs/`, `artifacts/data/` — generated, gitignored.

## The one decision that makes or breaks this

A custom **4k** BPE vocab (not the 50k GPT-2 vocab) keeps the embedding/head table small so
the transformer blocks dominate the byte total and the ternary savings actually show up. See
[`docs/byte_accounting.md`](docs/byte_accounting.md).
