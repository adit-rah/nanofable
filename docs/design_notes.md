# Design Notes — rationale & resolved ambiguities

This file records *why* the frozen config is what it is, and how each spec ambiguity was
resolved. The spec is the source of truth for *what*; this is the *why*.

## Resolved spec ambiguities

### 1. ctx conflict (§4 vs §5)
§5's tier table lists `ctx = 256` for tiny/small and `512` for medium/large. §4 says to
**hold context length fixed** across the sweep. These conflict. **Resolution: ctx = 512 for
all tiers**, honoring §4's "hold fixed" controlled-variable requirement (apples-to-apples
matters more than the §5 starting suggestion, which §5 itself calls "starting points").

### 2. Byte accounting and RMSNorm gains (§6) — EXCLUDE
The §6 formula has exactly three terms: `ternary_block_bytes + fp16_embedding_and_head_bytes
+ scale_factor_bytes`. It does not mention RMSNorm gains. **Resolution (user decision): count
`count_bytes` literally as the three §6 terms; RMSNorm gains are NOT counted.** Rationale:
the formula is frozen pre-registration and the headline number must match it verbatim; norm
gains are negligible (~`n_embd` per norm) and identical across both precision arms, so they
cannot change the fp16-vs-ternary comparison. Documented further in `byte_accounting.md`.

### 3. Peak LR fixed vs ternary instability (§12) — FIXED
Peak LR is held at **3e-4 for all 16 runs**. §12 flags ternary training instability at tiny
scale. **Resolution (user decision): do not retune per-arm.** If a ternary run collapses, that
is a *reported finding*, not a reason to change one arm's LR — doing so would break the
apples-to-apples contract (§4, §12). Any LR change must be applied to **all 16 runs** (re-run
fp16 too).

### 4. BitLinear scale granularity (§7) — per-tensor
§7 permits per-layer or per-output-channel absmean scaling. **Resolution (user decision):
per-tensor** (one fp16 scale per layer), matching cited BitNet practice and keeping the scale
byte count minimal (1 fp16 / layer). Per-channel is left as a possible stretch toggle.

## The embedding-dominance decision (§5 gotcha — the #1 killer)
We train a **custom 4k BPE tokenizer** rather than reusing the 50k GPT-2 vocab. At these tiny
sizes the embedding/LM-head table can dominate parameter count; since embeddings are not
ternarized, a large vocab would make ternary block savings invisible in the byte total. With
vocab=4096 the transformer blocks dominate, so the ternary byte savings actually show up. The
test `test_ternary_smaller_than_fp16` guards this at every tier.

## Scope of the current build
This session builds **code + tests + hand-authored frozen artifacts** (rubric, judge prompt)
and all generator scripts. The 16-run sweep, the judge calibration run (`eval/calibration.md`),
real eval results, the real frontier PNG, and `docs/writeup.md` are produced later on the
Kaggle GPU environment (spec §9). The committed tokenizer and 200 prefixes are CPU-generated
one-time artifacts and must exist before any sweep run (spec §8 freeze ordering).
