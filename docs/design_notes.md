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

## Eval-instrument audit & corrections (2026-07-12, post-sweep / pre-greedy-scores)

Full frozen rule text lives in `eval/calibration.md`; this is the decision ledger.

### Findings (audit of calibration + eval chain, triggered by sampled scores ≪ gate)
1. **Decoding asymmetry (the defect):** sweep models were judged on temp-1.0/top_k-40
   samples, but every calibration reference was decoding-free (gold, degenerate) or greedy
   (33M). Decoding was never in the §8 freeze list. Measured: 33M greedy 4.378 vs 33M
   sampled 3.687 → under sampling the gate exceeds a known-good 33M model, so it had no
   discriminative power at sweep scales. The sampled-policy gate was never valid.
2. **Silent zeros:** judge parse failures scored 0 indistinguishably from verdicts; judge
   max_new_tokens=32 could truncate the JSON. Fixed: parsed flag + n_parse_failures +
   raw judge text stored; budget 64.
3. **No audit trail:** eval.json didn't store completions. Fixed: stored per prefix.
4. **Vacuous stability check:** the judge decodes greedily, so intra-judge std over
   re-scores is 0 by construction. Not cited as reliability evidence; the n=200 CI is the
   real noise estimate.

### Decisions (all frozen blind — before any greedy sweep-model score existed)
- **Eval decoding = greedy, uniformly** (models + references). The only policy the
  original calibration validated. Instrument correction, not gate tuning; sampled scores
  retained and reported alongside.
- **Primary gate unchanged** (≥4.0 + PPL ≤ 1.5× best fp16, spec §8).
- **Secondary anchor is measured, not declared:** published judge scores don't transfer
  across instruments, so the TinyStories ladder (1M/3M/8M/28M/33M) is scored greedy n=200
  through the frozen judge; anchor = smallest checkpoint with mean ≥ 4.0. Checked
  2026-07-12: no post-2023 work lowers the published coherence floor below this family.
- **What gets used:** greedy eval.json scores → frontier + gates; ladder → anchor +
  external benchmark line; sampled scores → writeup appendix only.

### Eval performance decisions (2026-07-12)
- Judge is loaded once and reused across the reference ladder and all runs; eval
  parallelizes across both T4s with per-run claim files (same pattern as the sweep).
- Deliberately NOT done (caution > speed): KV-cache in `generate` (touches the
  experiment-critical model path), batched judge calls (padding subtly changes logits).
  Eval-time compute is not a reported variable, so these are safe to skip.
