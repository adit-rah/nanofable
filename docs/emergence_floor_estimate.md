# How small *could* it be? — first-principles floor estimates & the post-sweep fitting procedure

**Status:** scratch estimation + frozen procedure. This doc is NOT part of the pre-registered
capability gate (spec §8) and never overrides it. It exists to (a) sanity-check that the tier
ladder brackets the plausible emergence floor, and (b) freeze — before sweep results are in —
the procedure for extrapolating that floor from the sweep data.

## Units, first (prevents misreading the training logs)

Train/val loss is cross-entropy in **nats per token** on the 4,096-token vocab.
Conversions: `bits/token = nats × 1.4427`; `PPL = e^nats`; random guessing = ln(4096) = 8.32
nats (PPL 4096). Example: the tiny fp16 run's ~2.77 nats = 4.0 bits/token ≈ ~1 bit/char on
simple English — near the entropy of the distribution itself, not "high loss." Raw CE is not
comparable across tokenizers: a larger vocab packs more entropy into each token.

## Lens 1 — parameter accounting (exact, this repo's tiers)

Body ≈ `12·L·d²` per the standard accounting (4d² attention + 8d² SwiGLU MLP); embeddings
`V·d` (tied head). Measured from `build_model` + `count_bytes`:

| tier   | params     | emb (tied) | body       | fp16 bytes | ternary bytes | tern/fp16 |
|--------|-----------:|-----------:|-----------:|-----------:|--------------:|----------:|
| tiny   |  1,377,408 |    524,288 |    853,120 |  2,752,512 |     1,216,896 |     0.442 |
| small  |  5,868,800 |  1,048,576 |  4,820,224 | 11,730,944 |     3,048,573 |     0.260 |
| medium | 15,735,168 |  1,572,864 | 14,162,304 | 31,457,280 |     5,941,606 |     0.189 |
| large  | 27,795,968 |  2,097,152 | 25,698,816 | 55,574,528 |     9,268,213 |     0.167 |

Two structural facts fall out:
- **Vocab is the floor-setter at the bottom.** At tiny, embeddings are 38% of params and (kept
  fp16) 86% of ternary bytes. With a 50k GPT-2 vocab, tiny would be ~6.4M embedding params
  before one layer of compute — the spec §5 gotcha, quantified. Shrinking effective vocab is
  the real lever TinyStories pulls; architecture is secondary.
- **Ternary's compression ratio improves with size** (0.44 → 0.17) because only the body
  ternarizes. So ternary bends the bytes-frontier hardest exactly where models are larger —
  whether that beats fp16 at the *capable* end is the experiment.

(For reference: ternary tiny at 1.19 MB fits a 1.44 MB floppy.)

## Lens 2 — description length / capacity

Treat the model as storing the learnable regularity of the distribution at ~3.6 bits/param
(empirical memorization-capacity estimates for transformers). Hand-estimate of the TinyStories
distribution's content: compact grammar (~10⁴–10⁵ bits) + ~1.5k word usage profiles
(~10²–10³ bits each) + toddler-grade commonsense → order **10⁵–10⁶ bits** total. Dividing:
**~30k–300k params of "content."** Optimization overhead (nets train badly at exact capacity;
lottery-ticket overparameterization) plus embedding scaffolding plausibly multiplies that by
3–10×, putting the practical floor around **~1M params** — i.e., right at the tiny tier
(0.85M body / 1.38M total). Capacity check per tier at 3.6 b/param: tiny ≈ 5.0 Mbit,
small ≈ 21 Mbit, medium ≈ 57 Mbit, large ≈ 100 Mbit — all ≥ the estimated task content,
so *capacity* should not be the binding constraint above tiny; trainability and the judge
gate decide.

**Prediction this lens makes (recorded before eval results):** tiny is marginal — it sits at
the floor, not above it — small/medium should clear the gate if anything does, and the
interesting crossing is likely between tiny and small.

## Lens 3 — scaling-curve extrapolation (FROZEN PROCEDURE, run after the sweep)

The rigorous version. Executed once per precision arm when all 16 runs + evals are done:

1. **Data:** per tier, final `val_ppl` from `metrics.csv` (last row), averaged over seeds
   {0,1}; `L = ln(PPL)` in nats. `N` = **non-embedding (body) params** (primary fit; report
   the total-params fit as sensitivity).
2. **Fit** `L(N) = L∞ + (N_c/N)^α` by least squares (3 free params, 4 points → 1 df).
   Report residuals; refit dropping each tier in turn and report the spread of `N_c, α, L∞`
   as the honest instability of a 4-point fit.
3. **Locate the coherence-equivalent loss `L*`:** from each run's `eval.json`, pair
   (judge mean, val PPL) per tier; linearly interpolate the PPL at which the judge mean
   crosses 4.0, between the adjacent tiers that straddle the gate. `L* = ln(PPL*)`. If no
   tier clears 4.0, or all do, report "not bracketed" — no extrapolation.
4. **Solve** `N* = N_c · (L* − L∞)^(−1/α)` → the estimated minimum body params for capable
   English, per arm.
5. **Convert to bytes** with each arm's bytes/param (fp16: 2·N* + 2·V·d*; ternary:
   1.58/8·N* + 2·V·d* + scales, per §6), using the d of the nearest tier. Compare against
   the direct read-off (leftmost capable tier) — the fit interpolates *between* tiers; the
   gate result stays the headline.
6. **Uncertainty:** propagate seed spread on PPL and the ±CI on judge means through steps
   3–4; report N* as a range, not a point.

## Caveats (kept deliberately, from the original scratch reasoning)

None of this is a lower bound. Bits/param assumes reachable capacity; optimization wants
overparameterization even when a smaller net could represent the function; the task-content
estimate is hand-waved to an order of magnitude. Most importantly, **the floor is a property
of the chosen distribution, not of English**: TinyStories engineered a distribution simple
enough for ~1–10M params. Move the data, move the floor — which is exactly the spec §11
stretch direction (restrict the vocabulary/domain further and watch the threshold slide).
