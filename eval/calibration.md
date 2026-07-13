# Calibration & Judge Reliability (FROZEN)

- good (gold) mean: 4.568  (95% CI ±0.093)
- bad (degenerate) mean: 0.232
- good−bad gap: 4.337
- intra-judge std (mean over 3 re-scores): 0.0000

- rank-ordering good > bad: True
- judge reliable (intra_std < good−bad gap): True

If rank-ordering fails OR intra_std >= good−bad gap, STOP and upgrade the judge before freezing the gate (spec §8).
- mediocre (TinyStories-33M) mean: 4.378  (addendum)
- reference (TinyStories-33M, sampled temp1.0/topk40, n=50) mean: 3.687  (addendum, 2026-07-12)

## Decoding-policy correction (frozen 2026-07-12, before any greedy sweep scores existed)

**Defect found in a post-sweep audit:** evaluation decoding was never in the §8 freeze list.
The gate was calibrated only on decoding-free references (gold text, synthetic degenerate
text) and a *greedy* TinyStories-33M reference (4.378), while sweep models were judged on
temperature-1.0/top_k-40 *samples*. Scoring the 33M reference under that sampled policy
gives **3.687 < 4.0**: under sampled decoding the gate sits above what a known-good 33M
model achieves, so it has no discriminative power at the sweep's scales. The sampled-policy
gate was therefore never valid; this is an instrument correction, not gate tuning.

**Frozen rules (set blind — no greedy sweep-model scores had been computed):**
1. **Eval decoding = greedy** (temperature→0), uniformly, for every sweep model and every
   reference model. Greedy is the only policy the original calibration validated.
2. **Primary gate unchanged:** mean judge score ≥ 4.0 AND val PPL ≤ 1.5 × best fp16
   (spec §8, as pre-registered). Reported as the primary result.
3. **Reference-anchored secondary capability line:** a config is *reference-capable* iff its
   mean greedy judge score ≥ the mean greedy judge score of **roneneldan/TinyStories-1M**
   over the same 200 prefixes, same judge, same prompt. TinyStories-1M is the smallest
   checkpoint the TinyStories authors published and defend as producing grammatical,
   mostly-coherent stories, and is scale-matched to the sweep's tiny tier; it is chosen for
   those external reasons, before its score or any sweep model's greedy score is known.
   CI straddling the anchor ⇒ reported as indistinguishable, per the spec's power rule.
4. The original sampled-policy scores (five runs) are retained in the backup and reported
   alongside the greedy results in the writeup.

### Revision to rule 3 (2026-07-12, still blind — no greedy sweep-model scores computed)

Rule 3's anchor ("TinyStories-1M by fiat") is replaced before any measurement: published
judge scores (GPT-4, 1–10, 2023) do not transfer to our instrument (Qwen2.5, 0–5, our
rubric), so no fixed checkpoint can be *declared* the weakest capable one. Instead:

**Anchor rule:** score the published ladder — roneneldan/TinyStories-{1M, 3M, 8M, 28M, 33M}
— greedy, n=200, through the frozen judge. The anchor is the **smallest checkpoint whose
mean ≥ 4.0** (the calibrated primary bar). A sweep config is *reference-capable* iff its
mean greedy judge score ≥ that anchor's mean (CI straddling ⇒ indistinguishable). The full
ladder is reported in the writeup as the external benchmark line, and no post-2023 work
lowers the published coherence floor below this family (checked 2026-07-12; see
docs/related_work.md — stories260K is below coherence, nothing smaller claims it).

### Ladder results & anchor determination (2026-07-12, recorded before any greedy
### sweep-model score was observed)

- reference (TinyStories-1M, greedy, n=200) mean: 2.423  (95% CI ±0.123)
- reference (TinyStories-3M, greedy, n=200) mean: 3.330  (95% CI ±0.127)
- reference (TinyStories-8M, greedy, n=200) mean: 4.232  (95% CI ±0.112)
- reference (TinyStories-28M, greedy, n=200) mean: 4.447  (95% CI ±0.101)
- reference (TinyStories-33M, greedy, n=200) mean: 4.378  (95% CI ±0.106)

Rank-ordering is monotone through 28M (2.423 < 3.330 < 4.232 < 4.447; 28M vs 33M CIs
overlap — a plateau, reported as indistinguishable). 33M-greedy exactly reproduces the
pre-correction measurement (4.378), confirming instrument consistency across the judge
token-budget change.

**Anchor = TinyStories-8M (mean 4.232, CI [4.120, 4.344] — clears 4.0 without
straddling).** The reference-capable line for sweep configs is therefore mean ≥ 4.232,
with CI straddling of 4.232 reported as indistinguishable. Note the calibrated 4.0 bar
falls between the published 3M (3.330) and 8M (4.232) checkpoints — "capable" as frozen
means "closer to 8M-published quality than 3M's".
