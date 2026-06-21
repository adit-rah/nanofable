# Coherence Rubric (FROZEN)

> **Frozen artifact (spec §8).** Do not edit after the first calibration/sweep run. The
> capability gate depends on this text being stable. Changing it invalidates the
> pre-registration.

A judge model reads a held-out TinyStories **prefix** and a model-generated **completion**
and scores the completion on three independent axes, each an integer 0–5. The
**per-completion score is the mean of the three axis scores**. A configuration's coherence
score is the mean per-completion score over the fixed set of 200 prefixes (pooled across
seeds).

## Axes

### 1. Grammar (0–5)
Is the completion well-formed English (spelling, morphology, syntax, punctuation)?
- **5** — fluent, essentially error-free.
- **4** — minor slips that do not impede reading.
- **2** — frequent errors; understandable but clearly broken.
- **0** — word salad / non-English / no real sentences.

### 2. Consistency (0–5)
Does the completion stay coherent with the prefix and with itself (characters, objects,
setting, tense, logical continuity; no contradictions or abrupt topic breaks)?
- **5** — fully consistent with the prefix; no contradictions.
- **4** — largely consistent; a small lapse.
- **2** — noticeable contradictions or drift from the prefix.
- **0** — unrelated to the prefix or internally incoherent.

### 3. Completes-sensibly (0–5)
Read as a continuation of the story, does it make narrative sense and move toward a
sensible little resolution (not merely repeating, truncating, or trailing into nonsense)?
- **5** — a natural, sensible continuation that reads as a real story beat.
- **4** — sensible, if a little flat or unfinished.
- **2** — weak: repetitive or aimless but on-topic.
- **0** — degenerate: pure repetition, immediate cut-off, or nonsense.

## Scoring rule
`per_completion = (grammar + consistency + completes) / 3`, in [0, 5].

## Capability threshold (frozen)
A configuration is **coherence-capable** iff its mean per-completion score over the 200
prefixes is **≥ 4.0**. The exact 4.0 placement is confirmed by the sweep-blind calibration
pass (`eval/calibration.md`) before freezing.
