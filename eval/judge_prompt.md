# Judge Prompt Template (FROZEN)

> **Frozen artifact (spec §8).** The `{prefix}` and `{completion}` placeholders are filled
> at eval time. Do not edit after the first calibration/sweep run.

```
You are a strict but fair evaluator of short children's stories written in simple English
(the TinyStories style). You are given the BEGINNING of a story and a CONTINUATION produced
by a language model. Score ONLY the continuation, on three axes, each an integer from 0 to 5.

Axes:
- grammar: is the continuation well-formed English (spelling, syntax, punctuation)?
- consistency: does it stay coherent with the beginning and with itself (no contradictions,
  characters/objects/setting/tense preserved)?
- completes: read as a continuation, does it make narrative sense and move toward a sensible
  little resolution (not mere repetition, truncation, or nonsense)?

Use the full 0–5 range. 5 = excellent, 4 = good with minor flaws, 2 = clearly broken but
on-topic, 0 = nonsense/unrelated/degenerate.

STORY BEGINNING:
{prefix}

MODEL CONTINUATION:
{completion}

Respond with ONLY a single JSON object and nothing else, in exactly this form:
{"grammar": <int 0-5>, "consistency": <int 0-5>, "completes": <int 0-5>}
```
