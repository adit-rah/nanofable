"""Capability gate (spec §8) — frozen predicate over judge scores + val perplexity.

Capable iff BOTH:
- coherence: mean per-completion judge score >= 4.0 over the fixed 200 prefixes, AND
- perplexity: val PPL <= 1.5 * best fp16 val PPL in the sweep.

A tier-separation claim counts only if the 95% CI on the mean judge score does not straddle
4.0; `ci_straddles_4` flags the indistinguishable case.
"""

from __future__ import annotations

import math

COHERENCE_THRESHOLD = 4.0
PPL_MULTIPLIER = 1.5


def mean_and_ci(scores: list[float]) -> tuple[float, float]:
    """Return (mean, 95% CI half-width) using the normal approximation of the SEM."""
    n = len(scores)
    mean = sum(scores) / n
    if n < 2:
        return mean, float("inf")
    var = sum((s - mean) ** 2 for s in scores) / (n - 1)
    sem = math.sqrt(var) / math.sqrt(n)
    return mean, 1.96 * sem


def capability_gate(judge_scores: list[float], val_ppl: float, best_fp16_ppl: float) -> dict:
    """Evaluate the frozen capability gate for one configuration.

    `judge_scores` is the list of per-completion scores (mean of the three axes) over the
    200 prefixes, pooled across seeds.
    """
    mean, half = mean_and_ci(judge_scores)
    ci_low, ci_high = mean - half, mean + half
    coherence_pass = mean >= COHERENCE_THRESHOLD
    ppl_threshold = PPL_MULTIPLIER * best_fp16_ppl
    ppl_pass = val_ppl <= ppl_threshold
    return {
        "mean": mean,
        "ci_low": ci_low,
        "ci_high": ci_high,
        "ci_straddles_4": ci_low < COHERENCE_THRESHOLD < ci_high,
        "coherence_pass": coherence_pass,
        "ppl_pass": ppl_pass,
        "ppl_threshold": ppl_threshold,
        "capable": coherence_pass and ppl_pass,
    }
