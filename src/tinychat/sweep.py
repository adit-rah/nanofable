"""Idempotent sweep orchestration (spec §9): the 4×2×2 matrix.

Each run gets a deterministic directory `runs/<tier>_<precision>_<seed>`. A run whose dir
already contains a `DONE` marker is skipped, so re-running `run_sweep` after a session
timeout is safe and resumes only the unfinished work.
"""

from __future__ import annotations

import os

from .config import TIERS
from .train import train_run

PRECISIONS = ("fp16", "ternary")
SEEDS = (0, 1)


def sweep_matrix() -> list[tuple[str, str, int]]:
    """The full (tier, precision, seed) matrix — fp16 before ternary so the fp16 baseline
    (needed for the PPL gate) lands first."""
    return [
        (tier, precision, seed)
        for tier in TIERS
        for precision in PRECISIONS
        for seed in SEEDS
    ]


def run_dir_for(runs_root: str, tier: str, precision: str, seed: int) -> str:
    return os.path.join(runs_root, f"{tier}_{precision}_{seed}")


def run_sweep(
    runs_root: str,
    train_path: str,
    val_path: str,
    only: list[tuple[str, str, int]] | None = None,
    eval_fn=None,
    **train_kwargs,
) -> list[str]:
    """Train (and optionally eval) every matrix entry, skipping finished runs.

    `only` restricts to a subset of the matrix. `eval_fn(run_dir)` runs after training when
    provided (the CLI passes the judge-backed evaluator). Returns the list of run dirs that
    were trained this call.
    """
    trained = []
    for tier, precision, seed in sweep_matrix():
        if only is not None and (tier, precision, seed) not in only:
            continue
        run_dir = run_dir_for(runs_root, tier, precision, seed)
        if os.path.exists(os.path.join(run_dir, "DONE")):
            continue
        train_run(TIERS[tier], precision, seed, run_dir, train_path, val_path,
                  **train_kwargs)
        if eval_fn is not None:
            eval_fn(run_dir)
        trained.append(run_dir)
    return trained
