"""Distill the training curves into one committed CSV: results/curves.csv.

Reads every run dir's metrics.csv, averages the two seeds at each logged step, and
writes one row per (tier, precision, tokens_seen). Same contract as
summarize_results.py: this is the committed, inspectable form of the curves the README
plots; raw per-run outputs stay local.

    python scripts/summarize_curves.py                 # reads local/hf_runs, writes results/curves.csv
    python scripts/summarize_curves.py <runs_dir> <out_csv>
"""

from __future__ import annotations

import csv
import math
import os
import sys
from collections import defaultdict

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))

from nanofable.config import TIERS  # noqa: E402


def _run_dirs(runs_dir: str, tier: str, precision: str) -> list[str]:
    out = []
    for name in sorted(os.listdir(runs_dir)):
        parts = name.rsplit("_", 2)
        if len(parts) == 3 and parts[0] == tier and parts[1] == precision:
            out.append(os.path.join(runs_dir, name))
    return out


def summarize(runs_dir: str) -> list[dict]:
    rows = []
    for tier in TIERS:
        for precision in ("fp16", "ternary"):
            # tokens_seen -> [(train_loss, val_loss), ...] across seeds
            by_tokens: dict[int, list[tuple[float, float]]] = defaultdict(list)
            for d in _run_dirs(runs_dir, tier, precision):
                path = os.path.join(d, "metrics.csv")
                if not os.path.isfile(path):
                    continue
                with open(path) as f:
                    for r in csv.DictReader(f):
                        if not r.get("val_loss"):
                            continue
                        by_tokens[int(r["tokens_seen"])].append(
                            (float(r["train_loss"]), float(r["val_loss"]))
                        )
            for tokens in sorted(by_tokens):
                pairs = by_tokens[tokens]
                train = sum(p[0] for p in pairs) / len(pairs)
                val = sum(p[1] for p in pairs) / len(pairs)
                rows.append({
                    "config": f"{tier}_{precision}",
                    "tier": tier,
                    "precision": precision,
                    "tokens_seen": tokens,
                    "train_loss": round(train, 6),
                    "val_loss": round(val, 6),
                    "val_ppl": round(math.exp(val), 4),
                    "gap": round(val - train, 6),
                    "n_seeds": len(pairs),
                })
    return rows


def main():
    runs_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "local", "hf_runs")
    out_csv = sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, "results", "curves.csv")
    rows = summarize(runs_dir)
    if not rows:
        sys.exit(f"no runs with metrics.csv found under {runs_dir}")
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {out_csv} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
