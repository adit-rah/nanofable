"""Render the README training-curve figure from the committed results distillate.

Two panels, both read from results/curves.csv (seed-averaged, committed):
  left  - validation perplexity over the 500M-token budget. Every curve is still
          descending at the cutoff, and the ternary curves descend faster: the models
          are undertrained, ternary most of all.
  right - the generalization gap (val loss minus train loss). Flat and near zero for
          every tier and both precisions: nothing overfits, and ternary shows no
          regularization effect.

Emits a light and a dark variant for GitHub's <picture>.

    python scripts/plot_training_curves.py   # reads results/curves.csv, writes
                                             # docs/training-curves{,-dark}.png
"""

from __future__ import annotations

import csv
import os
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FixedLocator, NullFormatter  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
CURVES = os.path.join(ROOT, "results", "curves.csv")

PARAM_LABEL = {"tiny": "1M", "small": "6M", "medium": "16M", "large": "28M"}
TIER_ALPHA = {"tiny": 0.40, "small": 0.60, "medium": 0.80, "large": 1.0}
TIER_ORDER = ("tiny", "small", "medium", "large")

THEMES = {
    "light": {
        "out": os.path.join(ROOT, "docs", "training-curves.png"),
        "surface": "#ffffff",
        "ink": "#0b0b0b",
        "ink2": "#52514e",
        "muted": "#898781",
        "grid": "#e8e7e2",
        "fp16": "#2a78d6",
        "ternary": "#1baf7a",
    },
    "dark": {
        "out": os.path.join(ROOT, "docs", "training-curves-dark.png"),
        "surface": "#0d1117",
        "ink": "#f0f0ee",
        "ink2": "#c3c2b7",
        "muted": "#898781",
        "grid": "#24292f",
        "fp16": "#3987e5",
        "ternary": "#199e70",
    },
}


SMOOTH_WINDOW = 5  # val loss is a batched estimate; the raw gap is dominated by its jitter


def _rolling_mean(values: list[float], window: int) -> list[float]:
    half = window // 2
    out = []
    for i in range(len(values)):
        lo, hi = max(0, i - half), min(len(values), i + half + 1)
        chunk = values[lo:hi]
        out.append(sum(chunk) / len(chunk))
    return out


def load_curves():
    series = defaultdict(list)
    with open(CURVES) as f:
        for r in csv.DictReader(f):
            series[(r["precision"], r["tier"])].append({
                "mtok": int(r["tokens_seen"]) / 1e6,
                "ppl": float(r["val_ppl"]),
                "gap": float(r["gap"]),
            })
    for pts in series.values():
        pts.sort(key=lambda p: p["mtok"])
        for p, g in zip(pts, _rolling_mean([p["gap"] for p in pts], SMOOTH_WINDOW)):
            p["gap_smooth"] = g
    return series


def _style(ax, t):
    ax.yaxis.grid(True, color=t["grid"], linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(t["grid"])
    ax.tick_params(which="both", colors=t["muted"], labelsize=9.5, length=0, pad=6)
    ax.set_xlim(0, 500)
    ax.set_xticks([0, 100, 200, 300, 400, 500])
    ax.set_xticklabels(["0", "100M", "200M", "300M", "400M", "500M"])
    ax.set_xlabel("training tokens seen", fontsize=10, color=t["ink2"])


def render(series, t):
    plt.rcParams["font.family"] = ["Helvetica Neue", "Arial", "DejaVu Sans"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.4, 5.0), dpi=200)
    fig.patch.set_facecolor(t["surface"])

    for ax in (ax1, ax2):
        ax.set_facecolor(t["surface"])
        _style(ax, t)

    # ---- left: validation perplexity, still falling at the cutoff ----
    ax1.set_yscale("log")
    ax1.set_ylim(5, 120)
    ticks = [5, 10, 20, 50, 100]
    ax1.yaxis.set_major_locator(FixedLocator(ticks))
    ax1.yaxis.set_minor_formatter(NullFormatter())
    ax1.set_yticklabels([str(v) for v in ticks])
    ax1.set_ylabel("validation perplexity (log)", fontsize=10.5, color=t["ink2"])

    for precision in ("fp16", "ternary"):
        for tier in TIER_ORDER:
            pts = series[(precision, tier)]
            ax1.plot([p["mtok"] for p in pts], [p["ppl"] for p in pts],
                     color=t[precision], alpha=TIER_ALPHA[tier], linewidth=2,
                     solid_capstyle="round", zorder=3)
            end = pts[-1]
            if end["ppl"] <= 120:
                ax1.annotate(PARAM_LABEL[tier], (end["mtok"], end["ppl"]),
                             textcoords="offset points", xytext=(6, 0),
                             fontsize=8.5, color=t["muted"], va="center",
                             annotation_clip=False)

    ax1.text(0.03, 0.06, "every curve is still descending here",
             transform=ax1.transAxes, fontsize=9.5, color=t["ink2"], style="italic")

    # ---- right: generalization gap, flat and near zero ----
    ax2.set_ylim(-0.02, 0.12)
    ax2.axhline(0, color=t["muted"], linewidth=1, linestyle=(0, (5, 4)))
    ax2.set_ylabel(f"generalization gap, nats ({SMOOTH_WINDOW}-point rolling mean)",
                   fontsize=10.5, color=t["ink2"])

    for precision in ("fp16", "ternary"):
        for tier in TIER_ORDER:
            pts = series[(precision, tier)]
            xs = [p["mtok"] for p in pts]
            ax2.plot(xs, [p["gap"] for p in pts],
                     color=t[precision], alpha=TIER_ALPHA[tier] * 0.18, linewidth=1,
                     zorder=2)
            ax2.plot(xs, [p["gap_smooth"] for p in pts],
                     color=t[precision], alpha=TIER_ALPHA[tier], linewidth=2,
                     solid_capstyle="round", zorder=3)

    ax2.text(0.035, 0.90,
             "val loss minus train loss. All eight sit near 0.05 nats and never diverge:\n"
             "a val/train perplexity ratio of ≈1.05. Nothing here is overfitting.",
             transform=ax2.transAxes, fontsize=9.5, color=t["ink2"], style="italic",
             va="top", linespacing=1.5)

    # ---- shared legend: colour = precision, opacity = size ----
    handles = [
        plt.Line2D([], [], color=t["fp16"], linewidth=2, label="fp16"),
        plt.Line2D([], [], color=t["ternary"], linewidth=2, label="ternary"),
        plt.Line2D([], [], color=t["muted"], linewidth=2, alpha=0.4, label="1M params"),
        plt.Line2D([], [], color=t["muted"], linewidth=2, alpha=1.0, label="28M params"),
    ]
    leg = ax1.legend(handles=handles, loc="upper right", frameon=False, fontsize=9.5,
                     handlelength=1.6, borderaxespad=0.2, labelspacing=0.35)
    for text in leg.get_texts():
        text.set_color(t["ink2"])

    fig.text(0.055, 0.955, "Nobody finished training",
             fontsize=16, fontweight="bold", color=t["ink"], ha="left")
    fig.text(0.055, 0.906,
             "Validation perplexity and the train/val gap over an identical 500M-token budget · "
             "4 sizes × 2 precisions, seeds averaged",
             fontsize=10, color=t["ink2"], ha="left")

    fig.subplots_adjust(left=0.055, right=0.975, top=0.83, bottom=0.11, wspace=0.22)
    fig.savefig(t["out"], facecolor=t["surface"])
    plt.close(fig)
    print(f"Wrote {t['out']}")


def main():
    series = load_curves()
    for theme in THEMES.values():
        render(series, theme)


if __name__ == "__main__":
    main()
