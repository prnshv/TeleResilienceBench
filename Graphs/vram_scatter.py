#!/usr/bin/env python3
"""Scatter: VRAM usage (percent of reference budget) vs. CFR, 8 models."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

from scatter_common import (
    DEFAULT_BENCH_JSONL,
    FAMILY_STYLE,
    FIG_GRID_MAJOR_KW,
    MANUAL_OFFSETS_VRAM,
    MODELS,
    REFERENCE_VRAM_GB,
    unweighted_mean_cfr_seven_subsets,
    vram_usage_percent,
)

matplotlib.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "axes.unicode_minus": False,
})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--experiments",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "Experiments",
    )
    ap.add_argument(
        "-o",
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent / "VRAM_scatter",
    )
    ap.add_argument(
        "--ref-vram-gb",
        type=float,
        default=REFERENCE_VRAM_GB,
        help=f"Budget for VRAM usage %% denominator (default: {REFERENCE_VRAM_GB})",
    )
    ap.add_argument(
        "--bench",
        type=Path,
        default=DEFAULT_BENCH_JSONL,
        help="tele_resilience_bench.jsonl (gold/base indices for subset CFR)",
    )
    args = ap.parse_args()

    pts: list[tuple[str, str, str, float, float]] = []
    for fam, param_lab, sub in MODELS:
        cfr_pct = unweighted_mean_cfr_seven_subsets(
            args.experiments, sub, args.bench
        )
        v_pct = vram_usage_percent(sub, ref_gb=args.ref_vram_gb)
        if cfr_pct is None or v_pct is None:
            continue
        pts.append((fam, param_lab, sub, v_pct, cfr_pct))

    xs = [p[3] for p in pts]
    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    # Room above axes for horizontal legend (avoids covering low-x points: Qwen 4B/9B, Nemotron)
    fig.subplots_adjust(left=0.16, right=0.96, bottom=0.17, top=0.86)

    seen: set[str] = set()
    for fam, param_lab, sub, x, y in pts:
        sty = FAMILY_STYLE[fam]
        ax.scatter(
            x, y,
            s=90,
            c=sty["color"],
            marker=sty["marker"],
            edgecolors="white",
            linewidths=0.6,
            zorder=6,
            label=sty["label"] if fam not in seen else None,
        )
        seen.add(fam)

        dx, dy, ha, va = MANUAL_OFFSETS_VRAM[sub]
        ax.annotate(
            param_lab,
            xy=(x, y),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=12,
            fontweight="bold",
            color=sty["color"],
            ha=ha,
            va=va,
            arrowprops={
                "arrowstyle": "-",
                "color": "#9e9e9e",
                "lw": 0.5,
                "shrinkA": 0,
                "shrinkB": 4,
            },
            zorder=7,
        )

    for spine in ax.spines.values():
        spine.set_linewidth(0.6)
        spine.set_color("#333333")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    x_lo, x_hi = min(xs), max(xs)
    x_pad = max((x_hi - x_lo) * 0.12, 3.0)
    ax.set_xlim(x_lo - x_pad, x_hi + x_pad)
    ys = [p[4] for p in pts]
    y_pad = max((max(ys) - min(ys)) * 0.12, 1.5)
    ax.set_ylim(min(ys) - y_pad, max(ys) + y_pad)

    ax.set_xlabel("VRAM usage (%)", fontsize=14, labelpad=6)
    ax.set_ylabel("CFR (%)", fontsize=14, labelpad=6)
    ax.tick_params(axis="both", which="major", labelsize=12, width=0.6,
                   length=4, direction="out", colors="#333333")
    ax.grid(which="major", axis="both", zorder=0, **FIG_GRID_MAJOR_KW)
    ax.set_axisbelow(True)

    leg = ax.legend(
        ncol=3,
        fontsize=10,
        frameon=True,
        edgecolor="#cccccc",
        fancybox=False,
        framealpha=1.0,
        borderpad=0.45,
        handletextpad=0.35,
        columnspacing=1.0,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.02),
        bbox_transform=ax.transAxes,
    )
    leg.set_zorder(4)
    leg.get_frame().set_linewidth(0.5)

    out = args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{out}.pdf", bbox_inches="tight", pad_inches=0.04)
    fig.savefig(f"{out}.png", dpi=300, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)


if __name__ == "__main__":
    main()
