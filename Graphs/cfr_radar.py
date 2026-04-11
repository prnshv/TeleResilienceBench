#!/usr/bin/env python3
"""Radar chart: mean CFR (%) per MC sub-benchmark, averaged across all continuation models.

CFR uses the same rules as ``main_table.py`` / ``scatter_common.tally_subset`` (denominator
includes parse failures). Lower mean CFR on a spoke = harder sub-benchmark on average.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from scatter_common import (
    DEFAULT_BENCH_JSONL,
    MODELS,
    SEVEN_MC_SUBSETS,
    load_bench,
    load_main_jsonl_lines,
    tally_subset,
)

matplotlib.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "axes.unicode_minus": False,
})

R_MAX: float = 40.0

# Display order matches SEVEN_MC_SUBSETS (short labels for radial text).
# Two-line names shorten horizontal extent so the saved figure needs less width.
SUBSET_LABELS: dict[str, str] = {
    "teleqna": "TeleQnA",
    "teletables": "TeleTables",
    "telelogs": "Tele\nLogs",
    "3gpp_tsg": "3GPP TSG",
    "oranbench": "ORANBench",
    "srsranbench": "srsRAN\nBench",
    "sixg_bench": "SixG\nBench",
}


def subset_cfr_pct_for_model(
    exp_root: Path,
    subdir: str,
    bench: dict,
    subset: str,
) -> float | None:
    lines = load_main_jsonl_lines(exp_root, subdir)
    if lines is None:
        return None
    n, cfr, _, _ = tally_subset(lines, bench, subset)
    if n <= 0:
        return None
    return 100.0 * cfr / n


def mean_cfr_by_subset(
    exp_root: Path,
    bench: dict,
) -> tuple[list[str], list[float]]:
    """Return (subset_keys, mean_cfr_pct_each) averaged over models with data per subset."""
    keys = list(SEVEN_MC_SUBSETS)
    means: list[float] = []
    for subset in keys:
        per_model: list[float] = []
        for _, _, subdir in MODELS:
            p = subset_cfr_pct_for_model(exp_root, subdir, bench, subset)
            if p is not None:
                per_model.append(p)
        if not per_model:
            means.append(float("nan"))
        else:
            means.append(sum(per_model) / len(per_model))
    return keys, means


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
        default=Path(__file__).resolve().parent / "cfr_subset_radar",
    )
    ap.add_argument(
        "--bench",
        type=Path,
        default=DEFAULT_BENCH_JSONL,
        help="tele_resilience_bench.jsonl (gold/base indices)",
    )
    ap.add_argument(
        "--r-max",
        type=float,
        default=R_MAX,
        help="Radial axis limit (CFR %%); default 40",
    )
    args = ap.parse_args()

    if not args.bench.is_file():
        raise SystemExit(f"Bench file not found: {args.bench}")

    bench = load_bench(args.bench)
    keys, means = mean_cfr_by_subset(args.experiments, bench)

    if all(np.isnan(means)):
        raise SystemExit("No model data found under Experiments/*/main.jsonl")

    labels = [SUBSET_LABELS[k] for k in keys]
    n = len(means)
    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    values = np.array(means, dtype=float)
    angles_c = np.concatenate([angles, angles[:1]])
    values_c = np.concatenate([values, values[:1]])

    worst_i = int(np.nanargmin(values_c[:-1]))
    accent = "#c44e52"

    # Slightly narrower width: multiline spoke labels reduce horizontal label overhang
    fig, ax = plt.subplots(figsize=(3.45, 3.85), subplot_kw=dict(projection="polar"))
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)

    fill_c = "#5F8FB8"
    edge_c = "#2d4a63"
    ax.plot(angles_c, values_c, color=edge_c, linewidth=1.1, zorder=4)
    ax.fill(angles_c, values_c, color=fill_c, alpha=0.28, zorder=2)

    ax.scatter(
        [angles[worst_i]],
        [means[worst_i]],
        s=70,
        zorder=6,
        c=accent,
        edgecolors="white",
        linewidths=0.7,
    )

    r_max = float(args.r_max)
    if r_max <= 0:
        r_max = R_MAX
    ax.set_ylim(0.0, r_max)
    rt = np.arange(0.0, r_max + 0.01, 10.0)
    ax.set_yticks(rt)
    ax.set_yticklabels([f"{t:.0f}" for t in rt], fontsize=7.5, color="#666666")
    ax.set_rlabel_position(22.5)
    ax.grid(True, linestyle=(0, (3, 3)), linewidth=0.45, alpha=0.5, color="#889099")

    # Axis (spoke) labels — TeleQnA placed manually a bit inward to save vertical space
    ax.set_xticks(angles)
    spoke_labels = list(labels)
    teleqna_i = keys.index("teleqna")
    spoke_labels[teleqna_i] = ""
    ax.set_xticklabels(spoke_labels, fontsize=9.5)
    ax.tick_params(axis="x", which="major", pad=11)

    # TeleQnA: manual radius along top spoke (nudge vs default ticks)
    teleqna_r = r_max * 1.034
    ax.text(
        angles[teleqna_i],
        teleqna_r,
        SUBSET_LABELS["teleqna"],
        fontsize=9.5,
        ha="center",
        va="center",
        color=accent if teleqna_i == worst_i else "#000000",
        fontweight="bold" if teleqna_i == worst_i else "normal",
    )

    # Colour the worst-category axis label red
    for tick, key in zip(ax.get_xticklabels(), keys):
        if key == keys[worst_i]:
            tick.set_color(accent)
            tick.set_fontweight("bold")

    # CFR labels radially outside the filled polygon; a few spokes nudged toward outer name labels
    gap = max(2.5, r_max * 0.08)
    toward_legend = frozenset({"teletables", "telelogs", "3gpp_tsg"})
    for i in range(n):
        v = float(means[i])
        is_worst = i == worst_i
        cap = r_max - 0.28 if keys[i] in toward_legend else r_max - 0.55
        extra = 1.15 if keys[i] in toward_legend else 0.0
        r_ann = min(v + gap + extra, cap)
        pct = f"{v:.1f}%"
        kw = dict(
            fontsize=8.5,
            color=accent if is_worst else "#1a1a1a",
            fontweight="bold" if is_worst else "normal",
            zorder=8,
        )
        if keys[i] == "srsranbench":
            ax.annotate(
                pct,
                xy=(angles[i], r_ann),
                xytext=(0, 7),
                textcoords="offset points",
                ha="center",
                va="bottom",
                **kw,
            )
        else:
            ax.text(
                angles[i],
                r_ann,
                pct,
                ha="center",
                va="center",
                **kw,
            )

    fig.subplots_adjust(left=0.05, right=0.95, bottom=0.05, top=0.95)

    out = args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{out}.pdf", bbox_inches="tight", pad_inches=0.06)
    fig.savefig(f"{out}.png", dpi=300, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)


if __name__ == "__main__":
    main()
