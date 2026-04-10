#!/usr/bin/env python3
"""Bar chart: per-model CFR (%) for one MC subset (default: TeleTables) or aux TeleMath.

TeleTables matches ``main_table.tex`` (denominator = all rows in the subset, including
parse failures). For ``--artifact aux``, reads ``aux.jsonl`` and uses the same counting
rules as ``02_eval_continuation.summarize_continuation_jsonl`` (subset ``telemath``).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, MultipleLocator

from scatter_common import (
    DEFAULT_BENCH_JSONL,
    FAMILY_STYLE,
    FIG_GRID_MAJOR_KW,
    MODELS,
    load_bench,
    load_main_jsonl_lines,
    tally_subset,
)

matplotlib.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "axes.unicode_minus": False,
    # Fine lines: dense /// reads noisy; single-orientation hatches scale better in print.
    "hatch.linewidth": 0.42,
})

# One sparse hatch character per family: three orientations, easy to tell apart (incl. B&W).
HATCH_BY_FAMILY: dict[str, str] = {
    "qwen": "/",       # forward diagonal
    "gemma": "|",      # vertical
    "nemotron": "\\",  # back diagonal (single "\" in the hatch string)
}

# Lighter than the bar outline so stripes sit “inside” the fill, not harsh.
HATCH_LINE_COLOR = "#7a818a"


def _load_aux_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    out: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def aux_subset_cfr_pct(
    lines: list[dict[str, Any]],
    subset: str,
) -> tuple[float | None, int]:
    """Return (CFR %, n) for aux-style rows; CFR = 100 * correct_flips / n."""
    n = cf = 0
    for r in lines:
        if r.get("sub_benchmark") != subset:
            continue
        n += 1
        if r.get("correct_flip"):
            cf += 1
    if n <= 0:
        return None, 0
    return 100.0 * cf / n, n


def main_subset_cfr_pct(
    exp_root: Path,
    subdir: str,
    bench_path: Path,
    subset: str,
) -> float | None:
    lines = load_main_jsonl_lines(exp_root, subdir)
    if not lines or not bench_path.is_file():
        return None
    bench = load_bench(bench_path)
    n, cfr, _, _ = tally_subset(lines, bench, subset)
    if n <= 0:
        return None
    return 100.0 * cfr / n


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--experiments",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "Experiments",
    )
    ap.add_argument(
        "--bench",
        type=Path,
        default=DEFAULT_BENCH_JSONL,
        help="Gold/base indices for MC subsets (main artifact only).",
    )
    ap.add_argument(
        "--artifact",
        choices=("main", "aux"),
        default="main",
        help="main: main.jsonl + bench; aux: aux.jsonl (numeric TeleMath).",
    )
    ap.add_argument(
        "--subset",
        default=None,
        help="MC subset key (default: teletables) or aux key (default: telemath).",
    )
    ap.add_argument(
        "-o",
        "--out",
        type=Path,
        default=None,
        help="Output path without extension (writes .pdf and .png).",
    )
    args = ap.parse_args()

    if args.subset is not None:
        subset = args.subset
    else:
        subset = "telemath" if args.artifact == "aux" else "teletables"

    out = args.out
    if out is None:
        stem = f"{subset}_cfr_bars" if args.artifact == "main" else f"aux_{subset}_cfr_bars"
        out = Path(__file__).resolve().parent / stem

    labels: list[str] = []
    heights: list[float] = []
    colors: list[str] = []

    for fam, param_lab, sub in MODELS:
        labels.append(param_lab)
        if args.artifact == "main":
            pct_val = main_subset_cfr_pct(args.experiments, sub, args.bench, subset)
        else:
            jl = args.experiments / sub / "aux.jsonl"
            p, n = aux_subset_cfr_pct(_load_aux_jsonl(jl), subset)
            pct_val = p if n > 0 else None

        if pct_val is None:
            heights.append(float("nan"))
        else:
            heights.append(pct_val)
        colors.append(FAMILY_STYLE[fam]["color"])

    # Drop all-missing
    if all(h != h for h in heights):  # all NaN
        raise SystemExit("No data: check --experiments and per-model jsonl files.")

    fig_w = max(4.2, 0.42 * len(MODELS))
    fig, ax = plt.subplots(figsize=(fig_w, 3.15))
    # Match VRAM/token scatter: horizontal legend above axes; extra bottom for tilted x labels
    fig.subplots_adjust(left=0.14, right=0.97, bottom=0.18, top=0.82)

    x = list(range(len(labels)))
    bars = ax.bar(
        x,
        heights,
        color=colors,
        edgecolor="#3d4248",
        linewidth=0.55,
        width=0.72,
        zorder=3,
    )
    for bar, fam in zip(bars, (row[0] for row in MODELS)):
        bar.set_hatch(HATCH_BY_FAMILY.get(fam, ""))
        # Hatch color independent of edge (Matplotlib ≥3.8); else hatch follows edgecolor.
        setter = getattr(bar, "set_hatch_color", None)
        if callable(setter):
            setter(HATCH_LINE_COLOR)

    for spine in ax.spines.values():
        spine.set_linewidth(0.6)
        spine.set_color("#333333")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    y_upper = max(h for h in heights if h == h) * 1.12
    ax.set_ylim(0, max(y_upper, 8))
    ax.set_xticks(x)
    ax.set_xticklabels(
        labels,
        fontsize=11,
        ha="center",
        rotation=0,
    )
    ax.tick_params(
        axis="y",
        which="major",
        labelsize=12,
        width=0.6,
        length=4,
        direction="out",
        colors="#333333",
    )
    ax.tick_params(axis="x", which="major", width=0.6, length=3, direction="out", colors="#333333")

    # Vertical dashed separators between bar groups (publication-style column guides)
    if len(labels) > 1:
        ax.xaxis.set_minor_locator(FixedLocator([i + 0.5 for i in range(len(labels) - 1)]))
    ax.yaxis.set_minor_locator(MultipleLocator(5))

    ax.set_ylabel("CR%", fontsize=14, labelpad=6)

    ax.grid(which="major", axis="y", zorder=0, **FIG_GRID_MAJOR_KW)
    ax.grid(
        which="minor",
        axis="y",
        linestyle=(0, (1.2, 3)),
        linewidth=0.35,
        alpha=0.32,
        color="#b8bdc3",
        zorder=0,
    )
    ax.grid(
        which="minor",
        axis="x",
        linestyle=(0, (3.5, 3)),
        linewidth=0.35,
        alpha=0.34,
        color="#aeb3b9",
        zorder=0,
    )
    ax.set_axisbelow(True)

    # Same legend style as token_scatter / vram_scatter (family markers + top row)
    seen_f: set[str] = set()
    leg_handles: list[Line2D] = []
    leg_labels: list[str] = []
    for fam, _, _ in MODELS:
        if fam in seen_f:
            continue
        seen_f.add(fam)
        sty = FAMILY_STYLE[fam]
        leg_handles.append(
            Line2D(
                [0],
                [0],
                linestyle="None",
                marker=sty["marker"],
                markersize=9,
                markerfacecolor=sty["color"],
                markeredgecolor="white",
                markeredgewidth=0.6,
            )
        )
        leg_labels.append(sty["label"])

    leg = ax.legend(
        leg_handles,
        leg_labels,
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

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(f"{out}.pdf", bbox_inches="tight", pad_inches=0.03)
    fig.savefig(f"{out}.png", dpi=300, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


if __name__ == "__main__":
    main()
