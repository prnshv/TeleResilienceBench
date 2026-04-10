#!/usr/bin/env python3
"""
Build LaTeX table(s) of CFR / NF / WF from Experiments/*/main.jsonl and
data/tele_resilience_bench.jsonl (generator incorrect option = base_pred_index).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
DEFAULT_BENCH = ROOT / "data" / "tele_resilience_bench.jsonl"
DEFAULT_EXP = ROOT / "Experiments"
DEFAULT_OUT = DEFAULT_EXP / "main_table.tex"

TIER1_KEYS = ("teleqna", "teletables", "telelogs", "3gpp_tsg")
TIER2_KEYS = ("oranbench", "srsranbench", "sixg_bench")
# All seven MC sub-benchmarks (tier-1 + tier-2); macro column = unweighted mean over these.
ALL_MC_SUBSETS: tuple[str, ...] = TIER1_KEYS + TIER2_KEYS

# (LaTeX family cell, size cell, experiments subdirectory name)
MODEL_ROWS: list[tuple[str, str, str]] = [
    ("Qwen3.5", "4b", "qwen3.5_4b"),
    ("Qwen3.5", "9b", "qwen3.5_9b"),
    ("Qwen3.5", "27b", "qwen3.5_27b"),
    ("Gemma4", "e2b", "gemma4_e2b"),
    ("Gemma4", "e4b", "gemma4_e4b"),
    ("Gemma4", "26b", "gemma4_26b"),
    ("Gemma4", "31b", "gemma4_31b"),
    ("Nemotron-3", "4b", "nemotron-3-nano_4b"),
]


def load_bench(path: Path) -> dict[str, dict[str, Any]]:
    by_id: dict[str, dict[str, Any]] = {}
    with path.open() as f:
        for line in f:
            row = json.loads(line)
            by_id[row["sample_id"]] = row
    return by_id


def tally_subset(
    lines: list[dict[str, Any]],
    bench: dict[str, dict[str, Any]],
    subset: str | None,
) -> tuple[int, int, int, int]:
    """Returns (n_total, cfr, nf, wf) for subset (or all if subset is None).

    n_total counts every row in the subset (including parse failures). Percentages
    use this denominator so CFR matches reported correct_flip_rate (correct / n).
    """
    n_total = cfr = nf = wf = 0
    for r in lines:
        if subset is not None and r.get("sub_benchmark") != subset:
            continue
        n_total += 1
        if not r.get("parse_ok"):
            continue
        b = bench[r["sample_id"]]
        pred = r["continuation_pred_index"]
        gold = b["gold_index"]
        base = b["base_pred_index"]
        if pred == gold:
            cfr += 1
        elif pred == base:
            nf += 1
        else:
            wf += 1
    return n_total, cfr, nf, wf


def load_responses(jsonl_path: Path) -> list[dict[str, Any]]:
    if not jsonl_path.is_file():
        return []
    out: list[dict[str, Any]] = []
    with jsonl_path.open() as f:
        for line in f:
            out.append(json.loads(line))
    return out


def pct(count: int, denom: int) -> str:
    if denom <= 0:
        return "-"
    return f"{100.0 * count / denom:.1f}"


def macro_triplets(
    cells: list[tuple[str, str, str]],
) -> tuple[str, str, str]:
    """Unweighted mean of (CFR, NF, WF) percentage strings across all given subset cells."""
    vals: list[tuple[float, float, float]] = []
    for a, b, c in cells:
        if a == "-" or b == "-" or c == "-":
            continue
        vals.append((float(a), float(b), float(c)))
    if not vals:
        return "-", "-", "-"
    m0 = sum(t[0] for t in vals) / len(vals)
    m1 = sum(t[1] for t in vals) / len(vals)
    m2 = sum(t[2] for t in vals) / len(vals)
    return f"{m0:.1f}", f"{m1:.1f}", f"{m2:.1f}"


def model_row_cells(
    lines: list[dict[str, Any]],
    bench: dict[str, dict[str, Any]],
    keys: tuple[str, ...],
) -> list[tuple[str, str, str]]:
    cells: list[tuple[str, str, str]] = []
    for key in keys:
        n, cfr, nf, wf = tally_subset(lines, bench, key)
        cells.append((pct(cfr, n), pct(nf, n), pct(wf, n)))
    return cells


def build_tex(bench: dict[str, dict[str, Any]], exp_dir: Path) -> str:
    # Group model rows by family for \multirow
    families: list[tuple[str, int, list[tuple[str, str]]]] = []
    i = 0
    while i < len(MODEL_ROWS):
        fam = MODEL_ROWS[i][0]
        chunk: list[tuple[str, str]] = []
        while i < len(MODEL_ROWS) and MODEL_ROWS[i][0] == fam:
            chunk.append((MODEL_ROWS[i][1], MODEL_ROWS[i][2]))
            i += 1
        families.append((fam, len(chunk), chunk))

    tier1_header_cols = "ccc ccc ccc ccc"
    tier2_header_cols = "ccc ccc ccc >{\\columncolor{gray!15}}c >{\\columncolor{gray!15}}c >{\\columncolor{gray!15}}c"

    body_t1: list[str] = []
    body_t2: list[str] = []

    for fam_name, n_rows, sizes_dirs in families:
        for j, (size, subdir) in enumerate(sizes_dirs):
            jsonl = exp_dir / subdir / "main.jsonl"
            lines = load_responses(jsonl)
            if not lines:
                dash = " & ".join(["- & - & -"] * 4)
                tier1_line = (
                    f"        {fam_name} & {size} & {dash} \\\\"
                    if n_rows == 1
                    else (
                        f"        \\multirow{{{n_rows}}}{{*}}{{{fam_name}}} & {size} & {dash} \\\\"
                        if j == 0
                        else f"                                 & {size} & {dash} \\\\"
                    )
                )
                dash2 = " & ".join(["- & - & -"] * 3)
                tier2_line = (
                    f"        {fam_name} & {size} & {dash2} & - & - & - \\\\"
                    if n_rows == 1
                    else (
                        f"        \\multirow{{{n_rows}}}{{*}}{{{fam_name}}} & {size} & {dash2} & - & - & - \\\\"
                        if j == 0
                        else f"                                 & {size} & {dash2} & - & - & - \\\\"
                    )
                )
                body_t1.append(tier1_line)
                body_t2.append(tier2_line)
                continue

            c1 = model_row_cells(lines, bench, TIER1_KEYS)
            c2 = model_row_cells(lines, bench, TIER2_KEYS)
            c_all = model_row_cells(lines, bench, ALL_MC_SUBSETS)
            m = macro_triplets(c_all)
            t1s = " & ".join(f"{a} & {b} & {c}" for a, b, c in c1)
            t2s = " & ".join(f"{a} & {b} & {c}" for a, b, c in c2)
            t2m = f"{m[0]} & {m[1]} & {m[2]}"

            if n_rows == 1:
                body_t1.append(f"        {fam_name} & {size} & {t1s} \\\\")
                body_t2.append(f"        {fam_name} & {size} & {t2s} & {t2m} \\\\")
            elif j == 0:
                body_t1.append(f"        \\multirow{{{n_rows}}}{{*}}{{{fam_name}}} & {size} & {t1s} \\\\")
                body_t2.append(f"        \\multirow{{{n_rows}}}{{*}}{{{fam_name}}} & {size} & {t2s} & {t2m} \\\\")
            else:
                body_t1.append(f"                                 & {size} & {t1s} \\\\")
                body_t2.append(f"                                 & {size} & {t2s} & {t2m} \\\\")

        body_t1.append("        \\midrule")
        body_t2.append("        \\midrule")

    # drop trailing midrule from last group
    body_t1.pop()
    body_t2.pop()

    t1 = "\n".join(
        [
            r"\begin{table*}[h]",
            r"    \centering",
            r"    \caption{Reasoning Resilience across 7 discrete-choice GSMA sub-benchmarks. We report the Correct Flip Rate (CFR), No Flip (NF), and Wrong Flip (WF) percentages for each model. Macro Average is the unweighted mean of the seven subset percentages (each subset weighted equally, not by sample count).}",
            r"    \label{tab:flip_rates}",
            r"",
            r"    \resizebox{\linewidth}{!}{%",
            rf"    \begin{{tabular}}{{ll {tier1_header_cols}}}",
            r"        \toprule",
            r"        \multirow{2}{*}{\textbf{Family}} & \multirow{2}{*}{\textbf{Size}} & \multicolumn{3}{c}{\textbf{TeleQnA}} & \multicolumn{3}{c}{\textbf{TeleTables}} & \multicolumn{3}{c}{\textbf{TeleLogs}} & \multicolumn{3}{c}{\textbf{3GPP\_TSG}} \\",
            r"        \cmidrule(lr){3-5} \cmidrule(lr){6-8} \cmidrule(lr){9-11} \cmidrule(lr){12-14}",
            r"        & & CFR & NF & WF & CFR & NF & WF & CFR & NF & WF & CFR & NF & WF \\",
            r"        \midrule",
            *body_t1,
            r"        \bottomrule",
            r"    \end{tabular}%",
            r"    }",
            r"",
            r"    \resizebox{\linewidth}{!}{%",
            rf"    \begin{{tabular}}{{ll {tier2_header_cols}}}",
            r"        \toprule",
            r"        \multirow{2}{*}{\textbf{Family}} & \multirow{2}{*}{\textbf{Size}} & \multicolumn{3}{c}{\textbf{ORANBench}} & \multicolumn{3}{c}{\textbf{srsRANBench}} & \multicolumn{3}{c}{\textbf{SixG\_Bench}} & \multicolumn{3}{>{\columncolor{gray!15}}c}{\textbf{Macro Average}} \\",
            r"        \cmidrule(lr){3-5} \cmidrule(lr){6-8} \cmidrule(lr){9-11} \cmidrule(lr){12-14}",
            r"        & & CFR & NF & WF & CFR & NF & WF & CFR & NF & WF & \cellcolor{gray!15}CFR & \cellcolor{gray!15}NF & \cellcolor{gray!15}WF \\",
            r"        \midrule",
            *body_t2,
            r"        \bottomrule",
            r"    \end{tabular}%",
            r"    }",
            r"\end{table*}",
        ]
    )
    return t1


def main() -> None:
    p = argparse.ArgumentParser(description="Generate CFR/NF/WF LaTeX table.")
    p.add_argument(
        "--bench",
        type=Path,
        default=DEFAULT_BENCH,
        help="tele_resilience_bench.jsonl path",
    )
    p.add_argument(
        "--experiments",
        type=Path,
        default=DEFAULT_EXP,
        help="Experiments directory containing <model>/main.jsonl",
    )
    p.add_argument(
        "-o",
        "--output",
        type=Path,
        default=DEFAULT_OUT,
        help="Output .tex path",
    )
    args = p.parse_args()

    bench = load_bench(args.bench)
    tex = build_tex(bench, args.experiments)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(tex, encoding="utf-8")


if __name__ == "__main__":
    main()
