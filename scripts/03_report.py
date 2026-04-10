#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt

from src.models_config import load_models_from_yaml


def load_summary(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def merge_params(
    summaries: Dict[str, Any], models_yaml: Optional[Path]
) -> Dict[str, Optional[float]]:
    out: Dict[str, Optional[float]] = {}
    for tag, block in summaries.items():
        pb = block.get("params_b")
        out[tag] = float(pb) if pb is not None else None
    if models_yaml and models_yaml.is_file():
        for entry in load_models_from_yaml(models_yaml):
            tag = entry["tag"]
            if tag in out and out[tag] is None and entry.get("params_b") is not None:
                out[tag] = float(entry["params_b"])
    return out


def build_table_rows(summaries: Dict[str, Any], params: Dict[str, Optional[float]]) -> List[Tuple[str, float, int, int, Optional[float], Optional[float]]]:
    rows: List[Tuple[str, float, int, int, Optional[float], Optional[float]]] = []
    for model, block in summaries.items():
        rows.append(
            (
                model,
                float(block.get("correct_flip_rate", 0.0)),
                int(block.get("correct_flips", 0)),
                int(block.get("n", 0)),
                block.get("mean_total_tokens"),
                params.get(model),
            )
        )
    rows.sort(key=lambda r: (-r[1], r[0]))
    return rows


def write_markdown(
    path: Path,
    bench_rows: int,
    rows: List[Tuple[str, float, int, int, Optional[float], Optional[float]]],
) -> None:
    lines = [
        "# TeleResilienceBench — continuation flip rates",
        "",
        f"Benchmark rows (wrong + parseable base): **{bench_rows}**",
        "",
        "| Model | Correct flip rate | Correct flips | N | Mean total tokens | Nominal params (B) |",
        "|-------|------------------:|--------------:|--:|------------------:|-------------------:|",
    ]
    for model, rate, cf, n, tok, pb in rows:
        tok_s = f"{tok:.1f}" if tok is not None else "—"
        pb_s = f"{pb:.1f}" if pb is not None else "—"
        lines.append(f"| {model} | {rate:.4f} | {cf} | {n} | {tok_s} | {pb_s} |")
    lines.append("")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_bars(
    labels: List[str],
    values: List[float],
    ylabel: str,
    title: str,
    out_path: Path,
) -> None:
    plt.figure(figsize=(max(8, len(labels) * 0.45), 4.5))
    x = range(len(labels))
    plt.bar(x, values, color="steelblue")
    plt.xticks(x, labels, rotation=45, ha="right")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Tables and charts from continuation summary.")
    parser.add_argument("--summary", type=Path, default=ROOT / "out" / "eval" / "summary.json")
    parser.add_argument("--models-config", type=Path, default=ROOT / "configs" / "models.yaml")
    parser.add_argument("--artifacts-dir", type=Path, default=ROOT / "out" / "report")
    args = parser.parse_args()

    data = load_summary(args.summary)
    summaries: Dict[str, Any] = data.get("models") or {}
    if not summaries:
        print("No models in summary.json", file=sys.stderr)
        sys.exit(1)

    params = merge_params(summaries, args.models_config)
    table_rows = build_table_rows(summaries, params)
    write_markdown(args.artifacts_dir / "flip_rates.md", int(data.get("num_bench_rows", 0)), table_rows)

    labels = [r[0] for r in table_rows]
    toks = [r[4] for r in table_rows]
    pbs = [r[5] for r in table_rows]

    tok_vals = [t if t is not None else 0.0 for t in toks]
    plot_bars(
        labels,
        tok_vals,
        "Mean prompt + output tokens per continuation call",
        "Average token usage (continuation)",
        args.artifacts_dir / "avg_tokens.png",
    )

    pb_vals = [p if p is not None else 0.0 for p in pbs]
    plot_bars(
        labels,
        pb_vals,
        "Billions of parameters (nominal VRAM proxy)",
        "Nominal model size",
        args.artifacts_dir / "nominal_params.png",
    )

    print(f"Wrote {args.artifacts_dir / 'flip_rates.md'}")
    print(f"Wrote {args.artifacts_dir / 'avg_tokens.png'}")
    print(f"Wrote {args.artifacts_dir / 'nominal_params.png'}")


if __name__ == "__main__":
    main()
