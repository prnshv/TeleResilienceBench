"""Shared model list, VRAM table, and eval summary loading for scatter plots."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

# Same seven MC subsets as main_table.py (tier1 + tier2); CFR = unweighted mean of their rates.
SEVEN_MC_SUBSETS: tuple[str, ...] = (
    "teleqna",
    "teletables",
    "telelogs",
    "3gpp_tsg",
    "oranbench",
    "srsranbench",
    "sixg_bench",
)

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_BENCH_JSONL = REPO_ROOT / "data" / "tele_resilience_bench.jsonl"

# (family key, param label, Experiments subdir)
MODELS: list[tuple[str, str, str]] = [
    ("qwen", "4B", "qwen3.5_4b"),
    ("qwen", "9B", "qwen3.5_9b"),
    ("qwen", "27B", "qwen3.5_27b"),
    ("gemma", "e2b", "gemma4_e2b"),
    ("gemma", "e4b", "gemma4_e4b"),
    ("gemma", "26B", "gemma4_26b"),
    ("gemma", "31B", "gemma4_31b"),
    ("nemotron", "4B", "nemotron-3-nano_4b"),
]

FAMILY_STYLE: dict[str, dict] = {
    "qwen": {"color": "#1565C0", "marker": "o", "label": "Qwen 3.5"},
    "gemma": {"color": "#2E7D32", "marker": "s", "label": "Gemma 4"},
    "nemotron": {"color": "#E65100", "marker": "D", "label": "Nemotron-3"},
}

# Peak VRAM (GB) per run for continuation eval (user-provided).
# Qwen 3.5: qwen3.5:4b → 3.4 GB, qwen3.5:9b → 6.6 GB, qwen3.5:27b → 17 GB
VRAM_GB_BY_SUBDIR: dict[str, float] = {
    "qwen3.5_4b": 3.4,   # Qwen 3.5 4B
    "qwen3.5_9b": 6.6,   # Qwen 3.5 9B
    "qwen3.5_27b": 17.0,  # Qwen 3.5 27B
    "gemma4_e2b": 7.2,
    "gemma4_e4b": 9.6,
    "gemma4_26b": 18.0,
    "gemma4_31b": 20.0,
    "nemotron-3-nano_4b": 2.8,
}

# Denominator for "VRAM usage (%)": fraction of this budget each model occupies.
REFERENCE_VRAM_GB: float = 24.0

# Offsets for token scatter (x = mean output tokens)
MANUAL_OFFSETS_TOKEN: dict[str, tuple[float, float, str, str]] = {
    "qwen3.5_27b": (-72, 18, "right", "center"),
    "qwen3.5_9b": (-72, 6, "right", "center"),
    "qwen3.5_4b": (-72, -4, "right", "center"),
    "gemma4_e2b": (14, -8, "left", "center"),
    "gemma4_e4b": (14, 0, "left", "center"),
    "gemma4_26b": (14, -8, "left", "center"),
    "gemma4_31b": (14, 0, "left", "center"),
    "nemotron-3-nano_4b": (14, 0, "left", "center"),
}

# Offsets for VRAM scatter (x = usage % of REFERENCE_VRAM_GB); Qwen spread on x so simpler offsets
MANUAL_OFFSETS_VRAM: dict[str, tuple[float, float, str, str]] = {
    "qwen3.5_4b": (10, 8, "left", "bottom"),
    "qwen3.5_9b": (10, 0, "left", "center"),
    "qwen3.5_27b": (-10, 10, "right", "bottom"),
    "gemma4_e2b": (12, 8, "left", "bottom"),
    "gemma4_e4b": (12, 0, "left", "center"),
    "gemma4_26b": (-12, -10, "right", "top"),
    "gemma4_31b": (-12, 6, "right", "bottom"),
    "nemotron-3-nano_4b": (12, 6, "left", "bottom"),
}


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
    """Match main_table.py: (n_total, cfr, nf, wf); n includes parse failures."""
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


def load_main_jsonl_lines(exp_root: Path, subdir: str) -> list[dict[str, Any]] | None:
    jl = exp_root / subdir / "main.jsonl"
    if not jl.is_file():
        return None
    out: list[dict[str, Any]] = []
    with jl.open() as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out if out else None


def unweighted_mean_cfr_seven_subsets(
    exp_root: Path,
    subdir: str,
    bench_path: Path,
) -> float | None:
    """CFR in 0–100: mean of subset CFRs (each subset: 100 * correct / n, n includes parse fails)."""
    lines = load_main_jsonl_lines(exp_root, subdir)
    if lines is None:
        return None
    if not bench_path.is_file():
        return None
    bench = load_bench(bench_path)
    pcts: list[float] = []
    for key in SEVEN_MC_SUBSETS:
        n, cfr, _, _ = tally_subset(lines, bench, key)
        if n <= 0:
            return None
        pcts.append(100.0 * cfr / n)
    return sum(pcts) / len(pcts)


def _mean_tokens_from_jsonl(jsonl: Path) -> float | None:
    sum_ot = tok_n = 0
    with jsonl.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            pt, ot = r.get("prompt_tokens"), r.get("output_tokens")
            if pt is not None and ot is not None:
                sum_ot += int(ot)
                tok_n += 1
    return sum_ot / tok_n if tok_n > 0 else None


def load_mean_output_tokens(exp_root: Path, subdir: str) -> float | None:
    """Mean continuation output tokens (same rule as eval summary)."""
    jpath = exp_root / subdir / "main.json"
    if jpath.is_file():
        try:
            data = json.loads(jpath.read_text())
        except json.JSONDecodeError:
            pass
        else:
            s = data.get("summary") or {}
            out = s.get("mean_output_tokens")
            if out is not None:
                return float(out)
    jl = exp_root / subdir / "main.jsonl"
    if jl.is_file():
        return _mean_tokens_from_jsonl(jl)
    return None


def load_mean_tokens_and_cfr(exp_root: Path, subdir: str) -> tuple[float, float] | None:
    """Return (mean_output_tokens, global correct_flip_rate in 0..1) from summary/jsonl — legacy."""
    jpath = exp_root / subdir / "main.json"
    if jpath.is_file():
        try:
            data = json.loads(jpath.read_text())
        except json.JSONDecodeError:
            pass
        else:
            s = data.get("summary") or {}
            out, cfr = s.get("mean_output_tokens"), s.get("correct_flip_rate")
            if out is not None and cfr is not None:
                return float(out), float(cfr)
    jl = exp_root / subdir / "main.jsonl"
    if jl.is_file():
        n = cf = sum_ot = tok_n = 0
        with jl.open() as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                n += 1
                if r.get("correct_flip"):
                    cf += 1
                pt, ot = r.get("prompt_tokens"), r.get("output_tokens")
                if pt is not None and ot is not None:
                    sum_ot += int(ot)
                    tok_n += 1
        if n <= 0 or tok_n <= 0:
            return None
        return sum_ot / tok_n, cf / n
    return None


def vram_usage_percent(subdir: str, *, ref_gb: float = REFERENCE_VRAM_GB) -> float | None:
    gb = VRAM_GB_BY_SUBDIR.get(subdir)
    if gb is None or ref_gb <= 0:
        return None
    return 100.0 * gb / ref_gb
