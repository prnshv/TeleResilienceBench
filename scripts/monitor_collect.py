#!/usr/bin/env python3
"""
Summarize TeleResilienceBench base collection: per-sub-benchmark breakdown,
wrong rows in the bench JSONL, progress-log events, and ETA from recent throughput.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, DefaultDict, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.collect_constants import subset_row_counts_for_dataset


def parse_ts_iso(s: str) -> Optional[float]:
    try:
        s = s.replace("Z", "+00:00")
        return datetime.fromisoformat(s).timestamp()
    except Exception:
        return None


def load_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
    if not path.is_file():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def bench_stats(path: Path) -> Tuple[DefaultDict[str, int], int]:
    """Per-subset wrong-kept rows (valid benchmark lines) + total error lines."""
    by_sub: DefaultDict[str, int] = defaultdict(int)
    errors = 0
    for rec in load_jsonl(path):
        if rec.get("error") or rec.get("status") == "ollama_error":
            errors += 1
            continue
        if rec.get("gold_index") is None and rec.get("task_kind") != "telemath":
            continue
        if rec.get("task_kind") == "telemath" and rec.get("gold_float") is None:
            continue
        sub = str(rec.get("sub_benchmark") or "?")
        by_sub[sub] += 1
    return by_sub, errors


def progress_stats(rows: List[dict]) -> Tuple[Dict[str, Dict[str, int]], int, List[float]]:
    """Nested sub -> event -> count; total finished attempts; timestamps (parsed)."""
    by_sub_event: DefaultDict[str, DefaultDict[str, int]] = defaultdict(lambda: defaultdict(int))
    ts_list: List[float] = []
    for rec in rows:
        sub = str(rec.get("sub_benchmark") or "?")
        ev = str(rec.get("event") or "?")
        by_sub_event[sub][ev] += 1
        t = rec.get("ts")
        if isinstance(t, str):
            p = parse_ts_iso(t)
            if p is not None:
                ts_list.append(p)
    total = len(rows)
    return {k: dict(v) for k, v in by_sub_event.items()}, total, sorted(ts_list)


def rate_and_eta(
    timestamps: List[float],
    completed: int,
    planned_total: int,
    window: int = 200,
    *,
    now: float,
) -> Tuple[Optional[float], Optional[float], str]:
    """
    Returns (rate_per_sec, eta_seconds, mode).
    mode: 'done' | 'none' | 'first_event' (rough) | 'window'
    """
    if planned_total <= 0:
        return None, None, "none"
    if completed >= planned_total:
        return None, None, "done"
    remaining = planned_total - completed
    if not timestamps:
        return None, None, "none"
    if len(timestamps) == 1:
        dt = max(now - timestamps[0], 0.5)
        rate = 1.0 / dt
        return rate, remaining / rate, "first_event"
    tail = timestamps[-window:]
    if len(tail) < 2:
        tail = timestamps
    dt = tail[-1] - tail[0]
    if dt <= 0:
        return None, None, "none"
    rate = (len(tail) - 1) / dt
    if rate <= 0:
        return None, None, "none"
    return rate, remaining / rate, "window"


def fmt_duration(sec: Optional[float]) -> str:
    if sec is None or sec < 0 or sec != sec:
        return "n/a"
    if sec < 60:
        return f"{sec:.0f}s"
    if sec < 3600:
        return f"{sec / 60:.1f}m"
    if sec < 86400:
        return f"{sec / 3600:.1f}h"
    return f"{sec / 86400:.1f}d"


def print_report(
    metadata_path: Path,
    progress_path: Path,
    bench_path: Path,
    rate_window: int,
    *,
    dataset_override: Optional[str] = None,
) -> None:
    meta: Dict[str, Any] = {}
    if metadata_path.is_file():
        with metadata_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)

    ds_name = str(dataset_override or meta.get("dataset") or "GSMA/ot-full")
    plan_table = subset_row_counts_for_dataset(ds_name)

    planned = int(meta.get("planned_total_rows") or 0)
    if planned <= 0:
        subsets = meta.get("subsets") or list(plan_table.keys())
        planned = sum(plan_table.get(s, 0) for s in subsets)

    prog_rows = load_jsonl(progress_path)
    by_sub_ev, finished, ts = progress_stats(prog_rows)
    now_ts = time.time()
    rate, eta_sec, eta_mode = rate_and_eta(ts, finished, planned, window=rate_window, now=now_ts)

    bench_by_sub, bench_errs = bench_stats(bench_path)
    wrong_total = sum(bench_by_sub.values())

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    print(f"TeleResilienceBench collector — {now}")
    print(f"Dataset (metadata): {ds_name}")
    print(f"Planned Ollama attempts (this run): {planned}")
    print(f"Finished attempts (progress log lines): {finished} ({100.0 * finished / planned:.2f}%)" if planned else f"Finished attempts: {finished}")
    if eta_mode == "done":
        print("Run complete (all planned attempts logged in progress file).")
    elif rate is not None and eta_mode == "first_event":
        print(
            f"Rough rate (single sample so far): {60.0 * rate:.2f} attempts/min  →  ETA ~{fmt_duration(eta_sec)}  (refines after more events)"
        )
    elif rate is not None and eta_mode == "window":
        print(f"Recent rate (~last {rate_window} events): {60.0 * rate:.2f} attempts/min  →  ETA ~{fmt_duration(eta_sec)}")
    else:
        print("ETA: no timestamps in progress log yet (or planned total unknown).")
    print()

    subs = sorted(set(plan_table.keys()) | set(by_sub_ev.keys()) | set(bench_by_sub.keys()))
    print(f"{'subset':<14} {'plan':>6} {'done':>6} {'bench':>6} {'corr':>5} {'unpar':>5} {'norea':>5} {'err':>4}")
    print("-" * 62)
    for s in subs:
        plan = plan_table.get(s, 0)
        ev = by_sub_ev.get(s, {})
        done_sub = sum(ev.values())
        w = ev.get("wrong_kept", 0)
        c = ev.get("correct_base", 0)
        u = ev.get("unparsed", 0)
        nr = ev.get("no_reasoning", 0)
        er = ev.get("ollama_error", 0)
        bench_n = bench_by_sub.get(s, 0)
        print(f"{s:<14} {plan:>6} {done_sub:>6} {bench_n:>6} {c:>5} {u:>5} {nr:>5} {er:>4}")
    plan_table_sum = sum(plan_table.get(x, 0) for x in subs)
    print("-" * 62)
    print(f"{'TOTAL':<14} {plan_table_sum:>6} {finished:>6} {wrong_total:>6}")
    print()
    print(f"Bench JSONL: {bench_path}  — wrong-kept rows (column `bench`): {wrong_total}")
    if bench_errs:
        print(f"Bench JSONL error lines (ollama_error): {bench_errs}")
    print(f"Progress log: {progress_path}")
    print()
    print(
        "Column `bench` = wrong+parseable rows written to the benchmark JSONL. "
        "`unpar` = model replied but MC/number could not be parsed (not added to bench). "
        "Legend: corr=base matched gold | norea=empty reasoning trace | err=Ollama error"
    )


def main() -> None:
    p = argparse.ArgumentParser(
        description="Monitor TeleResilienceBench base collection progress.",
        epilog="Examples:  %(prog)s          # one snapshot\n           %(prog)s -w       # refresh every 30s\n           %(prog)s -w 10    # refresh every 10s",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--metadata", type=Path, default=ROOT / "data" / "collect_metadata.json")
    p.add_argument("--progress", type=Path, default=ROOT / "data" / "collect_progress.jsonl")
    p.add_argument("--bench", type=Path, default=ROOT / "data" / "tele_resilience_bench.jsonl")
    p.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="HF dataset id for per-subset plan column (e.g. GSMA/ot-lite). Default: read from metadata, else ot-full.",
    )
    p.add_argument("--rate-window", type=int, default=200, help="How many recent events to use for ETA rate.")
    p.add_argument(
        "-w",
        "--watch",
        type=float,
        nargs="?",
        const=30.0,
        default=0.0,
        metavar="SEC",
        help="Live mode: clear screen and refresh every SEC seconds (default 30 if -w/--watch is used alone). Ctrl+C to stop.",
    )
    args = p.parse_args()
    interval = float(args.watch)
    if interval < 0:
        p.error("--watch interval must be >= 0")

    while True:
        if interval > 0:
            print("\033[2J\033[H", end="")
        print_report(
            args.metadata,
            args.progress,
            args.bench,
            args.rate_window,
            dataset_override=args.dataset,
        )
        if interval > 0:
            print(f"\n--- Refresh every {interval:g}s (Ctrl+C to stop) ---")
        if interval <= 0:
            break
        time.sleep(interval)


if __name__ == "__main__":
    main()
