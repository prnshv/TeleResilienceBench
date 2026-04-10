#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tqdm import tqdm

from src.collect_constants import planned_rows
from src.gsma_loader import SUBSETS, iter_all_subsets
from src.ollama_client import (
    build_base_mc_prompt,
    build_base_telemath_prompt,
    ollama_generate,
    ollama_version,
)
from src.parsing import (
    Boundary,
    HalfBy,
    compute_reasoning_half,
    extract_delimited_reasoning,
    extract_float_answer,
    floats_match,
    model_text_for_parsing,
    parse_mc_output,
    strip_trailing_final_answer_lines,
)


def load_done_ids(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    done: set[str] = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            sid = rec.get("sample_id")
            if sid:
                done.add(sid)
    return done


def load_progress_sample_ids(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    out: set[str] = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            sid = rec.get("sample_id")
            if sid:
                out.add(sid)
    return out


def append_progress(path: Path | None, obj: dict, *, file_lock: Optional[threading.Lock] = None) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    obj = {**obj, "ts": datetime.now(timezone.utc).isoformat()}
    line = json.dumps(obj, ensure_ascii=False) + "\n"
    lock_cm = file_lock if file_lock is not None else nullcontext()
    with lock_cm:
        with path.open("a", encoding="utf-8") as pf:
            pf.write(line)


def write_bench_line(path: Path, rec: dict, *, file_lock: Optional[threading.Lock] = None) -> None:
    lock_cm = file_lock if file_lock is not None else nullcontext()
    with lock_cm:
        with path.open("a", encoding="utf-8") as out:
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")


def merge_reasoning(gen, delimiter_fallback: bool) -> str:
    if not delimiter_fallback:
        th = (gen.thinking or "").strip()
        if th:
            return th
    resp = (gen.response or "").strip()
    r, _ = extract_delimited_reasoning(resp)
    if r:
        return r
    if delimiter_fallback:
        return ""
    return resp


def process_one_row(
    row,
    *,
    args: argparse.Namespace,
    half_by: HalfBy,
    boundary: Boundary,
    progress_path: Path | None,
    output_path: Path,
    file_lock: Optional[threading.Lock],
) -> None:
    """Single sample: Ollama + parse + append bench/progress (thread-safe when file_lock set)."""
    sid = row.sample_id
    if row.task_kind == "telemath":
        prompt = build_base_telemath_prompt(row.question, use_delimiter_fallback=args.delimiter_fallback)
    else:
        prompt = build_base_mc_prompt(
            row.question,
            row.choices,
            use_delimiter_fallback=args.delimiter_fallback,
        )
    think_flag = not args.delimiter_fallback
    try:
        gen = ollama_generate(
            args.base_model,
            prompt,
            host=args.ollama_host,
            think=think_flag,
            num_predict=args.num_predict,
            temperature=args.temperature,
            timeout=args.timeout,
            num_ctx=args.ollama_num_ctx,
            num_batch=args.ollama_num_batch,
            num_gpu=args.ollama_num_gpu,
            num_thread=args.ollama_num_thread,
        )
    except Exception as e:
        rec = {
            "sample_id": sid,
            "sub_benchmark": row.subset,
            "error": str(e),
            "status": "ollama_error",
        }
        write_bench_line(output_path, rec, file_lock=file_lock)
        append_progress(
            progress_path,
            {
                "sample_id": sid,
                "sub_benchmark": row.subset,
                "event": "ollama_error",
                "detail": str(e)[:500],
            },
            file_lock=file_lock,
        )
        return

    raw_for_parse = (
        model_text_for_parsing(gen.thinking, gen.response)
        if think_flag
        else (gen.response or "").strip()
    )
    reasoning_merged = merge_reasoning(gen, args.delimiter_fallback)
    if not reasoning_merged.strip() and not args.allow_empty_reasoning:
        append_progress(
            progress_path,
            {"sample_id": sid, "sub_benchmark": row.subset, "event": "no_reasoning"},
            file_lock=file_lock,
        )
        return

    reasoning_full = strip_trailing_final_answer_lines(reasoning_merged)

    if row.task_kind == "telemath":
        assert row.gold_float is not None
        pred_f = extract_float_answer(raw_for_parse)
        if pred_f is None:
            append_progress(
                progress_path,
                {"sample_id": sid, "sub_benchmark": row.subset, "event": "unparsed"},
                file_lock=file_lock,
            )
            return
        if floats_match(
            pred_f,
            row.gold_float,
            rel_tol=args.telemath_rel_tol,
            abs_tol=args.telemath_abs_tol,
        ):
            append_progress(
                progress_path,
                {"sample_id": sid, "sub_benchmark": row.subset, "event": "correct_base"},
                file_lock=file_lock,
            )
            return
        reasoning_half = compute_reasoning_half(
            reasoning_full,
            half_by=half_by,
            boundary=boundary,
        )
        record = {
            "sample_id": sid,
            "sub_benchmark": row.subset,
            "row_index": row.row_index,
            "task_kind": "telemath",
            "question": row.question,
            "choices": [],
            "gold_index": None,
            "gold_float": row.gold_float,
            "correct_answer": row.correct_answer_text,
            "incorrect_answer": repr(pred_f),
            "base_pred_float": pred_f,
            "base_raw": gen.response,
            "reasoning_full": reasoning_full,
            "reasoning_half": reasoning_half,
            "base_prompt_tokens": gen.prompt_tokens,
            "base_output_tokens": gen.output_tokens,
            "difficulty": row.difficulty,
        }
    else:
        assert row.gold_index is not None
        pred = parse_mc_output(raw_for_parse, row.choices)
        if pred is None:
            append_progress(
                progress_path,
                {"sample_id": sid, "sub_benchmark": row.subset, "event": "unparsed"},
                file_lock=file_lock,
            )
            return
        if pred == row.gold_index:
            append_progress(
                progress_path,
                {"sample_id": sid, "sub_benchmark": row.subset, "event": "correct_base"},
                file_lock=file_lock,
            )
            return
        reasoning_half = compute_reasoning_half(
            reasoning_full,
            half_by=half_by,
            boundary=boundary,
        )
        incorrect_text = row.choices[pred] if 0 <= pred < len(row.choices) else ""
        record = {
            "sample_id": sid,
            "sub_benchmark": row.subset,
            "row_index": row.row_index,
            "task_kind": "mc",
            "question": row.question,
            "choices": row.choices,
            "gold_index": row.gold_index,
            "gold_float": None,
            "correct_answer": row.correct_answer_text,
            "incorrect_answer": incorrect_text,
            "base_pred_index": pred,
            "base_raw": gen.response,
            "reasoning_full": reasoning_full,
            "reasoning_half": reasoning_half,
            "base_prompt_tokens": gen.prompt_tokens,
            "base_output_tokens": gen.output_tokens,
            "difficulty": row.difficulty,
        }
    write_bench_line(output_path, record, file_lock=file_lock)
    append_progress(
        progress_path,
        {"sample_id": sid, "sub_benchmark": row.subset, "event": "wrong_kept"},
        file_lock=file_lock,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect wrong+parseable base traces for TeleResilienceBench.")
    parser.add_argument("--base-model", type=str, default="qwen3.5:2b")
    parser.add_argument("--dataset", type=str, default="GSMA/ot-full")
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "tele_resilience_bench.jsonl")
    parser.add_argument("--metadata", type=Path, default=ROOT / "data" / "collect_metadata.json")
    parser.add_argument("--subset", action="append", default=None, help="Repeat for each subset; default all.")
    parser.add_argument("--max-samples", type=int, default=None, help="Per-subset cap.")
    parser.add_argument("--ollama-host", type=str, default="http://localhost:11434")
    parser.add_argument("--num-predict", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--half-by", choices=["char", "token"], default="token")
    parser.add_argument("--boundary", choices=["none", "sentence", "paragraph"], default="sentence")
    parser.add_argument(
        "--delimiter-fallback",
        action="store_true",
        help="Use <redacted_thinking> prompt instead of native think split (non-Qwen bases).",
    )
    parser.add_argument(
        "--allow-empty-reasoning",
        action="store_true",
        help="Keep rows even if no reasoning trace was captured.",
    )
    parser.add_argument(
        "--telemath-rel-tol",
        type=float,
        default=1e-3,
        help="Relative tolerance when judging TeleMath numeric equality.",
    )
    parser.add_argument(
        "--telemath-abs-tol",
        type=float,
        default=1e-5,
        help="Absolute tolerance when judging TeleMath numeric equality.",
    )
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument(
        "--progress-log",
        type=Path,
        default=ROOT / "data" / "collect_progress.jsonl",
        help="Append one JSON line per finished Ollama attempt (for monitoring / resume).",
    )
    parser.add_argument(
        "--no-progress-log",
        action="store_true",
        help="Disable writing the progress log.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Concurrent HTTP workers (set OLLAMA_NUM_PARALLEL on the server to match). Default 1.",
    )
    parser.add_argument(
        "--ollama-num-ctx",
        type=int,
        default=2048,
        help="Per-request context (tokens). Default 2048; increase only if thinking/output truncates.",
    )
    parser.add_argument(
        "--ollama-num-batch",
        type=int,
        default=None,
        help="Prompt batch size (llama.cpp). Try 512–1024 on a 4090 if stable.",
    )
    parser.add_argument(
        "--ollama-num-gpu",
        type=int,
        default=None,
        help=(
            "Layers to offload to GPU (API options.num_gpu). Use 99 for “all layers” (llama.cpp convention). "
            "Same effect as PARAMETER num_gpu 99 in a Modelfile; omit to use Ollama defaults."
        ),
    )
    parser.add_argument("--ollama-num-thread", type=int, default=None, help="Optional: CPU thread hint.")
    args = parser.parse_args()

    half_by: HalfBy = args.half_by  # type: ignore[assignment]
    boundary: Boundary = args.boundary  # type: ignore[assignment]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    done = load_done_ids(args.output)
    progress_path: Path | None = None if args.no_progress_log else args.progress_log
    progress_seen = load_progress_sample_ids(progress_path) if progress_path else set()

    subsets = args.subset if args.subset else list(SUBSETS)
    ver = ollama_version(args.ollama_host)

    meta = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "dataset": args.dataset,
        "subsets": subsets,
        "base_model": args.base_model,
        "ollama_version": ver,
        "half_by": args.half_by,
        "boundary": args.boundary,
        "delimiter_fallback": args.delimiter_fallback,
        "prompt_version": "tele_resilience_base_v3_final_answer",
        "telemath_rel_tol": args.telemath_rel_tol,
        "telemath_abs_tol": args.telemath_abs_tol,
        "progress_log": str(progress_path) if progress_path else None,
        "workers": args.workers,
        "ollama_num_ctx": args.ollama_num_ctx,
        "ollama_num_batch": args.ollama_num_batch,
        "ollama_num_gpu": args.ollama_num_gpu,
        "ollama_num_thread": args.ollama_num_thread,
    }
    args.metadata.parent.mkdir(parents=True, exist_ok=True)

    rows_iter = iter_all_subsets(
        subsets=subsets,
        dataset_name=args.dataset,
        max_samples_per_subset=args.max_samples,
    )
    rows = list(rows_iter)
    meta["planned_total_rows"] = len(rows)
    meta["planned_total_rows_card"] = planned_rows(list(subsets), args.max_samples, args.dataset)
    with args.metadata.open("w", encoding="utf-8") as mf:
        json.dump(meta, mf, indent=2)

    to_do = [r for r in rows if r.sample_id not in done and r.sample_id not in progress_seen]
    file_lock: Optional[threading.Lock] = threading.Lock() if args.workers > 1 else None

    if args.workers <= 1:
        for row in tqdm(to_do, desc="collect base"):
            process_one_row(
                row,
                args=args,
                half_by=half_by,
                boundary=boundary,
                progress_path=progress_path,
                output_path=args.output,
                file_lock=file_lock,
            )
    else:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = [
                ex.submit(
                    process_one_row,
                    row,
                    args=args,
                    half_by=half_by,
                    boundary=boundary,
                    progress_path=progress_path,
                    output_path=args.output,
                    file_lock=file_lock,
                )
                for row in to_do
            ]
            for fut in tqdm(as_completed(futs), total=len(futs), desc="collect base"):
                fut.result()

    print(f"Wrote (appended) benchmark rows to {args.output}")
    print(f"Metadata saved to {args.metadata}")


if __name__ == "__main__":
    main()
