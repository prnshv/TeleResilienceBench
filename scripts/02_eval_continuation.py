#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, DefaultDict, Dict, List, Optional, Set

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tqdm import tqdm

from src.models_config import load_models_from_yaml
from src.ollama_client import (
    build_continuation_mc_prompt,
    build_continuation_telemath_prompt,
    ollama_generate,
)
from src.parsing import extract_float_answer, floats_match, model_text_for_parsing, parse_mc_output


def sanitize_model_tag(tag: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", tag)


def bench_row_ok(rec: Dict[str, Any]) -> bool:
    if rec.get("error") or rec.get("status") == "ollama_error":
        return False
    if "reasoning_half" not in rec:
        return False
    tk = rec.get("task_kind", "mc")
    if tk == "telemath":
        return rec.get("gold_float") is not None
    return rec.get("gold_index") is not None


def normalize_bench_row(raw: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Map release JSON (``half_reasoning_trace``, ``correct_answer``) and/or collector JSONL
    into the internal shape expected by eval: ``reasoning_half``, ``task_kind``,
    ``gold_index`` / ``gold_float``.
    """
    rec = dict(raw)
    half = rec.get("reasoning_half")
    if half is None:
        half = rec.get("half_reasoning_trace")
    if not isinstance(half, str) or not half.strip():
        return None
    rec["reasoning_half"] = half

    choices = rec.get("choices")
    is_mc = isinstance(choices, list) and len(choices) > 0
    tk = rec.get("task_kind")
    if tk == "telemath":
        pass
    elif tk == "mc":
        pass
    elif is_mc:
        rec["task_kind"] = "mc"
        tk = "mc"
    else:
        rec["task_kind"] = "telemath"
        tk = "telemath"

    if tk == "mc":
        gi = rec.get("gold_index")
        if gi is not None:
            rec["gold_index"] = int(gi)
        else:
            ca = rec.get("correct_answer")
            if not isinstance(choices, list) or ca is None:
                return None
            try:
                rec["gold_index"] = choices.index(ca)
            except ValueError:
                return None
        rec.setdefault("gold_float", None)
    else:
        gf = rec.get("gold_float")
        if gf is None:
            ca = rec.get("correct_answer")
            if ca is None:
                return None
            try:
                if isinstance(ca, (int, float)):
                    rec["gold_float"] = float(ca)
                else:
                    rec["gold_float"] = float(str(ca).strip().replace(",", ""))
            except ValueError:
                return None
        else:
            rec["gold_float"] = float(gf)
    return rec


def load_bench_rows(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            print(f"{path}: expected a JSON array of benchmark objects.", file=sys.stderr)
            return []
        for item in data:
            if not isinstance(item, dict):
                continue
            norm = normalize_bench_row(item)
            if norm is not None and bench_row_ok(norm):
                rows.append(norm)
        return rows

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            norm = normalize_bench_row(rec)
            if norm is not None and bench_row_ok(norm):
                rows.append(norm)
    return rows


def load_done_ids(path: Path) -> Set[str]:
    if not path.is_file():
        return set()
    done: Set[str] = set()
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


def infer_experiment_artifact(bench: Path) -> str:
    """``main`` = MC release benchmark; ``aux`` = TeleMath auxiliary."""
    stem = bench.stem.lower()
    if "auxiliary" in stem:
        return "aux"
    if "final_benchmark" in stem or stem == "final_benchmark":
        return "main"
    return "main"


def load_done_ids_experiment(model_dir: Path, artifact: str) -> Set[str]:
    """Resume from staging JSONL and/or a prior consolidated ``{artifact}.json``."""
    done = load_done_ids(model_dir / f"{artifact}.jsonl")
    bundle = model_dir / f"{artifact}.json"
    if bundle.is_file():
        try:
            with bundle.open("r", encoding="utf-8") as f:
                data = json.load(f)
        except json.JSONDecodeError:
            return done
        for r in data.get("responses", []):
            if isinstance(r, dict) and r.get("sample_id"):
                done.add(str(r["sample_id"]))
    return done


def write_experiment_bundle(
    *,
    model_dir: Path,
    artifact: str,
    bench_file: Path,
    model_tag: str,
    summary: Dict[str, Any],
    staging_jsonl: Path,
    rel_tol: float,
    abs_tol: float,
    num_ctx: int,
    think: bool = True,
) -> Path:
    """Write ``Experiments/<model>/{main|aux}.json`` with metadata, summary, and all responses."""
    responses: List[Dict[str, Any]] = []
    if staging_jsonl.is_file():
        with staging_jsonl.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                responses.append(json.loads(line))
    out_json = model_dir / f"{artifact}.json"
    bundle: Dict[str, Any] = {
        "bench_file": str(bench_file),
        "experiment_artifact": artifact,
        "model": model_tag,
        "ollama_think": think,
        "ollama_num_ctx": num_ctx,
        "telemath_rel_tol": rel_tol,
        "telemath_abs_tol": abs_tol,
        "summary": {k: v for k, v in summary.items() if k not in ("params_b", "output_jsonl")},
        "num_responses": len(responses),
        "responses": responses,
    }
    if summary.get("params_b") is not None:
        bundle["params_b"] = summary["params_b"]
    model_dir.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(bundle, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return out_json


def continuation_correct(
    rec: Dict[str, Any],
    pred_index: Optional[int],
    pred_float: Optional[float],
    *,
    rel_tol: float,
    abs_tol: float,
) -> bool:
    tk = rec.get("task_kind", "mc")
    if tk == "telemath":
        gold = rec.get("gold_float")
        if gold is None or pred_float is None:
            return False
        return floats_match(float(pred_float), float(gold), rel_tol=rel_tol, abs_tol=abs_tol)
    gold_i = rec.get("gold_index")
    if gold_i is None or pred_index is None:
        return False
    return int(pred_index) == int(gold_i)


def summarize_continuation_jsonl(
    path: Path,
    model: str,
    *,
    rel_tol: float,
    abs_tol: float,
) -> Dict[str, Any]:
    totals = {
        "n": 0,
        "correct_flips": 0,
        "parse_failures": 0,
        "still_wrong": 0,
        "sum_prompt_tokens": 0,
        "sum_output_tokens": 0,
        "token_records": 0,
    }
    by_subset: DefaultDict[str, Dict[str, int]] = defaultdict(
        lambda: {
            "n": 0,
            "correct_flips": 0,
            "parse_failures": 0,
            "still_wrong": 0,
        }
    )
    if not path.is_file():
        return _summary_dict(model, totals, by_subset)

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            err = rec.get("error")
            sub = str(rec.get("sub_benchmark") or "unknown")
            tk = rec.get("task_kind", "mc")
            if tk == "telemath":
                pred_f = rec.get("continuation_pred_float")
                parse_ok = pred_f is not None and not err
                correct = continuation_correct(rec, None, pred_f, rel_tol=rel_tol, abs_tol=abs_tol)
            else:
                pred_i = rec.get("continuation_pred_index")
                parse_ok = pred_i is not None and not err
                correct = continuation_correct(rec, pred_i, None, rel_tol=rel_tol, abs_tol=abs_tol)

            totals["n"] += 1
            by_subset[sub]["n"] += 1
            if err or not parse_ok:
                totals["parse_failures"] += 1
                by_subset[sub]["parse_failures"] += 1
            elif correct:
                totals["correct_flips"] += 1
                by_subset[sub]["correct_flips"] += 1
            else:
                totals["still_wrong"] += 1
                by_subset[sub]["still_wrong"] += 1

            pt, ot = rec.get("prompt_tokens"), rec.get("output_tokens")
            if pt is not None and ot is not None:
                totals["sum_prompt_tokens"] += int(pt)
                totals["sum_output_tokens"] += int(ot)
                totals["token_records"] += 1

    return _summary_dict(model, totals, by_subset)


def _summary_dict(model: str, totals: Dict[str, Any], by_subset: DefaultDict[str, Dict[str, int]]) -> Dict[str, Any]:
    n = totals["n"]
    cf = totals["correct_flips"]
    tr = totals["token_records"]
    return {
        "model": model,
        "n": n,
        "correct_flips": cf,
        "correct_flip_rate": cf / n if n else 0.0,
        "parse_failures": totals["parse_failures"],
        "parse_failure_rate": totals["parse_failures"] / n if n else 0.0,
        "still_wrong": totals["still_wrong"],
        "mean_prompt_tokens": totals["sum_prompt_tokens"] / tr if tr else None,
        "mean_output_tokens": totals["sum_output_tokens"] / tr if tr else None,
        "mean_total_tokens": (totals["sum_prompt_tokens"] + totals["sum_output_tokens"]) / tr
        if tr
        else None,
        "by_subset": {
            k: {
                "n": v["n"],
                "correct_flips": v["correct_flips"],
                "correct_flip_rate": v["correct_flips"] / v["n"] if v["n"] else 0.0,
                "parse_failures": v["parse_failures"],
                "still_wrong": v["still_wrong"],
            }
            for k, v in sorted(by_subset.items())
        },
    }


def eval_one_model(
    model: str,
    bench: List[Dict[str, Any]],
    out_path: Path,
    host: str,
    num_predict: int,
    temperature: float,
    timeout: int,
    *,
    rel_tol: float,
    abs_tol: float,
    num_ctx: int,
    done_ids: Optional[Set[str]] = None,
    think: bool = True,
) -> Dict[str, Any]:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = done_ids if done_ids is not None else load_done_ids(out_path)

    with out_path.open("a", encoding="utf-8") as out:
        for rec in tqdm(bench, desc=f"eval {model}"):
            sid = rec["sample_id"]
            if sid in done:
                continue

            q = rec["question"]
            half = str(rec.get("reasoning_half") or "")
            tk = rec.get("task_kind", "mc")

            pred_i: Optional[int] = None
            pred_f: Optional[float] = None
            raw_for_parse = ""
            continuation_thinking = ""
            continuation_response = ""
            pt: Optional[int] = None
            ot: Optional[int] = None
            err: Optional[str] = None

            try:
                if tk == "telemath":
                    prompt = build_continuation_telemath_prompt(q, half)
                else:
                    choices = list(rec["choices"])
                    prompt = build_continuation_mc_prompt(q, choices, half)

                gen = ollama_generate(
                    model,
                    prompt,
                    host=host,
                    think=think,
                    num_predict=num_predict,
                    temperature=temperature,
                    timeout=timeout,
                    num_ctx=num_ctx,
                )
                continuation_thinking = (gen.thinking or "").strip()
                continuation_response = (gen.response or "").strip()
                raw_for_parse = (
                    model_text_for_parsing(gen.thinking, gen.response)
                    if think
                    else continuation_response
                )
                if tk == "telemath":
                    pred_f = extract_float_answer(raw_for_parse)
                else:
                    pred_i = parse_mc_output(raw_for_parse, list(rec["choices"]))
                pt, ot = gen.prompt_tokens, gen.output_tokens
            except Exception as e:
                err = str(e)

            if tk == "telemath":
                gold_f = rec.get("gold_float")
                corr = continuation_correct(rec, None, pred_f, rel_tol=rel_tol, abs_tol=abs_tol)
                row_out: Dict[str, Any] = {
                    "sample_id": sid,
                    "sub_benchmark": rec.get("sub_benchmark"),
                    "task_kind": "telemath",
                    "model": model,
                    "continuation_raw": raw_for_parse,
                    "continuation_pred_index": None,
                    "continuation_pred_float": pred_f,
                    "gold_index": None,
                    "gold_float": gold_f,
                    "correct_flip": corr and not err,
                    "parse_ok": pred_f is not None and not err,
                    "prompt_tokens": pt,
                    "output_tokens": ot,
                    "error": err,
                }
                if continuation_thinking:
                    row_out["continuation_thinking"] = continuation_thinking
                if continuation_response:
                    row_out["continuation_response"] = continuation_response
            else:
                gold_i = rec.get("gold_index")
                corr = continuation_correct(rec, pred_i, None, rel_tol=rel_tol, abs_tol=abs_tol)
                row_out = {
                    "sample_id": sid,
                    "sub_benchmark": rec.get("sub_benchmark"),
                    "task_kind": "mc",
                    "model": model,
                    "continuation_raw": raw_for_parse,
                    "continuation_pred_index": pred_i,
                    "continuation_pred_float": None,
                    "gold_index": gold_i,
                    "gold_float": None,
                    "correct_flip": corr and not err,
                    "parse_ok": pred_i is not None and not err,
                    "prompt_tokens": pt,
                    "output_tokens": ot,
                    "error": err,
                }
                if continuation_thinking:
                    row_out["continuation_thinking"] = continuation_thinking
                if continuation_response:
                    row_out["continuation_response"] = continuation_response
            out.write(json.dumps(row_out, ensure_ascii=False) + "\n")
            out.flush()

    return summarize_continuation_jsonl(out_path, model, rel_tol=rel_tol, abs_tol=abs_tol)


def main() -> None:
    parser = argparse.ArgumentParser(description="Continuation eval for TeleResilienceBench.")
    parser.add_argument("--bench", type=Path, default=ROOT / "data" / "tele_resilience_bench.jsonl")
    parser.add_argument("--models-config", type=Path, default=ROOT / "configs" / "models.yaml")
    parser.add_argument("--model", type=str, default=None, help="Single Ollama model tag (overrides config list).")
    parser.add_argument(
        "--experiments-dir",
        type=Path,
        default=ROOT / "Experiments",
        help="Per-model folders with main.json / aux.json plus staging .jsonl for resume.",
    )
    parser.add_argument(
        "--experiment-artifact",
        type=str,
        choices=("auto", "main", "aux"),
        default="auto",
        help="Which file to write per model: main.json (MC) or aux.json (TeleMath). "
        "'auto' uses the benchmark filename (final_benchmark -> main, Auxiliary -> aux).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="If set, also write legacy continuation_<model>.jsonl and summary.json here.",
    )
    parser.add_argument("--ollama-host", type=str, default="http://localhost:11434")
    parser.add_argument("--num-predict", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--telemath-rel-tol", type=float, default=1e-3)
    parser.add_argument("--telemath-abs-tol", type=float, default=1e-5)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument(
        "--ollama-num-ctx",
        type=int,
        default=2048,
        help="Per-request context for continuation calls (default 2048, match collector).",
    )
    parser.add_argument(
        "--no-think",
        action="store_true",
        help="Disable Ollama native thinking; parse only the visible response (legacy behavior).",
    )
    args = parser.parse_args()
    think_enabled = not args.no_think

    bench = load_bench_rows(args.bench)
    if not bench:
        print("No valid benchmark rows found.", file=sys.stderr)
        sys.exit(1)

    artifact = infer_experiment_artifact(args.bench) if args.experiment_artifact == "auto" else args.experiment_artifact

    if args.model:
        models = [{"tag": args.model, "params_b": None}]
    else:
        models = load_models_from_yaml(args.models_config)

    summaries: List[Dict[str, Any]] = []
    args.experiments_dir.mkdir(parents=True, exist_ok=True)

    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)

    for entry in models:
        tag = entry["tag"]
        safe = sanitize_model_tag(tag)
        model_dir = args.experiments_dir / safe
        staging_jsonl = model_dir / f"{artifact}.jsonl"
        done = load_done_ids_experiment(model_dir, artifact)

        s = eval_one_model(
            tag,
            bench,
            staging_jsonl,
            host=args.ollama_host,
            num_predict=args.num_predict,
            temperature=args.temperature,
            timeout=args.timeout,
            rel_tol=args.telemath_rel_tol,
            abs_tol=args.telemath_abs_tol,
            num_ctx=args.ollama_num_ctx,
            done_ids=done,
            think=think_enabled,
        )
        s["params_b"] = entry.get("params_b")
        s["think"] = think_enabled
        s["output_jsonl"] = str(staging_jsonl)
        s["telemath_rel_tol"] = args.telemath_rel_tol
        s["telemath_abs_tol"] = args.telemath_abs_tol
        summaries.append(s)

        bundle_path = write_experiment_bundle(
            model_dir=model_dir,
            artifact=artifact,
            bench_file=args.bench,
            model_tag=tag,
            summary=s,
            staging_jsonl=staging_jsonl,
            rel_tol=args.telemath_rel_tol,
            abs_tol=args.telemath_abs_tol,
            num_ctx=args.ollama_num_ctx,
            think=think_enabled,
        )
        print(f"Wrote {bundle_path}")

        if args.output_dir is not None:
            legacy_jsonl = args.output_dir / f"continuation_{safe}.jsonl"
            if legacy_jsonl.resolve() != staging_jsonl.resolve():
                legacy_jsonl.write_text(staging_jsonl.read_text(encoding="utf-8"), encoding="utf-8")

    payload = {
        "bench_file": str(args.bench),
        "experiment_artifact": artifact,
        "num_bench_rows": len(bench),
        "ollama_num_ctx": args.ollama_num_ctx,
        "ollama_think": think_enabled,
        "models": {s["model"]: s for s in summaries},
    }
    summary_name = f"summary_{artifact}.json"
    summary_path = args.experiments_dir / summary_name
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(f"Wrote aggregate summary to {summary_path}")

    if args.output_dir is not None:
        legacy_summary = args.output_dir / "summary.json"
        with legacy_summary.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(f"Wrote legacy summary to {legacy_summary}")


if __name__ == "__main__":
    main()
