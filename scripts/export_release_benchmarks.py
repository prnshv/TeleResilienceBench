#!/usr/bin/env python3
"""Build slim JSON exports from ``tele_resilience_bench.jsonl``."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "data" / "tele_resilience_bench.jsonl",
    )
    parser.add_argument(
        "--mc-out",
        type=Path,
        default=ROOT / "data" / "final_benchmark.json",
        help="MC-only export",
    )
    parser.add_argument(
        "--telemath-out",
        type=Path,
        default=ROOT / "data" / "AuxiliaryBenchmark.json",
        help="TeleMath-only concise export",
    )
    args = parser.parse_args()

    if not args.input.is_file():
        print(f"Missing: {args.input}", file=sys.stderr)
        sys.exit(1)

    mc_rows: list[dict] = []
    tm_rows: list[dict] = []

    with args.input.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("status") == "ollama_error" or rec.get("error"):
                continue

            tk = rec.get("task_kind")
            half = rec.get("reasoning_half")
            if not isinstance(half, str):
                continue

            if tk == "mc":
                choices = rec.get("choices")
                if not isinstance(choices, list):
                    continue
                mc_rows.append(
                    {
                        "sample_id": rec.get("sample_id"),
                        "sub_benchmark": rec.get("sub_benchmark"),
                        "question": rec.get("question"),
                        "choices": choices,
                        "correct_answer": rec.get("correct_answer"),
                        "incorrect_answer": rec.get("incorrect_answer"),
                        "half_reasoning_trace": half,
                    }
                )
            elif tk == "telemath":
                tm_rows.append(
                    {
                        "sample_id": rec.get("sample_id"),
                        "sub_benchmark": rec.get("sub_benchmark"),
                        "question": rec.get("question"),
                        "correct_answer": rec.get("correct_answer"),
                        "incorrect_answer": rec.get("incorrect_answer"),
                        "half_reasoning_trace": half,
                    }
                )

    def write_json(path: Path, obj: object) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(obj, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    write_json(args.mc_out, mc_rows)
    write_json(args.telemath_out, tm_rows)

    print(
        json.dumps(
            {
                "mc_count": len(mc_rows),
                "telemath_count": len(tm_rows),
                "mc_out": str(args.mc_out),
                "telemath_out": str(args.telemath_out),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
