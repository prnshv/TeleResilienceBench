from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Iterator, List, Literal, Optional

from datasets import load_dataset

DATASET_NAME = "GSMA/ot-full"


def dataset_slug(dataset_name: str) -> str:
    """Last path segment of HF id, e.g. GSMA/ot-lite -> ot-lite."""
    return dataset_name.rstrip("/").split("/")[-1]

SUBSETS: tuple[str, ...] = (
    "teleqna",
    "teletables",
    "telemath",
    "telelogs",
    "3gpp_tsg",
    "oranbench",
    "srsranbench",
    "sixg_bench",
)

MC_SUBSETS = frozenset(
    {"teleqna", "teletables", "oranbench", "srsranbench", "sixg_bench"}
)
TELELOGS_SUBSET = "telelogs"
TELEMATH_SUBSET = "telemath"
TSG_SUBSET = "3gpp_tsg"

TaskKind = Literal["mc", "telemath"]


@dataclass
class UnifiedRow:
    """Per-row view aligned with TeleResilienceBench scripts."""

    subset: str
    row_index: int
    task_kind: TaskKind
    question: str
    choices: List[str]
    gold_index: Optional[int]
    gold_float: Optional[float]
    difficulty: Optional[str] = None
    # HF dataset id (e.g. GSMA/ot-full). Used to namespace sample_id for non-ot-full sets.
    dataset_name: str = DATASET_NAME

    @property
    def sample_id(self) -> str:
        # Keep legacy ids for ot-full so existing bench JSONL and resumes stay valid.
        if dataset_slug(self.dataset_name) == "ot-full":
            return f"{self.subset}:{self.row_index}"
        return f"{dataset_slug(self.dataset_name)}:{self.subset}:{self.row_index}"

    @property
    def correct_answer_text(self) -> str:
        if self.task_kind == "telemath" and self.gold_float is not None:
            return repr(self.gold_float)
        if self.gold_index is not None and 0 <= self.gold_index < len(self.choices):
            return str(self.choices[self.gold_index])
        return ""


def _difficulty_from_row(row: dict[str, Any]) -> Optional[str]:
    if row.get("difficulty") is not None:
        d = str(row["difficulty"]).strip().lower()
        return d or None
    if row.get("difficult") is not None:
        return "hard" if row["difficult"] else "easy"
    return None


def _telelogs_gold_index(answer: str) -> Optional[int]:
    a = (answer or "").strip().upper()
    if len(a) == 2 and a[0] == "C" and a[1].isdigit():
        n = int(a[1])
        if 1 <= n <= 8:
            return n - 1
    m = re.match(r"^C(\d+)$", a)
    if m:
        n = int(m.group(1))
        if 1 <= n <= 8:
            return n - 1
    return None


@lru_cache(maxsize=8)
def _tsg_sorted_labels(dataset_name: str, subset: str, split: str) -> tuple[str, ...]:
    ds = load_dataset(dataset_name, subset, split=split)
    labels = sorted({str(x).strip() for x in ds["answer"] if str(x).strip()})
    return tuple(labels)


def row_from_hf(subset: str, row_index: int, row: dict[str, Any], dataset_name: str, split: str) -> UnifiedRow:
    difficulty = _difficulty_from_row(row)
    question = str(row["question"])
    dn = dataset_name

    if subset in MC_SUBSETS:
        choices = [str(c) for c in list(row["choices"])]
        return UnifiedRow(
            subset=subset,
            row_index=row_index,
            task_kind="mc",
            question=question,
            choices=choices,
            gold_index=int(row["answer"]),
            gold_float=None,
            difficulty=difficulty,
            dataset_name=dn,
        )

    if subset == TELELOGS_SUBSET:
        choices = [f"C{i}" for i in range(1, 9)]
        gidx = _telelogs_gold_index(str(row["answer"]))
        if gidx is None:
            raise ValueError(f"telelogs bad answer: {row.get('answer')!r}")
        return UnifiedRow(
            subset=subset,
            row_index=row_index,
            task_kind="mc",
            question=question,
            choices=choices,
            gold_index=gidx,
            gold_float=None,
            difficulty=difficulty,
            dataset_name=dn,
        )

    if subset == TSG_SUBSET:
        labels = list(_tsg_sorted_labels(dataset_name, subset, split))
        gold = str(row["answer"]).strip()
        if gold not in labels:
            raise ValueError(f"3gpp_tsg unknown label {gold!r}")
        gidx = labels.index(gold)
        return UnifiedRow(
            subset=subset,
            row_index=row_index,
            task_kind="mc",
            question=question,
            choices=labels,
            gold_index=gidx,
            gold_float=None,
            difficulty=difficulty,
            dataset_name=dn,
        )

    if subset == TELEMATH_SUBSET:
        ans = row["answer"]
        gold_f = float(ans) if isinstance(ans, (int, float)) else float(str(ans).strip())
        return UnifiedRow(
            subset=subset,
            row_index=row_index,
            task_kind="telemath",
            question=question,
            choices=[],
            gold_index=None,
            gold_float=gold_f,
            difficulty=difficulty,
            dataset_name=dn,
        )

    raise ValueError(f"Unhandled subset {subset!r}")


def load_gsma_subset(
    subset: str,
    dataset_name: str = DATASET_NAME,
    split: str = "test",
    max_samples: Optional[int] = None,
) -> List[UnifiedRow]:
    if subset not in SUBSETS:
        raise ValueError(f"Unknown subset {subset!r}. Expected one of {SUBSETS}")
    if subset == TSG_SUBSET:
        _tsg_sorted_labels(dataset_name, subset, split)
    ds = load_dataset(dataset_name, subset, split=split)
    out: List[UnifiedRow] = []
    for i, row in enumerate(ds):
        out.append(row_from_hf(subset, i, row, dataset_name, split))
        if max_samples is not None and len(out) >= max_samples:
            break
    return out


def iter_all_subsets(
    subsets: Optional[List[str]] = None,
    dataset_name: str = DATASET_NAME,
    split: str = "test",
    max_samples_per_subset: Optional[int] = None,
) -> Iterator[UnifiedRow]:
    use = tuple(subsets) if subsets else SUBSETS
    for name in use:
        if name not in SUBSETS:
            raise ValueError(f"Unknown subset {name!r}")
        for row in load_gsma_subset(
            name,
            dataset_name=dataset_name,
            split=split,
            max_samples=max_samples_per_subset,
        ):
            yield row


# Backwards compatibility alias
GsmaRow = UnifiedRow
