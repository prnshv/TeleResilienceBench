"""Expected row counts for GSMA test splits (ETA, dashboards, card totals)."""

from __future__ import annotations

# Hugging Face GSMA/ot-full card (test split)
SUBSET_ROW_COUNTS_OT_FULL: dict[str, int] = {
    "teleqna": 10_000,
    "teletables": 500,
    "telemath": 500,
    "telelogs": 864,
    "3gpp_tsg": 2_000,
    "oranbench": 1_500,
    "srsranbench": 1_502,
    "sixg_bench": 3_722,
}

# Hugging Face GSMA/ot-lite card (test split; ~1,850 rows total)
SUBSET_ROW_COUNTS_OT_LITE: dict[str, int] = {
    "teleqna": 1_000,
    "teletables": 100,
    "telemath": 100,
    "telelogs": 100,
    "3gpp_tsg": 100,
    "oranbench": 150,
    "srsranbench": 150,
    "sixg_bench": 150,
}


def subset_row_counts_for_dataset(dataset_name: str) -> dict[str, int]:
    """Pick per-subset plan counts from the HF dataset id."""
    if "ot-lite" in dataset_name:
        return SUBSET_ROW_COUNTS_OT_LITE
    return SUBSET_ROW_COUNTS_OT_FULL


def planned_rows(
    subsets: list[str],
    max_samples_per_subset: int | None,
    dataset_name: str = "GSMA/ot-full",
) -> int:
    table = subset_row_counts_for_dataset(dataset_name)
    total = 0
    for s in subsets:
        n = table.get(s, 0)
        if max_samples_per_subset is not None:
            n = min(n, max_samples_per_subset)
        total += n
    return total
