# TeleResilienceBench

Resilience benchmark built from **[GSMA/ot-lite](https://huggingface.co/datasets/GSMA/ot-lite)** on Hugging Face. The pipeline runs a base model on the **test split**, keeps only **wrong but parseable** answers, **halves** the reasoning trace, then (optionally) evaluates whether continuation models **flip** to the correct answer.

All subsets below are loaded from **ot-lite**. `sample_id` values use an **`ot-lite:`** prefix.

## Subsets (ot-lite)

| Subset | Task shape |
|--------|------------|
| `teleqna`, `teletables`, `oranbench`, `srsranbench`, `sixg_bench` | Multiple choice: `choices` + index `answer`. |
| `telelogs` | Root-cause labels `C1`–`C8`; normalized to MC with options `C1`…`C8`. |
| `3gpp_tsg` | Working-group labels; normalized to MC over the **16** labels in the test split (sorted). |
| `telemath` | Numeric `answer`; graded with `math.isclose`-style tolerances in the scripts. |

## Setup

```bash
cd TeleResilienceBench
python3 -m venv .venv || virtualenv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Use a local [Ollama](https://ollama.com) install with the base model you pass to the collector (e.g. `qwen3.5:2b`).

## 1. Collect base traces (wrong + parseable only)

```bash
cd TeleResilienceBench
source .venv/bin/activate

python scripts/01_collect_base_traces.py \
  --dataset GSMA/ot-lite \
  --base-model qwen3.5:2b \
  --output data/tele_resilience_bench.jsonl \
  --metadata data/collect_metadata.json \
  --progress-log data/collect_progress.jsonl \
  2>&1 | tee logs/collect_base.log
```

Monitor progress (point at ot-lite if metadata is empty):

```bash
python scripts/monitor_collect.py --dataset GSMA/ot-lite -w
```

`--progress-log` defaults to `data/collect_progress.jsonl` so you can **resume** after interruptions. Prompts ask for a final line **`Final Answer: …`**; parsing uses **thinking + response** for Qwen-style runs.

To start clean, remove `data/tele_resilience_bench.jsonl`, `data/collect_progress.jsonl`, and `data/collect_metadata.json`.

**Export slim JSON releases** (MC vs TeleMath):

```bash
python3 scripts/export_release_benchmarks.py
```

## 2. Continuation eval

```bash
python scripts/02_eval_continuation.py \
  --bench data/tele_resilience_bench.jsonl \
  --models-config configs/models.yaml \
  --output-dir out/eval/
```

## 3. Report

```bash
python scripts/03_report.py \
  --summary out/eval/summary.json \
  --models-config configs/models.yaml \
  --artifacts-dir out/report/
```

## LaTeX main table + efficiency plots (`Graphs/`)

**Subset flip-rate table (LaTeX):** from the repo root, with `data/tele_resilience_bench.jsonl` and `Experiments/<model>/main.jsonl` in place:

```bash
python main_table.py \
  --bench data/tele_resilience_bench.jsonl \
  --experiments Experiments \
  -o Experiments/main_table.tex
```

Per-subset CFR/NF/WF match `main_table.py` (gold/base indices from the bench file). **Macro Average** is the **unweighted mean** of the **seven** MC subset percentages (TeleQnA, TeleTables, TeleLogs, 3GPP\_TSG, ORANBench, srsRANBench, SixG\_Bench)—the same aggregation as the scatter plots below.

**Efficiency scatter figures** (mean continuation tokens vs CFR, and VRAM usage % vs CFR for the eight continuation models under `Experiments/`):

```bash
source .venv/bin/activate
python Graphs/token_scatter.py    # → Graphs/token_scatter.png + .pdf
python Graphs/vram_scatter.py     # → Graphs/VRAM_scatter.png + .pdf
```

Shared model list, per-model VRAM (GB), label offsets, and seven-subset CFR logic live in `Graphs/scatter_common.py`. Optional: `--bench` (default `data/tele_resilience_bench.jsonl`), `--experiments`, `--ref-vram-gb` on `vram_scatter.py`.

---

## Release artifacts (from this ot-lite run)

The full collector output is **`data/tele_resilience_bench.jsonl`**. From it we ship two JSON files (regenerate with `scripts/export_release_benchmarks.py` after a new collection):

| Artifact | Description | Samples |
|----------|-------------|--------:|
| **`data/final_benchmark.json`** | Main benchmark: **MC only** — `sample_id`, `sub_benchmark`, `question`, `choices`, `correct_answer`, `incorrect_answer`, `half_reasoning_trace`. | **818** |
| **`data/AuxiliaryBenchmark.json`** | **TeleMath only** — same idea without `choices` (numeric gold vs wrong base answer). | **77** |

Together that is **895** resilience items (818 + 77), all from **ot-lite** wrong+parseable traces.

**MC distribution** (rows in `final_benchmark.json`):

| `sub_benchmark` | Count |
|-----------------|------:|
| `teleqna` | 359 |
| `sixg_bench` | 97 |
| `3gpp_tsg` | 90 |
| `telelogs` | 86 |
| `teletables` | 76 |
| `oranbench` | 63 |
| `srsranbench` | 47 |
| **Total** | **818** |

### Example: one MC row (`final_benchmark.json`)

Fields are abbreviated for display; `half_reasoning_trace` is much longer on disk.

```json
{
  "sample_id": "ot-lite:teleqna:4",
  "sub_benchmark": "teleqna",
  "question": "When can the SDT procedure over CG resources be initiated? [3GPP Release 18]",
  "choices": [
    "When there is valid UL timing alignment.",
    "When the DL RSRP of the SSB is above a configured threshold.",
    "When the UE is in RRC_INACTIVE state.",
    "When the UE is in RRC_IDLE state.",
    "When the UE receives an RRCRelease message."
  ],
  "correct_answer": "When there is valid UL timing alignment.",
  "incorrect_answer": "When the UE receives an RRCRelease message.",
  "half_reasoning_trace": "The user wants me to answer a multiple-choice question about 3GPP Release 18…"
}
```

### Example: one TeleMath row (`AuxiliaryBenchmark.json`)

```json
{
  "sample_id": "ot-lite:telemath:1",
  "sub_benchmark": "telemath",
  "question": "Calculate the probability of system error for a signal modulated using 2-PAM, …",
  "correct_answer": "0.033",
  "incorrect_answer": "2.0",
  "half_reasoning_trace": "Thinking Process:\n\n1. **Analyze the Request:**\n   * Role: Expert in telecommunications mathematics…"
}
```

## Layout

- `src/` — GSMA loading, Ollama client, parsing / half-trace helpers  
- `scripts/` — collect, eval, report, `export_release_benchmarks.py`  
- `data/` — `tele_resilience_bench.jsonl`, `final_benchmark.json`, `AuxiliaryBenchmark.json`  
- `Experiments/` — per-model `main.json` / `main.jsonl` and generated `main_table.tex`  
- `Graphs/` — `token_scatter.py`, `vram_scatter.py`, `scatter_common.py`, exported `.png` / `.pdf` figures  
- `main_table.py` — build LaTeX CFR/NF/WF table from bench + `Experiments/*/main.jsonl`  
- `out/eval/`, `out/report/` — optional local outputs from eval + `03_report.py` (gitignored)  
- `configs/models.yaml` — continuation model tags  
