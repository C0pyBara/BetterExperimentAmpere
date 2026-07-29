# Table Header Detection — Experiment Pipeline

A modular pipeline for evaluating how well LLMs recognize table headers, served
through **vLLM** (or any OpenAI-compatible endpoint). It supports the
RealHitBench annotation viewpoints (`pubtables_complex_top500`,
`maximum_viewpoint`, `table_normalization`), both **JSON** and **HTML** table
serializations, and the **zero / few-shot / reasoning** prompting strategies.

The pipeline sends each table to the model, parses the returned header
coordinates, scores them against the ground truth, and writes incremental
checkpoints and summary metrics. A separate analysis tool compares runs across
models and input formats with bootstrap confidence intervals.

---

## Table of contents

- [Overview](#overview)
- [Key features](#key-features)
- [Repository layout](#repository-layout)
- [How the pipeline works](#how-the-pipeline-works)
- [Installation](#installation)
- [Data layout](#data-layout)
- [Running an experiment](#running-an-experiment)
  - [Reproducibility across models](#reproducibility-across-models)
  - [Re-running failed / truncated tasks](#re-running-failed--truncated-tasks)
  - [Context window](#context-window)
- [Command-line reference](#command-line-reference)
- [Outputs](#outputs)
- [Metrics](#metrics)
- [Analysis](#analysis)
- [Auxiliary scripts](#auxiliary-scripts)
- [Authors](#authors)
- [Citation](#citation)

---

## Overview

The task is **table header recognition**: given the cells of a table, decide
which cells are headers. The pipeline treats an LLM as the predictor. For every
table it:

1. loads the table and its ground-truth header annotation;
2. serializes the table as JSON or HTML and builds a prompt
   (zero-shot, few-shot, or reasoning);
3. calls the model through an OpenAI-compatible `/v1/chat/completions` endpoint,
   handling token budgets, continuations, and API errors;
4. parses the model's answer into header coordinates (`row col | text`);
5. computes precision / recall / F1 against the ground truth;
6. persists results incrementally so long runs can resume.

The design goal is a fair, reproducible comparison **across models, prompts, and
formats** — the same tables, the same budgets, and analysis that only compares
tasks both runs actually completed.

---

## Key features

- **OpenAI-compatible transport** — works against a local vLLM server or a hosted
  gateway; the base URL, model name, and an alias for output naming are all
  configurable.
- **Token-budget control** — a `BudgetController` sizes `max_tokens` per request
  against the model's context window, which is auto-detected from `/v1/models`
  (overridable).
- **Header-aware chunking** — tables too large for the window are split while
  keeping header rows in view, with absolute coordinates preserved so the
  reassembled prediction still lines up with the ground truth.
- **Continuation handling** — truncated generations can be continued for a
  bounded (or unbounded) number of rounds, with an optional forced
  "thinking-off" final answer as a last resort.
- **Thinking / reasoning control** — reasoning can be toggled per prompt via
  several mechanisms (`chat_template`, `enable_thinking`, `reasoning`, or off),
  with a separate token budget for non-thinking calls.
- **Response caching** — responses can be cached by `request_id` so reruns don't
  re-bill or re-hit the endpoint.
- **Incremental checkpoints** — progress is saved continuously; runs can be
  retried, capped-retried, or restricted to an explicit task list.
- **Cross-model / cross-format analysis** — paired comparisons with
  bootstrap confidence intervals, computed only on tasks common to both runs.

---

## Repository layout

```
run.py                      experiment runner (CLI entry point)
analyze_results.py          cross-model / cross-format analysis (CIs, paired comparisons)
analyze_stats.py            dataset / run statistics helper
extract_undone.py           build a list of not-yet-completed tasks (-> undone.json)
undone.json                 example task list of outstanding {stem, prompt, fmt} items
requirements.txt

table_header_exp/           the pipeline package
    config.py               single source of configuration (Config, experiment plan)
    datamodel.py            ApiResult (per-request telemetry)
    parsing.py              parse the model answer and classify API errors
    evaluation.py           pure metrics (coord / type / soft-spanning / text)
    loading.py              load tables and ground truth
    prompts.py              assemble messages and compute the max_tokens budget
    transport.py            BudgetController, async calls, continuation
    chunking.py             header-aware chunking with absolute coordinates
    persistence.py          incremental checkpoints, summary metrics
    orchestrator.py         Collector — orchestrates a run

Get_500_Tables_from_PubTables/   PubTables-1M table selection (input data)
Convert_from_xlsx_to_Json/       XLSX -> JSON conversion (input data)
Convert_from_json_to_html/       JSON -> HTML conversion (input data)
prompts/                         prompt templates (zero / few-shot / reasoning)
api_cache/                       cached API responses (by request_id)
```

> The three `Convert_*` / `Get_*` directories and `prompts/` are resolved
> relative to `--project-root` (see [Data layout](#data-layout)).

---

## How the pipeline works

A single table flows through the package modules like this:

1. **`config.py`** builds a `Config` object and the experiment plan (which
   tables x which prompts x which formats). Every CLI flag maps onto a field of
   `Config`.
2. **`loading.py`** reads the selected tables and their ground-truth header
   annotations.
3. **`chunking.py`** decides whether a table fits the context budget; if not, it
   splits the table header-aware, keeping header rows visible and preserving
   absolute cell coordinates.
4. **`prompts.py`** serializes the (chunk of the) table as JSON or HTML, selects
   the prompt strategy, and computes the `max_tokens` budget for the call.
5. **`transport.py`** issues the async request through the OpenAI-compatible API.
   Its `BudgetController` enforces the token budget; continuations recover
   truncated answers; caching avoids re-billing.
6. **`parsing.py`** turns the raw completion into header coordinates and, on
   failure, classifies the API error (so genuine model misses are separated from
   transport/format failures).
7. **`evaluation.py`** computes the metrics (coordinate match, type, soft
   spanning, text).
8. **`persistence.py`** writes an incremental checkpoint and updates the summary
   metrics after each task.
9. **`orchestrator.py`** (`Collector`) drives the whole loop and exposes the
   `run`, `run_retry`, `run_retry_capped`, and `run_retry_list` entry points that
   `run.py` calls.

`datamodel.py` defines `ApiResult`, the per-request telemetry record that ties
the raw response, parsed prediction, and error classification together.

---

## Installation

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

You also need a running OpenAI-compatible endpoint (e.g. a vLLM server) reachable
at the URL you pass to `--vllm-url`.

---

## Data layout

The runner looks for the input-data directories and `prompts/` **relative to
`--project-root`** (which defaults to the current directory):

```
<project-root>/
    Get_500_Tables_from_PubTables/...
    Convert_from_xlsx_to_Json/...
    Convert_from_json_to_html/...
    prompts/
```

---

## Running an experiment

Basic run against a local vLLM server:

```bash
python run.py \
  --vllm-url http://127.0.0.1:8000/v1 \
  --model Qwen/Qwen3.5-9B --model-alias qwen35_9b \
  --project-root ~/okhotin/1/BetterExperimentAmpere \
  --output-dir results --concurrency 8 \
  --total-tables 1000 --format-ratio 50:50 \
  --temperature 0.0 --seed 42
```

### Reproducibility across models

To evaluate several models on the **exact same tables**, reuse the table
selection saved by a previous run:

```bash
python run.py ... --table-seed results/run_<...>/selected_tables.json
```

### Re-running failed / truncated tasks

```bash
# retry everything that failed or was truncated
python run.py ... --retry        results/run_<...>/checkpoints/checkpoint_latest.json

# retry, but cap continuation effort (for tasks that keep hitting the budget)
python run.py ... --retry-capped results/run_<...>/checkpoints/checkpoint_latest.json

# run ONLY an explicit list of {stem, prompt, fmt} tasks (saved as a new run)
python run.py ... --retry-list undone.json
```

An outstanding-task list can be generated with `extract_undone.py` (see
[Auxiliary scripts](#auxiliary-scripts)).

### Context window

The window size is auto-detected from `/v1/models`. Override it with
`--context-window N`, or disable detection entirely with
`--no-auto-detect-window`. For hosted APIs that lack a `/tokenize` endpoint, add
`--no-tokenizer`.

---

## Command-line reference

All flags override the corresponding field in `Config`; anything left unset keeps
the `Config` default. Grouped by purpose:

**Endpoint & model**

| Flag | Meaning |
|---|---|
| `--vllm-url` | Base URL of the OpenAI-compatible endpoint (e.g. `.../v1`). |
| `--model` | Model name passed to the API. |
| `--model-alias` | Short alias used in output/run naming. |
| `--extra-body JSON` | Extra JSON merged into every request (e.g. provider routing options). |
| `--no-tokenizer` | Skip the `/tokenize` endpoint (hosted APIs often lack it). |
| `--cache-dir DIR` | Cache responses by `request_id` to avoid re-billing on reruns. |

**Experiment scope**

| Flag | Meaning |
|---|---|
| `--project-root` | Root for input-data dirs and `prompts/`. Default: current dir. |
| `--output-dir` | Where runs and checkpoints are written. |
| `--total-tables` | Number of tables to evaluate. |
| `--format-ratio` | JSON:HTML split, e.g. `50:50`. |
| `--table-seed PATH` | Reuse a saved table selection for cross-model reproducibility. |

**Sampling & concurrency**

| Flag | Meaning |
|---|---|
| `--concurrency` | Number of concurrent requests. |
| `--max-tokens` | Generation token budget per request. |
| `--temperature` | Sampling temperature (use `0.0` for determinism). |
| `--seed` | Random seed. |
| `--timeout` | Per-request timeout (seconds). |
| `--inter-delay` | Delay inserted between requests (seconds). |
| `--early-stop N` | Abort after N consecutive failures. |

**Context window & chunking**

| Flag | Meaning |
|---|---|
| `--context-window N` | Force the context window size. |
| `--no-auto-detect-window` | Do not auto-detect the window from `/v1/models`. |
| `--chunk-strategy {header_aware,whole}` | How to split oversized tables. |
| `--header-zone-rows N` | Number of header rows kept visible in each chunk. |
| `--max-input-tokens N` | Skip tables whose input exceeds N tokens (`0` = no limit). |

**Continuation & reasoning**

| Flag | Meaning |
|---|---|
| `--no-continuation` | Disable continuing truncated generations. |
| `--max-continuation-rounds N` | Cap continuation rounds (`0` = unlimited, bounded by budget). |
| `--no-force-answer` | Do not force a thinking-off final answer as a last resort. |
| `--no-disable-thinking` | Treat the model as not supporting a thinking-off switch. |
| `--thinking-off-mode {chat_template,enable_thinking,reasoning,none}` | How to turn thinking off for non-reasoning prompts. |
| `--max-tokens-nonthinking` | Separate token budget for non-thinking calls. |

**Modes & logging**

| Flag | Meaning |
|---|---|
| `--retry CHECKPOINT_PATH` | Retry failed / truncated tasks from a checkpoint. |
| `--retry-capped CHECKPOINT_PATH` | Retry with capped continuation effort. |
| `--retry-list LIST_JSON` | Run only the `{stem, prompt, fmt}` tasks in a JSON list (saved as a new run). |
| `--log-level` | Logging level (e.g. `INFO`, `DEBUG`). |

> Exactly one mode runs per invocation: `--retry`, `--retry-capped`, and
> `--retry-list` are mutually exclusive; with none of them, a fresh collection
> run is performed.

---

## Outputs

A run writes into `--output-dir` under a `run_<...>` folder, typically including:

- `selected_tables.json` — the exact set of tables used (feed it back via
  `--table-seed` for reproducibility).
- `checkpoints/checkpoint_latest.json` — incremental progress (input to
  `--retry` / `--retry-capped`).
- summary metrics aggregated over completed tasks.

---

## Metrics

`evaluation.py` computes cell-level scores comparing predicted header cells to
the ground truth:

- **coord** — exact coordinate match of header cells.
- **type** — correctness of the assigned cell type/role.
- **soft-spanning** — a relaxed match that credits spanning (merged) header
  cells computed from geometry rather than requiring an exact literal match.
- **text** — textual match of the header content.

Primary reporting is **precision / recall / F1** at the cell level. Parsing and
API failures are classified separately (`parsing.py`) so that transport/format
errors are not counted as model misses.

---

## Analysis

```bash
python analyze_results.py results/run_A results/run_B \
  --output-dir analysis --metric f1
```

Produces three CSVs:

- **`paired_model_comparison.csv`** — paired F1 deltas between models on
  identical tasks, with bootstrap confidence intervals.
- **`paired_format_comparison.csv`** — JSON vs HTML on the intersection of
  successfully parsed tables (avoiding bias from chunking being available for
  one format but not the other).
- **`comparison_by_model_format.csv`** — the end-to-end picture, including
  failures/refusals.

---

## Auxiliary scripts

- **`extract_undone.py`** — scans a run's checkpoints and emits a JSON list of
  outstanding `{stem, prompt, fmt}` tasks (e.g. `undone.json`) to feed back via
  `--retry-list`.
- **`analyze_stats.py`** — computes dataset / run statistics.
- **`undone.json`** — example outstanding-task list.
- **`api_cache/`** — on-disk cache of API responses keyed by `request_id`
  (populated when `--cache-dir` is set).

---

## Authors

- Ilia I. Okhotin
- Nikita O. Dorodnykh
- Aleksandr Yu. Yurin

## Citation

*A citation to the accompanying publication will be added here once available.*
