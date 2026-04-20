"""
Table header detection experiment — v6

Changes vs v5:
  FIX-1  Chunking threshold now row×col based (not rows-only).
         Fixes context_length_exceeded for wide tables like economy-table106 (95×27).
         New config: CHUNK_CELL_THRESHOLD (default 2000 cells = rows×cols).

  FIX-2  Per-prompt max_tokens override (MAX_TOKENS_BY_PROMPT dict).
         Reasoning prompts get more tokens; zero/fewshot get baseline.

  NEW-1  --model-alias CLI param. Run dir named run_{ts}_{alias}_{runid}.
         Avoids confusion when running multiple models in parallel.

  NEW-2  selected_tables.json saved at run start — lists every table used.
         --table-seed PATH loads a prior selected_tables.json so all models
         run on exactly the same tables.

  NEW-3  model_alias stored in every result record and checkpoint metadata.
"""

import os
import json
import re
import time
import logging
import hashlib
import asyncio
import argparse
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import httpx


# =========================
# CONFIG
# =========================
PROJECT_ROOT = Path(__file__).resolve().parent
PROMPTS_DIR  = PROJECT_ROOT / "prompts"

VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://127.0.0.1:8000/v1")
VLLM_API_KEY  = os.getenv("VLLM_API_KEY",  "EMPTY")
MODEL_NAME    = os.getenv("MODEL_NAME",    "Qwen/Qwen3-VL-30B-A3B-Thinking")
MODEL_ALIAS   = os.getenv("MODEL_ALIAS",   "")   # short name for run dir, e.g. "qwen30b"

OUTPUT_DIR   = os.getenv("OUTPUT_DIR",   "results")
LOG_LEVEL    = os.getenv("LOG_LEVEL",    "INFO")

MAX_RETRIES         = int(os.getenv("MAX_RETRIES",          "2"))
RETRY_BACKOFF_BASE  = float(os.getenv("RETRY_BACKOFF_BASE", "3.0"))
TEMPERATURE         = float(os.getenv("TEMPERATURE",        "0.0"))
MAX_TOKENS          = int(os.getenv("MAX_TOKENS",           "8192"))

# Per-prompt token overrides — reasoning prompts need more headroom.
# Keys are prompt stem names (without .txt). Missing keys use MAX_TOKENS.
MAX_TOKENS_BY_PROMPT: Dict[str, int] = {
    "reasoning_max":         12288,
    "reasoning_min":         12288,
    "reasoning_domain":      12288,
    "reasoning_few_domain":  12288,
    "fewshot_reasoning_max": 12288,
    "fewshot_reasoning_min": 12288,
}

CONCURRENCY         = int(os.getenv("CONCURRENCY",          "2"))
CHECKPOINT_EVERY    = int(os.getenv("CHECKPOINT_EVERY",     "10"))
REQUEST_TIMEOUT_SEC = float(os.getenv("REQUEST_TIMEOUT_SECONDS", "300.0"))
INTER_REQUEST_DELAY = float(os.getenv("INTER_REQUEST_DELAY",     "1.0"))
EARLY_STOP_FAILURES = int(os.getenv("EARLY_STOP_FAILURES",       "10"))

# FIX-1: chunk threshold based on total cells (rows × cols), not rows alone.
# economy-table106: 95 rows × 27 cols = 2565 cells → exceeds 2000 → chunked.
# A plain 100-row × 5-col table: 500 cells → not chunked. Correct behaviour.
CHUNK_CELL_THRESHOLD = int(os.getenv("CHUNK_CELL_THRESHOLD", "2000"))  # rows×cols
CHUNK_SIZE           = int(os.getenv("CHUNK_SIZE",           "80"))    # rows per chunk
CHUNK_OVERLAP        = int(os.getenv("CHUNK_OVERLAP",        "10"))    # row overlap

# Kept for backward compat / override; set to 0 to disable row-only check
CHUNK_ROW_THRESHOLD  = int(os.getenv("CHUNK_ROW_THRESHOLD",  "0"))

# Total tables and JSON:HTML ratio — overridable via CLI
TOTAL_TABLES  = int(os.getenv("TOTAL_TABLES",  "0"))   # 0 = use all available
FORMAT_RATIO  = os.getenv("FORMAT_RATIO", "50:50")      # "JSON:HTML"

# =========================
# EXPERIMENT PLAN
# =========================
EXPERIMENT_PLAN = [
    {
        # Domain prompts — tailored for biomedical PubTables content
        "name":      "pubtables_complex_top500",
        "json_root": PROJECT_ROOT / "Get_500_Tables_from_PubTables" / "JSON_Complex_TOP500_normalized",
        "html_root": PROJECT_ROOT / "Get_500_Tables_from_PubTables" / "JSON_Complex_TOP500_normalized_html",
        "limit":     500,
        "prompts":   ["zero_domain", "fewshot_domain", "reasoning_domain"],
    },
    {
        # MAX-strategy prompts — matches dataset annotation
        "name":      "maximum_viewpoint",
        "json_root": PROJECT_ROOT / "Convert_from_xlsx_to_Json" / "maximum_viewpoint_converted_json",
        "html_root": PROJECT_ROOT / "Convert_from_json_to_html" / "maximum_viewpoint_converted_html",
        "limit":     500,
        "prompts":   ["zero_max", "fewshot_max", "reasoning_max"],
    },
    {
        # MIN-strategy prompts — matches dataset annotation
        "name":      "table_normalization",
        "json_root": PROJECT_ROOT / "Convert_from_xlsx_to_Json" / "table_normalization_converted_json",
        "html_root": PROJECT_ROOT / "Convert_from_json_to_html" / "table_normalization_converted_html",
        "limit":     500,
        "prompts":   ["zero_min", "fewshot_min", "reasoning_min"],
    },
]

EXCLUDE_DIR_NAMES = {
    "results", "raw_responses", "test_responses",
    "__pycache__", ".git", ".venv", "venv", "_logs",
}
EXCLUDE_FILE_PREFIXES = (
    "responses_", "failed_", "checkpoint_",
    "summary_", "parsed_responses_",
)
EXCLUDE_PROMPT_FILES = {"sc_max.txt", "sc_min.txt", "system.txt"}
LABEL_KEYS = {
    "is_column_header", "is_projected_row_header", "is_metadata",
    "is_row_header", "is_spanning", "label", "labels",
    "gold", "answer", "answers", "target",
}


# =========================
# LOGGING
# =========================
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
log_level_value = getattr(logging, LOG_LEVEL.upper(), logging.INFO)
logging.basicConfig(
    level=log_level_value,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)


# =========================
# HELPERS
# =========================
def slugify(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9._-]+", "_", value)
    value = re.sub(r"_+", "_", value)
    return value.strip("_") or "item"


def stable_hash(text: str, length: int = 12) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:length]


def sanitize_for_prompt(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: sanitize_for_prompt(v) for k, v in obj.items()
                if k not in LABEL_KEYS and not k.startswith("is_")}
    if isinstance(obj, list):
        return [sanitize_for_prompt(x) for x in obj]
    return obj


def rows_bin(n: int) -> str:
    if n <= 10:  return "<=10"
    if n <= 25:  return "11-25"
    return ">25"


def count_bin(n: int) -> str:
    if n == 0:  return "0"
    if n <= 2:  return "1-2"
    if n <= 5:  return "3-5"
    return "6+"


def parse_format_ratio(ratio_str: str) -> Tuple[float, float]:
    try:
        j, h = ratio_str.strip().split(":")
        j, h = float(j), float(h)
        total = j + h
        if total <= 0:
            raise ValueError
        return j / total, h / total
    except Exception:
        raise ValueError(f"Invalid FORMAT_RATIO '{ratio_str}'. Use e.g. '50:50'.")


def coords_to_set(coords: List[Dict[str, int]]) -> set:
    out = set()
    for h in coords or []:
        try:
            out.add((int(h["row"]), int(h["col"])))
        except Exception:
            continue
    return out


def to_one_based_coords(coords: List[Dict[str, int]]) -> List[Dict[str, int]]:
    out = set()
    for h in coords:
        try:
            out.add((int(h["row"]) + 1, int(h["col"]) + 1))
        except Exception:
            continue
    return [{"row": r, "col": c} for r, c in sorted(out)]


def extract_true_coords_from_cells(
        cells: List[Dict[str, Any]]
) -> Tuple[List[Dict[str, int]], Dict[Tuple[int,int], str]]:
    """
    Returns:
      coords  — list of 0-based anchor dicts for header cells
      gt_text — dict mapping (row, col) → ground truth cell text
    """
    coords:  set = set()
    gt_text: Dict[Tuple[int,int], str] = {}
    for cell in cells or []:
        if (bool(cell.get("is_column_header"))
                or bool(cell.get("is_projected_row_header"))
                or bool(cell.get("is_spanning"))):
            row_nums = cell.get("row_nums", []) or []
            col_nums = cell.get("column_nums", []) or []
            if row_nums and col_nums:
                try:
                    anchor = (int(row_nums[0]), int(col_nums[0]))
                    coords.add(anchor)
                    gt_text[anchor] = str(
                        cell.get("xml_text_content") or ""
                    ).strip()
                except Exception:
                    continue
    coord_list = [{"row": r, "col": c} for r, c in sorted(coords)]
    return coord_list, gt_text


def extract_type_coords_from_cells(cells: List[Dict[str, Any]]) -> Dict[str, Any]:
    col_c, proj_c, span_c = set(), set(), set()
    col_n = proj_n = span_n = 0
    for cell in cells or []:
        row_nums = cell.get("row_nums", []) or []
        col_nums = cell.get("column_nums", []) or []
        if not row_nums or not col_nums:
            continue
        try:
            anchor = (int(row_nums[0]), int(col_nums[0]))
        except Exception:
            continue
        if cell.get("is_column_header"):
            col_c.add(anchor);  col_n  += 1
        if cell.get("is_projected_row_header"):
            proj_c.add(anchor); proj_n += 1
        if cell.get("is_spanning"):
            span_c.add(anchor); span_n += 1
    return {
        "column_headers":                   [{"row": r, "col": c} for r, c in sorted(col_c)],
        "projected_row_headers":            [{"row": r, "col": c} for r, c in sorted(proj_c)],
        "spanning":                         [{"row": r, "col": c} for r, c in sorted(span_c)],
        "column_header_cell_count":         col_n,
        "projected_row_header_cell_count":  proj_n,
        "spanning_cell_count":              span_n,
    }


def extract_true_coords_from_headers(
        headers: List[Any]
) -> Tuple[List[Dict[str, int]], Dict[Tuple[int,int], str]]:
    """
    Returns:
      coords  — list of 0-based anchor dicts
      gt_text — dict mapping (row, col) → text (empty for matrix format)
    """
    coords: set = set()
    gt_text: Dict[Tuple[int,int], str] = {}
    for h in headers or []:
        if isinstance(h, dict) and "row" in h and "col" in h:
            try:
                anchor = (int(h["row"]), int(h["col"]))
                coords.add(anchor)
                gt_text[anchor] = str(h.get("text", "")).strip()
            except Exception:
                continue
    coord_list = [{"row": r, "col": c} for r, c in sorted(coords)]
    return coord_list, gt_text


def evaluate_coord_sets(true_set: set, pred_set: set,
                         table_rows: int = 0, table_cols: int = 0) -> Dict[str, Any]:
    if table_rows > 0 and table_cols > 0:
        pred_set = {(r, c) for r, c in pred_set
                    if 0 <= r < table_rows and 0 <= c < table_cols}
    support = len(true_set); pred_count = len(pred_set)
    if support == 0 and pred_count == 0:
        return dict(support=0, pred_count=0, tp=0, fp=0, fn=0,
                    precision=1.0, recall=1.0, f1=1.0, jaccard=1.0,
                    exact_match=True, partial_match=True, header_coverage=1.0)
    tp    = len(true_set & pred_set)
    fp    = len(pred_set - true_set)
    fn    = len(true_set - pred_set)
    prec  = tp / pred_count if pred_count else 0.0
    rec   = tp / support    if support    else 0.0
    f1    = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    union = len(true_set | pred_set)
    jacc  = tp / union if union else 1.0
    exact   = true_set == pred_set
    partial = exact or (support > 0 and rec >= 0.5)
    return dict(support=support, pred_count=pred_count, tp=tp, fp=fp, fn=fn,
                precision=prec, recall=rec, f1=f1, jaccard=jacc,
                exact_match=exact, partial_match=partial, header_coverage=rec)

def token_f1(pred_text: str, true_text: str) -> float:
    """Token-level F1 between two strings (whitespace tokenisation)."""
    pred_tokens = pred_text.lower().split()
    true_tokens = true_text.lower().split()
    if not pred_tokens and not true_tokens:
        return 1.0
    if not pred_tokens or not true_tokens:
        return 0.0
    from collections import Counter
    pc = Counter(pred_tokens)
    tc = Counter(true_tokens)
    common = sum((pc & tc).values())
    if common == 0:
        return 0.0
    p = common / len(pred_tokens)
    r = common / len(true_tokens)
    return 2 * p * r / (p + r)


def evaluate_text_metrics(
    true_text_map: Dict[Tuple[int,int], str],   # (r,c) → gt text
    pred_headers:  List[Dict[str, Any]],         # parsed headers with optional "text"
    true_set:      set,                          # 0-based coord set (after OOB filter)
    pred_set_filtered: set,                      # 0-based coord set (after OOB filter)
) -> Dict[str, Any]:
    """
    Three text-aware metrics:
      text_exact_match_rate  — among TP coords, fraction where pred text == gt text
                               (case-insensitive, stripped)
      text_token_f1_mean     — mean token F1 across TP coords
      joint_f1               — coord F1 recomputed counting a cell correct only if
                               coord AND text both match (exact, case-insensitive)
    """
    if not true_text_map:
        return {
            "text_exact_match_rate": None,
            "text_token_f1_mean":    None,
            "joint_f1":              None,
            "joint_precision":       None,
            "joint_recall":          None,
        }

    # Build pred text map: coord → pred text (if available)
    pred_text_map: Dict[Tuple[int,int], str] = {}
    for h in pred_headers or []:
        try:
            key = (int(h["row"]), int(h["col"]))
            pred_text_map[key] = str(h.get("text", "")).strip()
        except Exception:
            continue

    # TP coords
    tp_coords = true_set & pred_set_filtered

    exact_matches: List[int] = []
    token_f1s:     List[float] = []
    joint_tp = 0

    for coord in tp_coords:
        gt   = true_text_map.get(coord, "").lower().strip()
        pred = pred_text_map.get(coord, "").lower().strip()
        exact = int(gt == pred)
        tf1   = token_f1(pred, gt)
        exact_matches.append(exact)
        token_f1s.append(tf1)
        if exact:
            joint_tp += 1

    n_tp = len(tp_coords)
    text_exact = sum(exact_matches) / n_tp if n_tp else None
    text_tf1   = sum(token_f1s)    / n_tp if n_tp else None

    # Joint F1: coord must match AND text exact match
    n_pred = len(pred_set_filtered)
    n_true = len(true_set)
    if n_pred == 0 and n_true == 0:
        joint_p = joint_r = joint_f = 1.0
    else:
        joint_p = joint_tp / n_pred if n_pred else 0.0
        joint_r = joint_tp / n_true if n_true else 0.0
        joint_f = (2 * joint_p * joint_r / (joint_p + joint_r)
                   if (joint_p + joint_r) else 0.0)

    return {
        "text_exact_match_rate": text_exact,
        "text_token_f1_mean":    text_tf1,
        "joint_f1":              joint_f,
        "joint_precision":       joint_p,
        "joint_recall":          joint_r,
    }




def dataframe_friendly(records: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for rec in records:
        row = dict(rec)
        for key in ["true_headers_raw", "true_headers_1based",
                    "true_headers_by_type_raw", "true_headers_text",
                    "parsed_headers"]:
            if key in row and isinstance(row[key], (list, dict)):
                row[key] = json.dumps(row[key], ensure_ascii=False)
        rows.append(row)
    return pd.DataFrame(rows)


def classify_api_error(msg: str) -> str:
    m = msg.lower()
    if any(x in m for x in ["maximum context length", "context length", "too many tokens"]):
        return "context_length_exceeded"
    if "timeout" in m or "timed out" in m:
        return "timeout"
    if "out of memory" in m or "oom" in m or "cuda" in m:
        return "oom"
    if "all connection attempts failed" in m or "connection refused" in m:
        return "connection_error"
    if "rate limit" in m or "429" in m:
        return "rate_limit"
    if "bad request" in m or "400" in m:
        return "bad_request"
    return "api_error"


def needs_chunking(table_rows: int, table_cols: int) -> bool:
    """
    FIX-1: Use cell count (rows×cols) as chunking trigger, not rows alone.
    Also respects legacy CHUNK_ROW_THRESHOLD if set > 0.
    """
    cell_count = table_rows * table_cols
    if cell_count > CHUNK_CELL_THRESHOLD:
        return True
    if CHUNK_ROW_THRESHOLD > 0 and table_rows > CHUNK_ROW_THRESHOLD:
        return True
    return False


def get_max_tokens_for_prompt(prompt_name: str) -> int:
    """FIX-2: Return per-prompt token limit, falling back to global MAX_TOKENS."""
    return MAX_TOKENS_BY_PROMPT.get(prompt_name, MAX_TOKENS)


# =========================
# HTML PREPROCESSING
# =========================
def strip_html_header_hints(html: str) -> str:
    html = re.sub(r"<th(\b[^>]*)>", r"<td\1>", html, flags=re.IGNORECASE)
    html = re.sub(r"</th>",          "</td>",    html, flags=re.IGNORECASE)
    html = re.sub(r'\s+(?:class|id|style)=["\'][^"\']*["\']', "", html, flags=re.IGNORECASE)
    return html


# =========================
# OUTPUT PARSER
# =========================
def parse_output(raw_text: str) -> Tuple[bool, List[Dict[str, Any]], str]:
    """
    Parse model output into a list of header dicts.

    Primary format (with text):
        0 1 | Treatment A
        1 0 | Characteristic

    Legacy fallbacks (coord-only): "row col", [[r,c]], (r,c)

    Each returned dict: {"row": int, "col": int, "text": str}
    "text" is empty string if not provided by the model.

    Thinking-block handling (Qwen3 and other reasoning models):
    - If </think> is present: parse ONLY the content after it.
      Prevents picking up coordinates from the reasoning chain,
      which are written without pipe separators and produce text="".
    - If truncated inside <think> (no </think>): return parse failure.
      Better an honest empty result than coords from incomplete reasoning.
    """
    if not raw_text or not str(raw_text).strip():
        return True, [], ""

    text = str(raw_text).strip()

    # Thinking-block extraction
    think_close = "</think>"
    if think_close in text:
        # Take only what comes after </think>
        text = text.split(think_close, 1)[1].strip()
    elif "<think>" in text:
        # Started thinking but truncated — no usable output
        return False, [], "truncated_inside_think_block"

    text = re.sub(r"^```[a-z]*\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$",        "", text, flags=re.IGNORECASE).strip()

    # Empty output after think block = model predicted no headers
    if not text:
        return True, [], ""

    seen   = set()
    coords = []

    # Primary: "row col | text" per line (pipe separator)
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if "|" in line:
            left, _, cell_text = line.partition("|")
            m = re.search(r"(\d+)\s+(\d+)", left)
            if m:
                r, c = int(m.group(1)), int(m.group(2))
                if (r, c) not in seen:
                    seen.add((r, c))
                    coords.append({"row": r, "col": c,
                                   "text": cell_text.strip()})
        else:
            m = re.fullmatch(r"(\d+)\s+(\d+)", line)
            if m:
                r, c = int(m.group(1)), int(m.group(2))
                if (r, c) not in seen:
                    seen.add((r, c))
                    coords.append({"row": r, "col": c, "text": ""})

    if coords:
        coords.sort(key=lambda x: (x["row"], x["col"]))
        return True, coords, ""

    # Fallback A: loose "int ... int" per line (no pipe, no text)
    for line in text.splitlines():
        m = re.search(r"(\d+)\D+(\d+)", line.strip())
        if m:
            r, c = int(m.group(1)), int(m.group(2))
            if (r, c) not in seen:
                seen.add((r, c))
                coords.append({"row": r, "col": c, "text": ""})
    if coords:
        coords.sort(key=lambda x: (x["row"], x["col"]))
        return True, coords, "fallback_loose_lines"

    # Fallback B: [[row,col],...] JSON array
    pairs = re.findall(r"\[\s*(\d+)\s*,\s*(\d+)\s*\]", text)
    if pairs:
        for rs, cs in pairs:
            r, c = int(rs), int(cs)
            if (r, c) not in seen:
                seen.add((r, c))
                coords.append({"row": r, "col": c, "text": ""})
        coords.sort(key=lambda x: (x["row"], x["col"]))
        return True, coords, "fallback_json_array"

    # Fallback C: (row,col) paren format
    parens = re.findall(r"\(\s*(\d+)\s*,\s*(\d+)\s*\)", text)
    if parens:
        for rs, cs in parens:
            r, c = int(rs), int(cs)
            if (r, c) not in seen:
                seen.add((r, c))
                coords.append({"row": r, "col": c, "text": ""})
        coords.sort(key=lambda x: (x["row"], x["col"]))
        return True, coords, "fallback_paren_format"

    return False, [], "no_parseable_coordinates"


# =========================
# CHUNKING
# =========================
def chunk_cells_table(obj: Dict, s: int, e: int) -> Tuple[Dict, int]:
    out_cells = []
    for cell in obj.get("cells", []):
        rows  = cell.get("row_nums", []) or []
        in_c  = [r for r in rows if s <= r < e]
        if not in_c:
            continue
        nc = {k: v for k, v in cell.items() if k != "row_nums"}
        nc["row_nums"]    = [r - s for r in in_c]
        nc["column_nums"] = cell.get("column_nums", []) or []
        out_cells.append(nc)
    result = {k: v for k, v in obj.items() if k != "cells"}
    result["cells"] = out_cells
    return result, s


def chunk_matrix_table(obj: Dict, s: int, e: int) -> Tuple[Dict, int]:
    result = {k: v for k, v in obj.items() if k != "data"}
    result["data"] = (obj.get("data") or [])[s:e]
    return result, s


def make_chunks(table_json: str, table_rows: int,
                table_kind: str) -> List[Tuple[str, int]]:
    try:
        obj = json.loads(table_json)
    except Exception:
        return [(table_json, 0)]

    chunks = []
    start  = 0
    while start < table_rows:
        end = min(start + CHUNK_SIZE, table_rows)
        try:
            if table_kind == "cells":
                co, off = chunk_cells_table(obj, start, end)
            elif table_kind == "matrix":
                co, off = chunk_matrix_table(obj, start, end)
            else:
                co, off = (chunk_cells_table(obj, start, end)
                           if "cells" in obj
                           else chunk_matrix_table(obj, start, end))
            repr_str = json.dumps(sanitize_for_prompt(co),
                                  ensure_ascii=False, separators=(",", ":"))
            chunks.append((repr_str, off))
        except Exception as ex:
            logging.warning(f"Chunk error start={start}: {ex}")
            chunks.append((table_json, 0))
            break
        if end >= table_rows:
            break
        start = start + CHUNK_SIZE - CHUNK_OVERLAP

    return chunks or [(table_json, 0)]


def merge_chunk_predictions(chunk_results: List[Tuple[List[Dict], int]]) -> List[Dict[str, int]]:
    seen   = set()
    merged = []
    for headers, row_offset in chunk_results:
        for h in headers:
            key = (h["row"] + row_offset, h["col"])
            if key not in seen:
                seen.add(key)
                merged.append({"row": key[0], "col": key[1]})
    merged.sort(key=lambda x: (x["row"], x["col"]))
    return merged


# =========================
# ASYNC API CALL
# =========================
async def async_api_call(
    client:   httpx.AsyncClient,
    model:    str,
    messages: List[Dict[str, str]],
    max_tokens: int = MAX_TOKENS,          # FIX-2: per-call token limit
) -> Dict[str, Any]:
    url     = f"{VLLM_BASE_URL}/chat/completions"
    payload = {
        "model":       model,
        "messages":    messages,
        "temperature": TEMPERATURE,
        "max_tokens":  max_tokens,
        # No guided decoding — free output, parser handles any format
    }
    hdrs = {
        "Content-Type":  "application/json",
        "Authorization": f"Bearer {VLLM_API_KEY}",
    }

    last_error = ""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            t0   = time.time()
            resp = await client.post(url, json=payload, headers=hdrs)

            if resp.status_code >= 400:
                try:
                    eb  = resp.json()
                    em  = eb.get("message") or eb.get("detail") or resp.text[:300]
                except Exception:
                    em = resp.text[:300]
                raise RuntimeError(f"HTTP {resp.status_code}: {em}")

            data_r   = resp.json()
            duration = time.time() - t0
            raw      = data_r["choices"][0]["message"]["content"] or ""

            ok, parsed_hdrs, pe = parse_output(raw)
            usage = data_r.get("usage") or {}

            return {
                "api_success":    True,
                "raw_response":   raw,
                "parse_success":  ok,
                "parsed_headers": parsed_hdrs,
                "parse_error":    pe,
                "duration_sec":   duration,
                "retry_attempts": attempt,
                "max_tokens_used": max_tokens,
                "tokens_used": {
                    "prompt":     usage.get("prompt_tokens"),
                    "completion": usage.get("completion_tokens"),
                    "total":      usage.get("total_tokens"),
                } if usage else None,
                "error_type":    "",
                "error_message": "",
            }

        except Exception as e:
            last_error = str(e)
            logging.warning(f"Attempt {attempt}/{MAX_RETRIES} failed: {last_error[:200]}")
            if attempt < MAX_RETRIES:
                await asyncio.sleep(min(RETRY_BACKOFF_BASE ** attempt, 60.0))

    return {
        "api_success": False, "raw_response": "", "parse_success": False,
        "parsed_headers": [], "parse_error": "", "duration_sec": None,
        "retry_attempts": MAX_RETRIES, "max_tokens_used": max_tokens,
        "tokens_used": None,
        "error_type":    classify_api_error(last_error),
        "error_message": last_error,
    }


# =========================
# MAIN CLASS
# =========================
class ResponseCollector:
    def __init__(self, output_dir: str = OUTPUT_DIR,
                 total_tables: int = TOTAL_TABLES,
                 format_ratio: str = FORMAT_RATIO,
                 model_alias: str = MODEL_ALIAS,
                 table_seed_path: Optional[str] = None):

        self.total_tables  = total_tables
        self.json_frac, self.html_frac = parse_format_ratio(format_ratio)
        self.model_alias   = model_alias or slugify(MODEL_NAME.split("/")[-1])[:12]
        self.table_seed_path = table_seed_path

        ts_short = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self.run_id   = f"{ts_short}_{self.model_alias}"
        self.base_dir = Path(output_dir)
        self.run_dir  = self.base_dir / f"run_{self.run_id}"

        self.logs_dir    = self.run_dir / "logs"
        self.results_dir = self.run_dir / "results"
        self.metrics_dir = self.run_dir / "metrics"
        self.ckpt_dir    = self.run_dir / "checkpoints"

        for d in [self.logs_dir, self.results_dir, self.metrics_dir, self.ckpt_dir]:
            d.mkdir(parents=True, exist_ok=True)

        self._setup_file_logging()

        self.system_prompt = self._load_system_prompt()
        self.prompts       = self._load_prompt_configs()
        self.table_map: Dict[str, Dict[str, List[Dict[str, Any]]]] = self._load_all_tables()

        self.responses:             List[Dict[str, Any]] = []
        self.valid_responses:       List[Dict[str, Any]] = []
        self.api_failed_requests:   List[Dict[str, Any]] = []
        self.parse_failed_requests: List[Dict[str, Any]] = []

        self.start_time:      Optional[datetime] = None
        self.completed_count: int  = 0
        self._consec_fail:    int  = 0
        self._abort:          bool = False
        self._lock = asyncio.Lock()

        logging.info(
            f"Run dir: {self.run_dir}\n"
            f"  Model: {MODEL_NAME} (alias: {self.model_alias})\n"
            f"  max_tokens={MAX_TOKENS} overrides={MAX_TOKENS_BY_PROMPT}\n"
            f"  concurrency={CONCURRENCY} timeout={REQUEST_TIMEOUT_SEC}s\n"
            f"  total_tables={self.total_tables or 'all'} "
            f"format_ratio=json:{self.json_frac:.0%} html:{self.html_frac:.0%}\n"
            f"  chunk_cell_threshold={CHUNK_CELL_THRESHOLD} "
            f"chunk_row_threshold={CHUNK_ROW_THRESHOLD}"
        )

    # ---------- setup ----------

    def _setup_file_logging(self):
        log_file = self.logs_dir / f"experiment_{self.run_id}.log"
        root = logging.getLogger()
        for h in list(root.handlers):
            if isinstance(h, logging.FileHandler):
                root.removeHandler(h); h.close()
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(log_level_value)
        fh.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        root.addHandler(fh)

    async def _check_server(self) -> bool:
        base = VLLM_BASE_URL.rstrip("/").removesuffix("/v1")
        candidates = [
            f"{base}/health",
            f"{base}/ping",
            f"{VLLM_BASE_URL}/models",
        ]
        async with httpx.AsyncClient(timeout=10.0) as c:
            for url in candidates:
                try:
                    r = await c.get(url)
                    if r.status_code in (200, 405):
                        logging.info(f"Server reachable via {url} (status {r.status_code})")
                        return True
                except Exception:
                    continue
        logging.critical(f"Server not reachable. Tried: {candidates}")
        return False

    def _load_system_prompt(self) -> str:
        path = PROMPTS_DIR / "system.txt"
        if not path.exists():
            raise FileNotFoundError(f"System prompt not found: {path}")
        return path.read_text(encoding="utf-8").strip()

    def _load_prompt_configs(self) -> List[Dict[str, Any]]:
        files = [p for p in sorted(PROMPTS_DIR.glob("*.txt"), key=lambda p: p.name)
                 if p.name not in EXCLUDE_PROMPT_FILES]
        if not files:
            raise FileNotFoundError(f"No prompt .txt files in {PROMPTS_DIR}")
        return [{"name": p.stem, "file": str(p), "user": p.read_text(encoding="utf-8").strip()}
                for p in files if p.read_text(encoding="utf-8").strip()]

    # ---------- table loading ----------

    def _should_skip_path(self, path: Path) -> bool:
        return (any(part in EXCLUDE_DIR_NAMES for part in path.parts)
                or path.name.startswith(EXCLUDE_FILE_PREFIXES))

    def _iter_json_files(self, root: Path):
        seen = set()
        for path in sorted(root.rglob("*.json"), key=str):
            if self._should_skip_path(path): continue
            r = str(path.resolve())
            if r not in seen:
                seen.add(r); yield path

    def _table_dims_cells(self, cells) -> Tuple[int, int]:
        rows, cols = set(), set()
        for cell in cells or []:
            for r in cell.get("row_nums", []) or []:
                try: rows.add(int(r))
                except: pass
            for c in cell.get("column_nums", []) or []:
                try: cols.add(int(c))
                except: pass
        return (max(rows) + 1 if rows else 0, max(cols) + 1 if cols else 0)

    def _table_dims_matrix(self, data) -> Tuple[int, int]:
        if not isinstance(data, list): return 0, 0
        return len(data), max((len(r) for r in data if isinstance(r, list)), default=0)

    def _make_table_record(self, filepath: Path, item: Dict,
                            idx: int, source_name: str) -> Optional[Dict[str, Any]]:
        if not isinstance(item, dict): return None

        prompt_obj = sanitize_for_prompt(item)
        table_json = json.dumps(prompt_obj, ensure_ascii=False, separators=(",", ":"))
        table_hash = stable_hash(table_json, 12)

        if "cells" in item:
            ti            = extract_type_coords_from_cells(item.get("cells", []))
            true_raw, gt_text = extract_true_coords_from_cells(item.get("cells", []))
            kind          = "cells"
            nr, nc        = self._table_dims_cells(item.get("cells", []))
            has_ti        = True
        elif "data" in item:
            true_raw, gt_text = extract_true_coords_from_headers(item.get("headers", []))
            kind          = "matrix"
            nr, nc        = self._table_dims_matrix(item.get("data", []))
            has_ti        = False
            ti = {"column_headers": [], "projected_row_headers": [], "spanning": [],
                  "column_header_cell_count": 0, "projected_row_header_cell_count": 0,
                  "spanning_cell_count": 0}
        else:
            return None

        th_by_type = {k: ti[k] for k in
                      ["column_headers", "projected_row_headers", "spanning"]}

        return {
            "source_group":   source_name,
            "source_file":    str(filepath),
            "source_stem":    filepath.stem,
            "table_index":    idx,
            "table_kind":     kind,
            "table_rows":     nr,
            "table_cols":     nc,
            "table_rows_bin": rows_bin(nr),
            "table_hash":     table_hash,
            "table_json":     table_json,
            "table_html":     "",
            "true_headers_raw":         true_raw,
            "true_headers_1based":      to_one_based_coords(true_raw),
            "true_headers_count":       len(true_raw),
            "true_headers_count_bin":   count_bin(len(true_raw)),
            "true_headers_by_type_raw": th_by_type,
            # gt_text: (row,col) → ground truth cell text (str keys for JSON compat)
            "true_headers_text":        {f"{r},{c}": t
                                         for (r, c), t in gt_text.items()},
            "has_type_info":            has_ti,
            "column_header_cell_count":         ti["column_header_cell_count"],
            "projected_row_header_cell_count":  ti["projected_row_header_cell_count"],
            "spanning_cell_count":              ti["spanning_cell_count"],
            "spanning_cell_count_bin":          count_bin(ti["spanning_cell_count"]),
        }

    def _load_json_records(self, json_root: Path, source_name: str,
                            limit: int) -> List[Dict[str, Any]]:
        if not json_root.exists():
            raise FileNotFoundError(f"JSON root does not exist: {json_root}")
        records = []
        for fp in self._iter_json_files(json_root):
            try:
                with open(fp, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                items = []
                if isinstance(raw, dict):
                    items = [raw]
                elif isinstance(raw, list):
                    items = [x for x in raw if isinstance(x, dict)
                             and ("cells" in x or "data" in x)]
                    if not items:
                        items = [x for x in raw if isinstance(x, dict)]
                for idx, item in enumerate(items):
                    rec = self._make_table_record(fp, item, idx, source_name)
                    if rec:
                        records.append(rec)
                if len(records) >= limit:
                    break
            except json.JSONDecodeError as e:
                logging.error(f"JSON parse error {fp}: {e}")
            except Exception as e:
                logging.error(f"Error reading {fp}: {e}")
        return records[:limit]

    def _attach_html(self, records: List[Dict[str, Any]],
                     html_root: Path) -> List[Dict[str, Any]]:
        if not html_root.exists():
            logging.warning(f"HTML root not found: {html_root}. HTML format unavailable.")
            return records
        html_files: Dict[str, Path] = {}
        for hp in sorted(html_root.rglob("*.html"), key=str):
            html_files[hp.stem] = hp
        attached = 0
        for rec in records:
            stem = rec["source_stem"]
            if stem in html_files:
                try:
                    raw_html = html_files[stem].read_text(encoding="utf-8")
                    rec["table_html"] = strip_html_header_hints(raw_html)
                    attached += 1
                except Exception as e:
                    logging.warning(f"Could not read HTML for {stem}: {e}")
        logging.info(f"  HTML attached: {attached}/{len(records)} records")
        return records

    def _sample_tables(self, records: List[Dict[str, Any]],
                        n_total: int) -> Tuple[List[Dict], List[Dict]]:
        """
        Return the same N tables for both JSON and HTML formats.
        JSON sample = first n_total records.
        HTML sample = same records that have table_html loaded.
        Both lists refer to the same underlying table set.
        """
        selected    = records[:n_total]
        json_sample = selected
        html_sample = [r for r in selected if r.get("table_html")]
        return json_sample, html_sample

    def _load_all_tables(self) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
        # Load seed if provided — ensures identical table selection across models
        seed: Optional[Dict] = None
        if self.table_seed_path:
            with open(self.table_seed_path, "r", encoding="utf-8") as f:
                seed = json.load(f)
            logging.info(f"Loaded table seed from {self.table_seed_path}")

        raw_map: Dict[str, List[Dict[str, Any]]] = {}
        for src in EXPERIMENT_PLAN:
            records = self._load_json_records(
                src["json_root"], src["name"], int(src["limit"])
            )
            records = self._attach_html(records, src["html_root"])
            raw_map[src["name"]] = records
            logging.info(f"Loaded {len(records)} records from {src['name']}")

        if self.total_tables > 0:
            n_sources = len(EXPERIMENT_PLAN)
            per_source = self.total_tables // n_sources
            remainder  = self.total_tables - per_source * n_sources
        else:
            per_source = None
            remainder  = 0

        table_map: Dict[str, Dict[str, List[Dict]]] = {}
        total_json = total_html = 0
        seed_data: Dict[str, Dict[str, List[str]]] = {}  # NEW-2: for selected_tables.json

        for i, src in enumerate(EXPERIMENT_PLAN):
            records = raw_map[src["name"]]

            if seed:
                # Restrict to stems listed in seed file
                allowed_json = set(seed.get(src["name"], {}).get("json", []))
                allowed_html = set(seed.get(src["name"], {}).get("html", []))
                json_sample = [r for r in records if r["source_stem"] in allowed_json]
                html_sample = [r for r in records
                               if r.get("table_html") and r["source_stem"] in allowed_html]
            else:
                if per_source is not None:
                    n_src = per_source + (1 if i < remainder else 0)
                    records = records[:n_src]
                # Same tables for both formats — format_ratio controls only
                # whether HTML tasks are generated, not which tables are used.
                # json_frac=1.0 → JSON only; html_frac=1.0 → HTML only; 50:50 → both.
                if self.html_frac == 0:
                    json_sample = records
                    html_sample = []
                elif self.json_frac == 0:
                    json_sample = []
                    html_sample = [r for r in records if r.get("table_html")]
                else:
                    # Both formats — same table set
                    json_sample, html_sample = self._sample_tables(records, len(records))

            table_map[src["name"]] = {"json": json_sample, "html": html_sample}
            seed_data[src["name"]] = {
                "json": [r["source_stem"] for r in json_sample],
                "html": [r["source_stem"] for r in html_sample],
            }
            total_json += len(json_sample)
            total_html += len(html_sample)
            logging.info(f"  {src['name']}: json={len(json_sample)} html={len(html_sample)}")

        logging.info(f"Total: json={total_json} html={total_html} sum={total_json+total_html}")

        # NEW-2: Save selected_tables.json at run start
        seed_out = self.run_dir / "selected_tables.json"
        with open(seed_out, "w", encoding="utf-8") as f:
            json.dump({
                "model":        MODEL_NAME,
                "model_alias":  self.model_alias,
                "total_tables": self.total_tables,
                "format_ratio": f"json:{self.json_frac:.0%} html:{self.html_frac:.0%}",
                "sources":      seed_data,
            }, f, ensure_ascii=False, indent=2)
        logging.info(f"Table selection saved: {seed_out}")

        return table_map

    # ---------- request building ----------

    def _prepare_messages(self, prompt_config: Dict, table_repr: str,
                           table_format: str,
                           chunk_info: str = "") -> List[Dict[str, str]]:
        tpl = str(prompt_config.get("user", ""))
        up  = (tpl
               .replace("{table_json}", table_repr)
               .replace("{table_html}", table_repr)
               .replace("{table_text}", table_repr)
               .replace("{table}",      table_repr))
        if up == tpl:
            label = "HTML TABLE" if table_format == "html" else "TABLE (JSON)"
            up = f"{tpl}\n\n{label}:\n{table_repr}"
        if chunk_info:
            up += f"\n\n[NOTE: {chunk_info}]"
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user",   "content": up},
        ]

    def _build_request_id(self, prompt_name: str, tr: Dict,
                           table_format: str) -> str:
        base = (f"{prompt_name}__{tr['source_group']}__{tr['source_stem']}"
                f"__t{tr['table_index']}__{tr['table_hash']}__{table_format}"
                f"__{self.model_alias}")
        return slugify(base)

    # ---------- result assembly ----------

    def _make_result(self, prompt_idx: int, prompt_config: Dict,
                     table_record: Dict, api_result: Dict,
                     table_format: str,
                     chunked: bool = False, n_chunks: int = 1) -> Dict[str, Any]:
        pname    = str(prompt_config.get("name", f"prompt_{prompt_idx}"))
        tr       = table_record
        true_set = coords_to_set(tr["true_headers_raw"])
        pred_set = (coords_to_set(api_result["parsed_headers"])
                    if api_result["api_success"] and api_result["parse_success"]
                    else set())

        overall = evaluate_coord_sets(
            true_set, pred_set,
            table_rows=tr["table_rows"], table_cols=tr["table_cols"]
        )

        # Filtered pred_set (OOB already removed inside evaluate_coord_sets)
        nr, nc = tr["table_rows"], tr["table_cols"]
        pred_set_f = {(r, c) for r, c in pred_set
                      if 0 <= r < nr and 0 <= c < nc}

        # Ground truth text map: convert str-key dict back to tuple keys
        raw_gt = tr.get("true_headers_text", {})
        gt_text_map: Dict[Tuple[int,int], str] = {
            (int(k.split(",")[0]), int(k.split(",")[1])): v
            for k, v in raw_gt.items()
        }

        text_metrics = evaluate_text_metrics(
            gt_text_map,
            api_result.get("parsed_headers", []),
            true_set,
            pred_set_f,
        )

        tu  = api_result.get("tokens_used") or {}
        ct  = tu.get("completion")
        tef = (ct / overall["f1"]) if (ct and overall["f1"] > 0) else None
        mt  = api_result.get("max_tokens_used", MAX_TOKENS)
        capped = (ct is not None and ct >= mt)

        type_metrics: Dict[str, Any] = {}
        for tname in ["column_headers", "projected_row_headers", "spanning"]:
            if tr.get("has_type_info"):
                ts = coords_to_set(tr["true_headers_by_type_raw"].get(tname, []))
                for k, v in evaluate_coord_sets(
                        ts, pred_set,
                        table_rows=tr["table_rows"], table_cols=tr["table_cols"]
                ).items():
                    type_metrics[f"{tname}_{k}"] = v
            else:
                for k in ["support","pred_count","tp","fp","fn","precision","recall",
                          "f1","jaccard","exact_match","partial_match","header_coverage"]:
                    type_metrics[f"{tname}_{k}"] = None

        result = {
            "request_id":    self._build_request_id(pname, tr, table_format),
            "timestamp":     datetime.now().isoformat(),
            "model":         MODEL_NAME,
            "model_alias":   self.model_alias,       # NEW-1
            "table_format":  table_format,
            "prompt_idx":    prompt_idx,
            "prompt_name":   pname,
            "prompt_file":   prompt_config.get("file", ""),
            "source_group":  tr["source_group"],
            "source_file":   tr["source_file"],
            "source_stem":   tr["source_stem"],
            "table_index":   tr["table_index"],
            "table_kind":    tr["table_kind"],
            "table_rows":    tr["table_rows"],
            "table_cols":    tr["table_cols"],
            "table_rows_bin":tr["table_rows_bin"],
            "table_hash":    tr["table_hash"],
            "true_headers_raw":          tr["true_headers_raw"],
            "true_headers_1based":       tr["true_headers_1based"],
            "true_headers_count":        tr["true_headers_count"],
            "true_headers_count_bin":    tr["true_headers_count_bin"],
            "true_headers_by_type_raw":  tr["true_headers_by_type_raw"],
            "has_type_info":             tr["has_type_info"],
            "column_header_cell_count":          tr["column_header_cell_count"],
            "projected_row_header_cell_count":   tr["projected_row_header_cell_count"],
            "spanning_cell_count":               tr["spanning_cell_count"],
            "spanning_cell_count_bin":           tr["spanning_cell_count_bin"],
            "chunked":           chunked,
            "n_chunks":          n_chunks,
            "api_success":       api_result["api_success"],
            "parse_success":     api_result["parse_success"],
            "status": ("api_failed"  if not api_result["api_success"]
                       else ("ok"    if api_result["parse_success"] else "parse_failed")),
            "raw_response":      api_result["raw_response"],
            "parsed_headers":    api_result["parsed_headers"],
            "parse_error":       api_result["parse_error"],
            "error_type":        api_result["error_type"],
            "error_message":     api_result["error_message"],
            "duration_sec":      api_result["duration_sec"],
            "retry_attempts":    api_result["retry_attempts"],
            "max_tokens_used":   mt,
            "completion_capped": capped,             # FIX-2: flag for truncation
            "tokens_used":       api_result.get("tokens_used"),
            "prompt_tokens":     tu.get("prompt"),
            "completion_tokens": ct,
            "total_tokens":      tu.get("total"),
            "token_efficiency":  tef,
            "system_prompt_file":"prompts/system.txt",
        }
        result.update(overall)
        result.update(type_metrics)
        result.update(text_metrics)   # text_exact_match_rate, text_token_f1_mean, joint_f1, …
        return result

    # ---------- async workers ----------

    async def _call_one(self, client: httpx.AsyncClient,
                         semaphore: asyncio.Semaphore,
                         messages: List[Dict],
                         max_tokens: int = MAX_TOKENS) -> Dict[str, Any]:
        async with semaphore:
            result = await async_api_call(client, MODEL_NAME, messages, max_tokens)
            if INTER_REQUEST_DELAY > 0:
                await asyncio.sleep(INTER_REQUEST_DELAY)
        return result

    async def _process_one(self, client: httpx.AsyncClient,
                            semaphore: asyncio.Semaphore,
                            prompt_idx: int, prompt_config: Dict,
                            table_record: Dict, table_format: str) -> Dict[str, Any]:
        if self._abort:
            return self._make_result(prompt_idx, prompt_config, table_record, {
                "api_success": False, "parse_success": False, "parsed_headers": [],
                "parse_error": "", "raw_response": "", "duration_sec": None,
                "retry_attempts": 0, "max_tokens_used": MAX_TOKENS,
                "tokens_used": None,
                "error_type": "aborted", "error_message": "early stop",
            }, table_format)

        pname      = prompt_config.get("name", "")
        mt         = get_max_tokens_for_prompt(pname)   # FIX-2
        table_repr = (table_record.get("table_html") or table_record["table_json"]
                      if table_format == "html" else table_record["table_json"])

        # FIX-1: cell-count based chunking
        do_chunk = needs_chunking(table_record["table_rows"], table_record["table_cols"])

        if not do_chunk:
            msgs = self._prepare_messages(prompt_config, table_repr, table_format)
            ar   = await self._call_one(client, semaphore, msgs, mt)
            return self._make_result(prompt_idx, prompt_config, table_record,
                                     ar, table_format)

        # HTML: send full file, no JSON chunking
        if table_format == "html":
            msgs = self._prepare_messages(prompt_config, table_repr, table_format)
            ar   = await self._call_one(client, semaphore, msgs, mt)
            return self._make_result(prompt_idx, prompt_config, table_record,
                                     ar, table_format)

        # JSON chunked
        chunks = make_chunks(table_record["table_json"],
                             table_record["table_rows"],
                             table_record["table_kind"])
        logging.info(
            f"Chunking {table_record['source_stem']} "
            f"({table_record['table_rows']}×{table_record['table_cols']}"
            f" = {table_record['table_rows']*table_record['table_cols']} cells)"
            f" → {len(chunks)} chunks"
        )

        chunk_results: List[Tuple[List[Dict], int]] = []
        total_dur = total_pt = total_ct = total_ret = 0
        any_fail  = False
        raw_parts = []

        for ci, (chunk_repr, offset) in enumerate(chunks):
            if self._abort:
                any_fail = True; break
            info = (f"Chunk {ci+1}/{len(chunks)} of a large table. "
                    f"Row 0 in this chunk = row {offset} in the full table (0-based).")
            msgs = self._prepare_messages(prompt_config, chunk_repr, "json", info)
            ar   = await self._call_one(client, semaphore, msgs, mt)

            if ar["api_success"] and ar["parse_success"]:
                chunk_results.append((ar["parsed_headers"], offset))
            elif not ar["api_success"]:
                any_fail = True
                logging.warning(
                    f"Chunk {ci+1}/{len(chunks)} of {table_record['source_stem']} "
                    f"failed: {ar['error_message'][:80]}")

            total_dur += ar.get("duration_sec") or 0
            total_ret += ar.get("retry_attempts", 1)
            raw_parts.append(ar.get("raw_response", ""))
            tu2 = ar.get("tokens_used") or {}
            total_pt += tu2.get("prompt", 0) or 0
            total_ct += tu2.get("completion", 0) or 0

        merged   = merge_chunk_predictions(chunk_results)
        combined = {
            "api_success":    not any_fail or bool(chunk_results),
            "parse_success":  bool(merged) or not any_fail,
            "parsed_headers": merged,
            "parse_error":    "" if chunk_results else "all_chunks_failed",
            "raw_response":   " | ".join(r for r in raw_parts if r)[:500],
            "error_type":     ("partial_chunk_failure" if any_fail and chunk_results
                               else ("api_error" if any_fail else "")),
            "error_message":  (f"{len(chunks)-len(chunk_results)}/{len(chunks)} chunks failed"
                               if any_fail else ""),
            "duration_sec":   total_dur,
            "retry_attempts": total_ret,
            "max_tokens_used": mt,
            "tokens_used":    {"prompt": total_pt, "completion": total_ct,
                               "total":  total_pt + total_ct},
        }
        return self._make_result(prompt_idx, prompt_config, table_record,
                                 combined, table_format,
                                 chunked=True, n_chunks=len(chunks))

    def _register_result(self, result: Dict[str, Any]):
        if result["api_success"]:
            self.responses.append(result)
            (self.valid_responses if result["parse_success"]
             else self.parse_failed_requests).append(result)
        else:
            self.api_failed_requests.append(result)
        self.completed_count += 1

    # ---------- producer-consumer queue ----------

    async def _run_tasks(self, tasks: List[Tuple], timestamp: str):
        total = len(tasks)
        sem   = asyncio.Semaphore(CONCURRENCY)
        since = 0
        queue = asyncio.Queue()
        for t in tasks:
            await queue.put(t)

        limits = httpx.Limits(
            max_connections=CONCURRENCY + 2,
            max_keepalive_connections=CONCURRENCY,
        )

        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT_SEC, limits=limits) as client:

            async def worker():
                nonlocal since
                while True:
                    try:
                        item = queue.get_nowait()
                    except asyncio.QueueEmpty:
                        return
                    pi, pc, tr, fmt = item
                    result = await self._process_one(client, sem, pi, pc, tr, fmt)

                    async with self._lock:
                        self._register_result(result)
                        since += 1

                        if result.get("error_type") == "connection_error" \
                                or result.get("error_message") == "early stop":
                            self._consec_fail += 1
                        else:
                            self._consec_fail = 0

                        if self._consec_fail >= EARLY_STOP_FAILURES and not self._abort:
                            self._abort = True
                            logging.critical(
                                f"EARLY STOP: {self._consec_fail} consecutive "
                                f"connection failures. Saving checkpoint.")
                            self._save_checkpoint(timestamp)

                        dur_s  = (f"{result['duration_sec']:.1f}s"
                                  if result.get("duration_sec") else "n/a")
                        c_str  = (f" [×{result.get('n_chunks',1)}ch]"
                                  if result.get("chunked") else "")
                        f_str  = f" [{result.get('table_format','?')}]"
                        f1_s   = (f" F1={result.get('f1',0):.3f}"
                                  if result.get("f1") is not None else "")
                        cap_s  = " [CAP]" if result.get("completion_capped") else ""
                        mt_s   = f" mt={result.get('max_tokens_used',MAX_TOKENS)}"
                        logging.info(
                            f"[{self.completed_count}/{total}] "
                            f"{result['prompt_name']}{f_str} | "
                            f"{result['source_stem']}{c_str} | "
                            f"status={result['status']}{f1_s}{cap_s}"
                            f"{mt_s} | dur={dur_s}"
                        )

                        if since >= CHECKPOINT_EVERY:
                            self._save_checkpoint(timestamp)
                            since = 0

                    queue.task_done()

            workers = [asyncio.ensure_future(worker())
                       for _ in range(CONCURRENCY)]
            await queue.join()
            for w in workers:
                w.cancel()

        self._save_checkpoint(timestamp)

    # ---------- main run ----------

    async def _run_async(self):
        self.start_time = datetime.now()
        ts = self.start_time.strftime("%Y%m%d_%H%M%S")

        if not await self._check_server():
            logging.critical("Aborting: server not healthy."); return

        tasks = []
        for src in EXPERIMENT_PLAN:
            allowed = src["prompts"]
            for fmt in ("json", "html"):
                records = self.table_map[src["name"]][fmt]
                for pi, pc in enumerate(self.prompts):
                    if allowed is None or pc["name"] in allowed:
                        for tr in records:
                            tasks.append((pi, pc, tr, fmt))

        total = len(tasks)
        for src in EXPERIMENT_PLAN:
            for fmt in ("json", "html"):
                recs    = self.table_map[src["name"]][fmt]
                allowed = src["prompts"]
                np = (len(self.prompts) if allowed is None
                      else len([p for p in self.prompts if p["name"] in allowed]))
                logging.info(f"  {src['name']} [{fmt}]: "
                             f"{np} prompts × {len(recs)} tables = {np*len(recs)}")
        logging.info(
            f"Total {total} tasks | CONCURRENCY={CONCURRENCY} "
            f"MAX_TOKENS={MAX_TOKENS} overrides={MAX_TOKENS_BY_PROMPT} "
            f"CHUNK_CELL_THRESHOLD={CHUNK_CELL_THRESHOLD} "
            f"early_stop={EARLY_STOP_FAILURES}"
        )

        await self._run_tasks(tasks, ts)
        self._save_final_results(ts)
        self._build_metrics_artifacts(ts)

    def run(self):
        asyncio.run(self._run_async())

    # ---------- retry mode ----------

    async def _run_retry_async(self, checkpoint_path: str):
        self.start_time = datetime.now()
        ts = self.start_time.strftime("%Y%m%d_%H%M%S") + "_retry"

        if not await self._check_server():
            logging.critical("Aborting retry: server not healthy."); return

        with open(checkpoint_path, "r", encoding="utf-8") as f:
            ckpt = json.load(f)

        self.responses             = ckpt.get("responses", [])
        self.parse_failed_requests = ckpt.get("parse_failed_requests", [])
        self.valid_responses       = [r for r in self.responses if r.get("parse_success")]
        failed                     = ckpt.get("api_failed_requests", [])

        logging.info(f"Retry: restoring {len(self.responses)} ok | "
                     f"retrying {len(failed)} failed")
        if not failed:
            logging.info("Nothing to retry."); return

        pbn = {pc["name"]: (pi, pc) for pi, pc in enumerate(self.prompts)}
        tlookup: Dict[Tuple, Dict] = {}
        for fmt_records in self.table_map.values():
            for records in fmt_records.values():
                for tr in records:
                    key = (tr["source_group"], tr["source_stem"],
                           tr["table_index"], tr["table_hash"])
                    tlookup[key] = tr

        tasks, skipped = [], 0
        for rec in failed:
            pname = rec.get("prompt_name", "")
            if pname not in pbn:
                logging.warning(f"Prompt '{pname}' not found, skip")
                skipped += 1; continue
            pi, pc = pbn[pname]
            key = (rec.get("source_group"), rec.get("source_stem"),
                   rec.get("table_index", 0), rec.get("table_hash", ""))
            tr  = tlookup.get(key)
            if tr is None:
                key2 = next(((sg, ss, ti, th) for (sg, ss, ti, th) in tlookup
                             if sg == rec.get("source_group")
                             and ss == rec.get("source_stem")
                             and ti == rec.get("table_index", 0)), None)
                tr = tlookup.get(key2) if key2 else None
            if tr is None:
                logging.warning(f"Table not found for {key}, skip")
                skipped += 1; continue
            fmt = rec.get("table_format", "json")
            tasks.append((pi, pc, tr, fmt))

        logging.info(f"Rebuilt {len(tasks)} retry tasks (skipped {skipped})")
        self.completed_count = len(self.responses) + len(self.parse_failed_requests)
        await self._run_tasks(tasks, ts)
        self._save_final_results(ts)
        self._build_metrics_artifacts(ts)

    def run_retry(self, checkpoint_path: str):
        asyncio.run(self._run_retry_async(checkpoint_path))

    # ---------- persistence ----------

    def _save_checkpoint(self, timestamp: str):
        path = self.ckpt_dir / f"checkpoint_{timestamp}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump({
                "metadata": {
                    "timestamp":            datetime.now().isoformat(),
                    "model":                MODEL_NAME,
                    "model_alias":          self.model_alias,
                    "guided_decoding":      "none",
                    "chunk_cell_threshold": CHUNK_CELL_THRESHOLD,
                    "max_tokens":           MAX_TOKENS,
                    "max_tokens_by_prompt": MAX_TOKENS_BY_PROMPT,
                    "completed_count":      self.completed_count,
                    "responses":            len(self.responses),
                    "api_failed":           len(self.api_failed_requests),
                    "parse_failed":         len(self.parse_failed_requests),
                },
                "responses":             self.responses,
                "api_failed_requests":   self.api_failed_requests,
                "parse_failed_requests": self.parse_failed_requests,
            }, f, ensure_ascii=False, indent=2)
        logging.info(f"Checkpoint ({self.completed_count} done): {path}")

    def _save_final_results(self, timestamp: str):
        base = self.results_dir
        for name, d in [
            (f"responses_{timestamp}.json",    self.responses),
            (f"api_failed_{timestamp}.json",   self.api_failed_requests),
            (f"parse_failed_{timestamp}.json", self.parse_failed_requests),
        ]:
            with open(base / name, "w", encoding="utf-8") as f:
                json.dump(d, f, ensure_ascii=False, indent=2)

        for name, d in [
            (f"responses_{timestamp}.csv",    self.responses),
            (f"api_failed_{timestamp}.csv",   self.api_failed_requests),
            (f"parse_failed_{timestamp}.csv", self.parse_failed_requests),
        ]:
            if d: dataframe_friendly(d).to_csv(base / name, index=False, encoding="utf-8-sig")

        try:
            with pd.ExcelWriter(base / f"results_{timestamp}.xlsx",
                                engine="openpyxl") as w:
                dataframe_friendly(self.responses).to_excel(
                    w, sheet_name="responses", index=False)
                dataframe_friendly(self.api_failed_requests).to_excel(
                    w, sheet_name="api_failed", index=False)
                dataframe_friendly(self.parse_failed_requests).to_excel(
                    w, sheet_name="parse_failed", index=False)
        except Exception as e:
            logging.warning(f"XLSX save failed: {e}")

        logging.info(f"Results → {base}")

    def _summarize_table(self, df: pd.DataFrame, group_cols=None,
                          metric_prefix: str = "") -> pd.DataFrame:
        if df.empty: return pd.DataFrame()
        metrics = ["support","pred_count","tp","fp","fn","precision","recall",
                   "f1","jaccard","exact_match","partial_match","header_coverage"]
        rows = []
        iterable = df.groupby(group_cols, dropna=False) if group_cols else [((), df)]
        for key, sub in iterable:
            row: Dict[str, Any] = {}
            if group_cols:
                ks = [key] if len(group_cols) == 1 else list(key)
                for col, val in zip(group_cols, ks): row[col] = val
            row["count"] = len(sub)
            for col, alias in [("api_success", "api_success_rate"),
                                ("parse_success", "parse_success_rate"),
                                ("completion_capped", "capped_rate")]:
                if col in sub.columns:
                    row[alias] = float(pd.to_numeric(sub[col], errors="coerce").mean())
            for m in metrics:
                col = f"{metric_prefix}_{m}" if metric_prefix else m
                if col in sub.columns:
                    s = pd.to_numeric(sub[col], errors="coerce")
                    if s.notna().any():
                        row[f"{m}_mean"]   = float(s.mean())
                        row[f"{m}_median"] = float(s.median())
            for col in ["duration_sec", "prompt_tokens", "completion_tokens",
                        "total_tokens", "token_efficiency", "max_tokens_used"]:
                if col in sub.columns:
                    s = pd.to_numeric(sub[col], errors="coerce").dropna()
                    if len(s):
                        row[f"{col}_mean"]   = float(s.mean())
                        row[f"{col}_median"] = float(s.median())
                        if col == "duration_sec":
                            row["duration_p90"] = float(s.quantile(0.90))
                            row["duration_p99"] = float(s.quantile(0.99))
            rows.append(row)
        return pd.DataFrame(rows)

    def _build_metrics_artifacts(self, timestamp: str):
        all_r = self.responses + self.api_failed_requests
        if not all_r: return
        df = pd.DataFrame(all_r)

        views = {
            "overall":           self._summarize_table(df),
            "by_model":          self._summarize_table(df, ["model_alias"]),
            "by_prompt":         self._summarize_table(df, ["prompt_name"]),
            "by_format":         self._summarize_table(df, ["table_format"]),
            "by_prompt_format":  self._summarize_table(df, ["prompt_name", "table_format"]),
            "by_source":         self._summarize_table(df, ["source_group"]),
            "by_prompt_source":  self._summarize_table(df, ["prompt_name", "source_group"]),
            "by_rows_bin":       self._summarize_table(df, ["table_rows_bin"]),
            "by_headers_bin":    self._summarize_table(df, ["true_headers_count_bin"]),
            "by_spanning_bin":   self._summarize_table(df, ["spanning_cell_count_bin"]),
        }

        type_frames = []
        for ht in ["column_headers", "projected_row_headers", "spanning"]:
            if (f"{ht}_f1" in df.columns
                    and pd.to_numeric(df[f"{ht}_f1"], errors="coerce").notna().any()):
                tmp = self._summarize_table(
                    df, ["prompt_name", "table_format"], metric_prefix=ht
                )
                if not tmp.empty:
                    tmp.insert(2, "header_type", ht)
                    type_frames.append(tmp)
        views["by_prompt_type"] = (pd.concat(type_frames, ignore_index=True)
                                   if type_frames else pd.DataFrame())

        for key, vdf in views.items():
            if not vdf.empty:
                vdf.to_csv(self.metrics_dir / f"metrics_{key}_{timestamp}.csv",
                           index=False, encoding="utf-8-sig")
        try:
            with pd.ExcelWriter(self.metrics_dir / f"metrics_{timestamp}.xlsx",
                                engine="openpyxl") as w:
                for key, vdf in views.items():
                    if not vdf.empty:
                        vdf.to_excel(w, sheet_name=key[:31], index=False)
        except Exception as e:
            logging.warning(f"Metrics XLSX failed: {e}")

        with open(self.metrics_dir / f"metrics_{timestamp}.json",
                  "w", encoding="utf-8") as f:
            json.dump({k: v.to_dict(orient="records") for k, v in views.items()},
                      f, ensure_ascii=False, indent=2)

        ov    = views["overall"].iloc[0].to_dict() if not views["overall"].empty else {}
        total = len(all_r); ok = len(self.responses)
        with open(self.metrics_dir / f"metrics_summary_{timestamp}.txt",
                  "w", encoding="utf-8") as f:
            f.write("METRICS SUMMARY\n" + "=" * 80 + "\n")
            f.write(f"Model:            {MODEL_NAME}\n")
            f.write(f"Model alias:      {self.model_alias}\n")
            f.write("guided_decoding: none (free output)\n")
            f.write(f"Coord system:     0-based (eval vs true_headers_raw)\n")
            f.write(f"Chunk threshold:  {CHUNK_CELL_THRESHOLD} cells (rows×cols)\n")
            f.write(f"Max tokens:       {MAX_TOKENS} default | overrides={MAX_TOKENS_BY_PROMPT}\n")
            f.write(f"Total: {total}  ok: {ok} ({ok/total*100:.1f}%)\n\n")
            for label, key in [
                ("Precision",     "precision_mean"),
                ("Recall",        "recall_mean"),
                ("F1",            "f1_mean"),
                ("Jaccard",       "jaccard_mean"),
                ("Exact match",   "exact_match_mean"),
                ("Partial match", "partial_match_mean"),
            ]:
                f.write(f"  {label:14s}: {ov.get(key, 0):.4f}\n")
            f.write(f"  Capped rate:   {ov.get('capped_rate', 0):.1%}\n")
            f.write(f"  Completion tok: {ov.get('completion_tokens_mean', 0):.1f} (mean)\n\n")
            f.write("JSON vs HTML:\n")
            if not views["by_format"].empty:
                f.write(views["by_format"].to_string(index=False) + "\n\n")
            f.write("By prompt × format:\n")
            if not views["by_prompt_format"].empty:
                f.write(views["by_prompt_format"].to_string(index=False) + "\n\n")
            f.write("By prompt:\n")
            if not views["by_prompt"].empty:
                f.write(views["by_prompt"].to_string(index=False) + "\n\n")
            f.write("By table size:\n")
            if not views["by_rows_bin"].empty:
                f.write(views["by_rows_bin"].to_string(index=False) + "\n")

        logging.info(f"Metrics → {self.metrics_dir}")
        logging.info(f"F1={ov.get('f1_mean',0):.4f}  "
                     f"Exact={ov.get('exact_match_mean',0):.4f}  "
                     f"Capped={ov.get('capped_rate',0):.1%}")


# =========================
# ENTRY POINT
# =========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Table header detection — v6 (multi-model, fixed chunking)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--vllm-url",      default=None)
    parser.add_argument("--model",         default=None,
                        help="Full model name on vLLM server")
    parser.add_argument("--model-alias",   default=None,
                        help="Short alias for run dir name, e.g. 'qwen30b', 'llama8b'")
    parser.add_argument("--output-dir",    default=None)
    parser.add_argument("--concurrency",   type=int,   default=None)
    parser.add_argument("--max-tokens",    type=int,   default=None,
                        help="Default max tokens (reasoning prompts get more via override)")
    parser.add_argument("--timeout",       type=float, default=None)
    parser.add_argument("--inter-delay",   type=float, default=None)
    parser.add_argument("--early-stop",    type=int,   default=None)
    parser.add_argument("--total-tables",  type=int,   default=None)
    parser.add_argument("--format-ratio",  default=None,
                        help="JSON:HTML ratio, e.g. '50:50' or '100:0'")
    parser.add_argument("--chunk-cells",   type=int,   default=None,
                        help=f"Chunking threshold in cells (rows×cols). "
                             f"Default: {CHUNK_CELL_THRESHOLD}")
    parser.add_argument("--table-seed",    default=None, metavar="PATH",
                        help="Path to selected_tables.json from a prior run. "
                             "Ensures all models use identical table sets.")
    parser.add_argument("--retry",         default=None, metavar="CHECKPOINT_PATH",
                        help="Retry api_failed entries from a checkpoint")
    args = parser.parse_args()

    if args.vllm_url:    VLLM_BASE_URL      = args.vllm_url
    if args.model:       MODEL_NAME         = args.model
    if args.output_dir:
        OUTPUT_DIR = args.output_dir
        Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    if args.concurrency  is not None: CONCURRENCY          = args.concurrency
    if args.max_tokens   is not None: MAX_TOKENS           = args.max_tokens
    if args.timeout      is not None: REQUEST_TIMEOUT_SEC  = args.timeout
    if args.inter_delay  is not None: INTER_REQUEST_DELAY  = args.inter_delay
    if args.early_stop   is not None: EARLY_STOP_FAILURES  = args.early_stop
    if args.chunk_cells  is not None: CHUNK_CELL_THRESHOLD = args.chunk_cells

    model_alias  = args.model_alias  if args.model_alias  else MODEL_ALIAS
    total_tables = args.total_tables if args.total_tables is not None else TOTAL_TABLES
    format_ratio = args.format_ratio if args.format_ratio is not None else FORMAT_RATIO

    collector = ResponseCollector(
        output_dir=OUTPUT_DIR,
        total_tables=total_tables,
        format_ratio=format_ratio,
        model_alias=model_alias,
        table_seed_path=args.table_seed,
    )
    if args.retry:
        logging.info(f"=== RETRY MODE: {args.retry} ===")
        collector.run_retry(args.retry)
    else:
        collector.run()