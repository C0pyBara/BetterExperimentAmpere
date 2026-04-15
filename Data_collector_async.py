"""
Table header detection experiment — v5

Changes vs v4:
  STRUCTURED OUTPUT: guided_regex with compact pattern (no whitespace allowed)
    - Pattern: (\\d+ \\d+\\n)* — "row col\n" per line, lm-format-enforcer backend
    - Eliminates outlines flooding; guarantees 100% parseable output
    - max_tokens raised to 8192 for reasoning prompts headroom

  HTML FORMAT:
    - Reads pre-converted .html files from html_root directory
    - Strips <th>/<th> → <td>/<td> and removes class/id/style attrs before prompting
    - Ground truth stays identical (coordinates from JSON, same anchor logic)
    - HTML coordinate = (r0, c0) from json_to_html_table converter = row_nums[0], col_nums[0]

  SAMPLING:
    - --total-tables N: total unique tables to use across experiment
    - --format-ratio R: JSON:HTML ratio, e.g. "50:50" or "100:0" or "0:100"
    - Tables sampled deterministically (sorted filenames) from available files

  COORD SYSTEM: 0-based throughout (matches converter and source JSON)
  METRICS: compare pred_set vs true_headers_raw, OOB filtered by table dims
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

OUTPUT_DIR   = os.getenv("OUTPUT_DIR",   "results")
LOG_LEVEL    = os.getenv("LOG_LEVEL",    "INFO")

MAX_RETRIES         = int(os.getenv("MAX_RETRIES",          "2"))
RETRY_BACKOFF_BASE  = float(os.getenv("RETRY_BACKOFF_BASE", "3.0"))
TEMPERATURE         = float(os.getenv("TEMPERATURE",        "0.0"))
MAX_TOKENS          = int(os.getenv("MAX_TOKENS",           "8192"))

CONCURRENCY         = int(os.getenv("CONCURRENCY",          "2"))
CHECKPOINT_EVERY    = int(os.getenv("CHECKPOINT_EVERY",     "10"))
REQUEST_TIMEOUT_SEC = float(os.getenv("REQUEST_TIMEOUT_SECONDS", "300.0"))
INTER_REQUEST_DELAY = float(os.getenv("INTER_REQUEST_DELAY",     "1.0"))
EARLY_STOP_FAILURES = int(os.getenv("EARLY_STOP_FAILURES",       "10"))

CHUNK_THRESHOLD     = int(os.getenv("CHUNK_THRESHOLD", "100"))
CHUNK_SIZE          = int(os.getenv("CHUNK_SIZE",      "80"))
CHUNK_OVERLAP       = int(os.getenv("CHUNK_OVERLAP",   "10"))

# Total tables and JSON:HTML ratio — overridable via CLI
TOTAL_TABLES  = int(os.getenv("TOTAL_TABLES",  "0"))   # 0 = use all available
FORMAT_RATIO  = os.getenv("FORMAT_RATIO", "50:50")      # "JSON:HTML"

# guided_regex: "row col\n" lines, no whitespace inside numbers, no extra text
# lm-format-enforcer enforces this at token level → 100% parseable output
# Pattern allows: empty response OR one-or-more "int int\n" lines
GUIDED_REGEX = r"(\d+ \d+\n)*"
GUIDED_BACKEND = "lm-format-enforcer"

# =========================
# EXPERIMENT PLAN
# Each entry: name, json_root, html_root, limit, prompts
# limit is the max tables to load; actual count depends on --total-tables
# =========================
EXPERIMENT_PLAN = [
    {
        "name":      "pubtables_complex_top500",
        "json_root": PROJECT_ROOT / "Get_500_Tables_from_PubTables" / "JSON_Complex_TOP500_normalized",
        "html_root": PROJECT_ROOT / "Get_500_Tables_from_PubTables" / "HTML_Complex_TOP500",
        "limit":     500,
        "prompts":   None,  # all prompts
    },
    {
        "name":      "expert_viewpoint",
        "json_root": PROJECT_ROOT / "Convert_from_xlsx_to_Json" / "expert_viewpoint_converted_json",
        "html_root": PROJECT_ROOT / "Convert_from_xlsx_to_Json" / "expert_viewpoint_converted_html",
        "limit":     500,
        "prompts":   None,
    },
    {
        "name":      "maximum_viewpoint",
        "json_root": PROJECT_ROOT / "Convert_from_xlsx_to_Json" / "maximum_viewpoint_converted_json",
        "html_root": PROJECT_ROOT / "Convert_from_xlsx_to_Json" / "maximum_viewpoint_converted_html",
        "limit":     500,
        "prompts":   ["zero_max", "reasoning_max"],
    },
    {
        "name":      "table_normalization",
        "json_root": PROJECT_ROOT / "Convert_from_xlsx_to_Json" / "table_normalization_converted_json",
        "html_root": PROJECT_ROOT / "Convert_from_xlsx_to_Json" / "table_normalization_converted_html",
        "limit":     500,
        "prompts":   ["zero_min", "reasoning_min"],
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
# FORMAT RATIO PARSER
# =========================
def parse_format_ratio(ratio_str: str) -> Tuple[float, float]:
    """Parse 'J:H' string, return (json_frac, html_frac) summing to 1.0."""
    try:
        j, h = ratio_str.strip().split(":")
        j, h = float(j), float(h)
        total = j + h
        if total <= 0:
            raise ValueError
        return j / total, h / total
    except Exception:
        raise ValueError(f"Invalid FORMAT_RATIO '{ratio_str}'. Use e.g. '50:50' or '70:30'.")


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


def coords_to_set(coords: List[Dict[str, int]]) -> set:
    out = set()
    for h in coords or []:
        try:
            out.add((int(h["row"]), int(h["col"])))
        except Exception:
            continue
    return out


def to_one_based_coords(coords: List[Dict[str, int]]) -> List[Dict[str, int]]:
    """0-based → 1-based; stored as reference only, not used in eval."""
    out = set()
    for h in coords:
        try:
            out.add((int(h["row"]) + 1, int(h["col"]) + 1))
        except Exception:
            continue
    return [{"row": r, "col": c} for r, c in sorted(out)]


def extract_true_coords_from_cells(cells: List[Dict[str, Any]]) -> List[Dict[str, int]]:
    """
    Extract 0-based anchor coords (row_nums[0], col_nums[0]) for header cells.
    Matches json_to_html_table converter logic: r0=rows[0], c0=cols[0].
    Includes is_column_header, is_projected_row_header, is_spanning.
    """
    coords = set()
    for cell in cells or []:
        if (bool(cell.get("is_column_header"))
                or bool(cell.get("is_projected_row_header"))
                or bool(cell.get("is_spanning"))):
            row_nums = cell.get("row_nums", []) or []
            col_nums = cell.get("column_nums", []) or []
            if row_nums and col_nums:
                try:
                    coords.add((int(row_nums[0]), int(col_nums[0])))
                except Exception:
                    continue
    return [{"row": r, "col": c} for r, c in sorted(coords)]


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


def extract_true_coords_from_headers(headers: List[Any]) -> List[Dict[str, int]]:
    coords = set()
    for h in headers or []:
        if isinstance(h, dict) and "row" in h and "col" in h:
            try:
                coords.add((int(h["row"]), int(h["col"])))
            except Exception:
                continue
    return [{"row": r, "col": c} for r, c in sorted(coords)]


def evaluate_coord_sets(true_set: set, pred_set: set,
                         table_rows: int = 0, table_cols: int = 0) -> Dict[str, Any]:
    """
    Compute P/R/F1/Jaccard/Exact/Partial in 0-based space.
    OOB filter: drop pred coords outside [0, table_rows) x [0, table_cols).
    """
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


def dataframe_friendly(records: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for rec in records:
        row = dict(rec)
        for key in ["true_headers_raw", "true_headers_1based",
                    "true_headers_by_type_raw", "parsed_headers"]:
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


# =========================
# HTML PREPROCESSING
# =========================
def strip_html_header_hints(html: str) -> str:
    """
    Replace <th ...> with <td> and </th> with </td>.
    Remove class, id, style attributes from all tags.
    This prevents the model from seeing semantic hints about header cells.
    """
    # th → td
    html = re.sub(r"<th(\b[^>]*)>", r"<td\1>", html, flags=re.IGNORECASE)
    html = re.sub(r"</th>",          "</td>",    html, flags=re.IGNORECASE)
    # strip class, id, style attributes
    html = re.sub(r'\s+(?:class|id|style)=["\'][^"\']*["\']', "", html, flags=re.IGNORECASE)
    return html


# =========================
# OUTPUT PARSER
# =========================
def parse_output(raw_text: str) -> Tuple[bool, List[Dict[str, int]], str]:
    """
    Parse guided_regex output: zero or more "row col\n" lines (0-based).
    Also handles fallback formats for robustness.
    """
    if not raw_text or not str(raw_text).strip():
        return True, [], ""  # empty = no headers, valid

    text = str(raw_text).strip()

    seen   = set()
    coords = []

    # Primary: "int int" per line (guided_regex guaranteed format)
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        m = re.fullmatch(r"(\d+)\s+(\d+)", line)
        if m:
            r, c = int(m.group(1)), int(m.group(2))
            if (r, c) not in seen:
                seen.add((r, c))
                coords.append({"row": r, "col": c})

    if coords:
        coords.sort(key=lambda x: (x["row"], x["col"]))
        return True, coords, ""

    # Fallback A: any "int int" pair on a line (looser match)
    for line in text.splitlines():
        m = re.search(r"(\d+)\D+(\d+)", line.strip())
        if m:
            r, c = int(m.group(1)), int(m.group(2))
            if (r, c) not in seen:
                seen.add((r, c))
                coords.append({"row": r, "col": c})
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
                coords.append({"row": r, "col": c})
        coords.sort(key=lambda x: (x["row"], x["col"]))
        return True, coords, "fallback_json_array"

    # Fallback C: (row,col) paren format
    parens = re.findall(r"\(\s*(\d+)\s*,\s*(\d+)\s*\)", text)
    if parens:
        for rs, cs in parens:
            r, c = int(rs), int(cs)
            if (r, c) not in seen:
                seen.add((r, c))
                coords.append({"row": r, "col": c})
        coords.sort(key=lambda x: (x["row"], x["col"]))
        return True, coords, "fallback_paren_format"

    return False, [], "no_parseable_coordinates"


# =========================
# CHUNKING (JSON only; HTML tables use pre-built files)
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
    """Returns list of (json_repr, row_offset_0based)."""
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
    """Merge 0-based per-chunk predictions to full-table 0-based coords."""
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
) -> Dict[str, Any]:
    url     = f"{VLLM_BASE_URL}/chat/completions"
    payload = {
        "model":       model,
        "messages":    messages,
        "temperature": TEMPERATURE,
        "max_tokens":  MAX_TOKENS,
        # Structured output via guided_regex:
        # Pattern "(\\d+ \\d+\\n)*" forces output to be zero or more "row col\n" lines.
        # lm-format-enforcer backend — handles this simple pattern reliably.
        # No whitespace injection possible → 100% parseable output guaranteed.
        "guided_regex":             GUIDED_REGEX,
        "guided_decoding_backend":  GUIDED_BACKEND,
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
        "retry_attempts": MAX_RETRIES, "tokens_used": None,
        "error_type":    classify_api_error(last_error),
        "error_message": last_error,
    }


# =========================
# MAIN CLASS
# =========================
class ResponseCollector:
    def __init__(self, output_dir: str = OUTPUT_DIR,
                 total_tables: int = TOTAL_TABLES,
                 format_ratio: str = FORMAT_RATIO):
        self.total_tables  = total_tables
        self.json_frac, self.html_frac = parse_format_ratio(format_ratio)

        self.run_id      = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self.base_dir    = Path(output_dir)
        self.run_dir     = self.base_dir / f"run_{self.run_id}"
        self.logs_dir    = self.run_dir / "logs"
        self.results_dir = self.run_dir / "results"
        self.metrics_dir = self.run_dir / "metrics"
        self.ckpt_dir    = self.run_dir / "checkpoints"

        for d in [self.logs_dir, self.results_dir, self.metrics_dir, self.ckpt_dir]:
            d.mkdir(parents=True, exist_ok=True)

        self._setup_file_logging()

        self.system_prompt = self._load_system_prompt()
        self.prompts       = self._load_prompt_configs()
        # table_map[source_name] = {"json": [...], "html": [...]}
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
            f"  Model: {MODEL_NAME}\n"
            f"  guided_regex: {GUIDED_REGEX!r} backend: {GUIDED_BACKEND}\n"
            f"  max_tokens={MAX_TOKENS} concurrency={CONCURRENCY} "
            f"timeout={REQUEST_TIMEOUT_SEC}s\n"
            f"  total_tables={self.total_tables or 'all'} "
            f"format_ratio=json:{self.json_frac:.0%} html:{self.html_frac:.0%}"
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
        url = f"{VLLM_BASE_URL}/health"
        try:
            async with httpx.AsyncClient(timeout=10.0) as c:
                r = await c.get(url)
                if r.status_code == 200:
                    logging.info(f"Server healthy: {url}"); return True
                logging.critical(f"Server returned {r.status_code}"); return False
        except Exception as e:
            logging.critical(f"Server not reachable at {url}: {e}"); return False

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
        """Build a table record from a JSON item. Contains both JSON and HTML repr."""
        if not isinstance(item, dict): return None

        prompt_obj = sanitize_for_prompt(item)
        table_json = json.dumps(prompt_obj, ensure_ascii=False, separators=(",", ":"))
        table_hash = stable_hash(table_json, 12)

        if "cells" in item:
            ti       = extract_type_coords_from_cells(item.get("cells", []))
            true_raw = extract_true_coords_from_cells(item.get("cells", []))
            kind     = "cells"
            nr, nc   = self._table_dims_cells(item.get("cells", []))
            has_ti   = True
        elif "data" in item:
            true_raw = extract_true_coords_from_headers(item.get("headers", []))
            kind     = "matrix"
            nr, nc   = self._table_dims_matrix(item.get("data", []))
            has_ti   = False
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
            "table_json":     table_json,          # JSON repr (sanitized)
            # HTML repr loaded separately from html_root; set to "" initially
            "table_html":     "",
            # Ground truth — 0-based anchor coords
            "true_headers_raw":         true_raw,
            "true_headers_1based":      to_one_based_coords(true_raw),
            "true_headers_count":       len(true_raw),
            "true_headers_count_bin":   count_bin(len(true_raw)),
            "true_headers_by_type_raw": th_by_type,
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
                # Handle both single-item dict and list formats
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
        """
        For each record, look up the matching .html file by stem.
        If found, load and preprocess (strip <th>, attrs).
        Records without an HTML file are kept but table_html stays "".
        """
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
                        n_json: int, n_html: int) -> Tuple[List[Dict], List[Dict]]:
        """
        Split records into json_sample and html_sample.
        html_sample only includes records that have table_html loaded.
        """
        # Records with HTML available
        with_html = [r for r in records if r.get("table_html")]
        # All records can be used for JSON
        json_sample = records[:n_json]
        html_sample = with_html[:n_html]
        return json_sample, html_sample

    def _load_all_tables(self) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
        """
        Returns table_map[source_name] = {"json": [...records...], "html": [...records...]}
        Respects --total-tables and --format-ratio.
        """
        # First pass: load all JSON records per source
        raw_map: Dict[str, List[Dict[str, Any]]] = {}
        for src in EXPERIMENT_PLAN:
            records = self._load_json_records(
                src["json_root"], src["name"], int(src["limit"])
            )
            records = self._attach_html(records, src["html_root"])
            raw_map[src["name"]] = records
            logging.info(f"Loaded {len(records)} records from {src['name']}")

        # Determine per-source sampling counts
        if self.total_tables > 0:
            # Distribute total_tables proportionally across sources
            n_sources   = len(EXPERIMENT_PLAN)
            per_source  = self.total_tables // n_sources
            remainder   = self.total_tables - per_source * n_sources
        else:
            per_source  = None
            remainder   = 0

        table_map: Dict[str, Dict[str, List[Dict]]] = {}
        total_json = total_html = 0

        for i, src in enumerate(EXPERIMENT_PLAN):
            records = raw_map[src["name"]]
            if per_source is not None:
                n_src = per_source + (1 if i < remainder else 0)
                records = records[:n_src]

            n_json = round(len(records) * self.json_frac)
            n_html = len(records) - n_json

            json_sample, html_sample = self._sample_tables(records, n_json, n_html)
            table_map[src["name"]] = {
                "json": json_sample,
                "html": html_sample,
            }
            total_json += len(json_sample)
            total_html += len(html_sample)
            logging.info(
                f"  {src['name']}: "
                f"json={len(json_sample)} html={len(html_sample)}"
            )

        logging.info(f"Total: json={total_json} html={total_html} "
                     f"sum={total_json+total_html}")
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
                f"__t{tr['table_index']}__{tr['table_hash']}__{table_format}")
        return slugify(base)

    # ---------- result assembly ----------

    def _make_result(self, prompt_idx: int, prompt_config: Dict,
                     table_record: Dict, api_result: Dict,
                     table_format: str,
                     chunked: bool = False, n_chunks: int = 1) -> Dict[str, Any]:
        pname    = str(prompt_config.get("name", f"prompt_{prompt_idx}"))
        tr       = table_record
        true_set = coords_to_set(tr["true_headers_raw"])   # 0-based
        pred_set = (coords_to_set(api_result["parsed_headers"])
                    if api_result["api_success"] and api_result["parse_success"]
                    else set())

        overall  = evaluate_coord_sets(
            true_set, pred_set,
            table_rows=tr["table_rows"], table_cols=tr["table_cols"]
        )

        tu  = api_result.get("tokens_used") or {}
        ct  = tu.get("completion")
        tef = (ct / overall["f1"]) if (ct and overall["f1"] > 0) else None

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
            # Ground truth — 0-based anchor coords (same for JSON and HTML)
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
            "tokens_used":       api_result.get("tokens_used"),
            "prompt_tokens":     tu.get("prompt"),
            "completion_tokens": ct,
            "total_tokens":      tu.get("total"),
            "token_efficiency":  tef,
            "system_prompt_file":"prompts/system.txt",
        }
        result.update(overall)
        result.update(type_metrics)
        return result

    # ---------- async workers ----------

    async def _call_one(self, client: httpx.AsyncClient,
                         semaphore: asyncio.Semaphore,
                         messages: List[Dict]) -> Dict[str, Any]:
        async with semaphore:
            result = await async_api_call(client, MODEL_NAME, messages)
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
                "retry_attempts": 0, "tokens_used": None,
                "error_type": "aborted", "error_message": "early stop",
            }, table_format)

        # Select representation
        if table_format == "html":
            table_repr = table_record.get("table_html") or table_record["table_json"]
        else:
            table_repr = table_record["table_json"]

        # Non-chunked path
        if table_record["table_rows"] <= CHUNK_THRESHOLD:
            msgs = self._prepare_messages(prompt_config, table_repr, table_format)
            ar   = await self._call_one(client, semaphore, msgs)
            return self._make_result(prompt_idx, prompt_config, table_record,
                                     ar, table_format)

        # Chunked path — JSON only (HTML files are pre-built, not chunked)
        # For HTML of large tables: send full HTML (may be large but no JSON overhead)
        if table_format == "html":
            msgs = self._prepare_messages(prompt_config, table_repr, table_format)
            ar   = await self._call_one(client, semaphore, msgs)
            return self._make_result(prompt_idx, prompt_config, table_record,
                                     ar, table_format)

        # JSON chunked
        chunks = make_chunks(table_record["table_json"],
                             table_record["table_rows"],
                             table_record["table_kind"])
        logging.info(f"Chunking {table_record['source_stem']} "
                     f"({table_record['table_rows']} rows) → {len(chunks)} chunks")

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
            ar   = await self._call_one(client, semaphore, msgs)

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

                        # Early-stop on connection failures
                        if result.get("error_type") == "connection_error" \
                                or result.get("error_message") == "early stop":
                            self._consec_fail += 1
                        else:
                            self._consec_fail = 0

                        if self._consec_fail >= EARLY_STOP_FAILURES and not self._abort:
                            self._abort = True
                            logging.critical(
                                f"EARLY STOP: {self._consec_fail} consecutive connection "
                                f"failures. Saving checkpoint.")
                            self._save_checkpoint(timestamp)

                        dur_s = (f"{result['duration_sec']:.1f}s"
                                 if result.get("duration_sec") else "n/a")
                        c_str = (f" [×{result.get('n_chunks',1)}ch]"
                                 if result.get("chunked") else "")
                        f_str = f" [{result.get('table_format','?')}]"
                        f1_s  = (f" F1={result.get('f1',0):.3f}"
                                 if result.get("f1") is not None else "")
                        logging.info(
                            f"[{self.completed_count}/{total}] "
                            f"{result['prompt_name']}{f_str} | "
                            f"{result['source_stem']}{c_str} | "
                            f"status={result['status']}{f1_s} | dur={dur_s}"
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
                logging.info(f"  {src['name']} [{fmt}]: {np} prompts × {len(recs)} tables"
                             f" = {np * len(recs)}")
        logging.info(
            f"Total {total} tasks | CONCURRENCY={CONCURRENCY} "
            f"MAX_TOKENS={MAX_TOKENS} guided_regex={GUIDED_REGEX!r} "
            f"backend={GUIDED_BACKEND} early_stop={EARLY_STOP_FAILURES}"
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
                # Fallback without hash
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
                    "timestamp":       datetime.now().isoformat(),
                    "model":           MODEL_NAME,
                    "guided_regex":    GUIDED_REGEX,
                    "guided_backend":  GUIDED_BACKEND,
                    "completed_count": self.completed_count,
                    "responses":       len(self.responses),
                    "api_failed":      len(self.api_failed_requests),
                    "parse_failed":    len(self.parse_failed_requests),
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
                                ("parse_success", "parse_success_rate")]:
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
                        "total_tokens", "token_efficiency"]:
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
            f.write(f"Model:          {MODEL_NAME}\n")
            f.write(f"guided_regex:   {GUIDED_REGEX!r}\n")
            f.write(f"guided_backend: {GUIDED_BACKEND}\n")
            f.write(f"Coord system:   0-based (eval vs true_headers_raw)\n")
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
            f.write(f"  Completion tok:  {ov.get('completion_tokens_mean', 0):.1f} (mean)\n\n")
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
                     f"Exact={ov.get('exact_match_mean',0):.4f}")


# =========================
# ENTRY POINT
# =========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Table header detection — v5 (guided_regex, JSON+HTML)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--vllm-url",      default=None)
    parser.add_argument("--model",         default=None)
    parser.add_argument("--output-dir",    default=None)
    parser.add_argument("--concurrency",   type=int,   default=None)
    parser.add_argument("--max-tokens",    type=int,   default=None)
    parser.add_argument("--timeout",       type=float, default=None)
    parser.add_argument("--inter-delay",   type=float, default=None,
                        help="Seconds between requests (throttle)")
    parser.add_argument("--early-stop",    type=int,   default=None,
                        help="Consecutive connection failures before abort")
    parser.add_argument("--total-tables",  type=int,   default=None,
                        help="Total unique tables to use (0 = all). "
                             "Distributed proportionally across sources.")
    parser.add_argument("--format-ratio",  default=None,
                        help="JSON:HTML ratio, e.g. '50:50' or '70:30'")
    parser.add_argument("--retry", metavar="CHECKPOINT_PATH", default=None,
                        help="Retry api_failed entries from a checkpoint")
    args = parser.parse_args()

    if args.vllm_url:    VLLM_BASE_URL      = args.vllm_url
    if args.model:       MODEL_NAME         = args.model
    if args.output_dir:
        OUTPUT_DIR = args.output_dir
        Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    if args.concurrency  is not None: CONCURRENCY         = args.concurrency
    if args.max_tokens   is not None: MAX_TOKENS          = args.max_tokens
    if args.timeout      is not None: REQUEST_TIMEOUT_SEC = args.timeout
    if args.inter_delay  is not None: INTER_REQUEST_DELAY = args.inter_delay
    if args.early_stop   is not None: EARLY_STOP_FAILURES = args.early_stop

    total_tables = args.total_tables if args.total_tables is not None else TOTAL_TABLES
    format_ratio = args.format_ratio  if args.format_ratio  is not None else FORMAT_RATIO

    collector = ResponseCollector(
        output_dir=OUTPUT_DIR,
        total_tables=total_tables,
        format_ratio=format_ratio,
    )
    if args.retry:
        logging.info(f"=== RETRY MODE: {args.retry} ===")
        collector.run_retry(args.retry)
    else:
        collector.run()