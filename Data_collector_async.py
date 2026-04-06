import os
import json
import time
import logging
import re
import hashlib
import asyncio
import argparse
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import pandas as pd
import httpx


# =========================
# CONFIG
# =========================
PROJECT_ROOT = Path(__file__).resolve().parent
PROMPTS_DIR = PROJECT_ROOT / "prompts"

VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://127.0.0.1:8000/v1")
VLLM_API_KEY = os.getenv("VLLM_API_KEY", "EMPTY")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen3-VL-30B-A3B-Thinking")

OUTPUT_DIR = os.getenv("OUTPUT_DIR", "results")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.0"))

CONCURRENCY = int(os.getenv("CONCURRENCY", "2"))
CHECKPOINT_EVERY = int(os.getenv("CHECKPOINT_EVERY", "20"))

REQUEST_TIMEOUT_SECONDS = float(os.getenv("REQUEST_TIMEOUT_SECONDS", "300.0"))
RETRY_BACKOFF_BASE = float(os.getenv("RETRY_BACKOFF_BASE", "2.0"))

REQUEST_PAUSE_EVERY = int(os.getenv("REQUEST_PAUSE_EVERY", "2"))
REQUEST_PAUSE_SECONDS = float(os.getenv("REQUEST_PAUSE_SECONDS", "2.0"))

# Large table chunking
CHUNK_THRESHOLD = int(os.getenv("CHUNK_THRESHOLD", "120"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "100"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "15"))

# Structured output: compact regex
GUIDED_REGEX = r"\[\]|\[(\[\d+,\d+\])(,\[\d+,\d+\])*\]"

EXPERIMENT_PLAN = [
    {
        "name": "pubtables_complex_top500",
        "root": PROJECT_ROOT / "Get_500_Tables_from_PubTables" / "JSON_Complex_TOP500_normalized",
        "limit": 500,
        "prompts": None,
    },
    {
        "name": "expert_viewpoint",
        "root": PROJECT_ROOT / "Convert_from_xlsx_to_Json" / "expert_viewpoint_converted_json",
        "limit": 500,
        "prompts": None,
    },
    {
        "name": "maximum_viewpoint",
        "root": PROJECT_ROOT / "Convert_from_xlsx_to_Json" / "maximum_viewpoint_converted_json",
        "limit": 500,
        "prompts": ["zero_max", "reasoning_max"],
    },
    {
        "name": "table_normalization",
        "root": PROJECT_ROOT / "Convert_from_xlsx_to_Json" / "table_normalization_converted_json",
        "limit": 500,
        "prompts": ["zero_min", "reasoning_min"],
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
base_output_dir = Path(OUTPUT_DIR)
base_output_dir.mkdir(parents=True, exist_ok=True)

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
        return {
            k: sanitize_for_prompt(v)
            for k, v in obj.items()
            if k not in LABEL_KEYS and not k.startswith("is_")
        }
    if isinstance(obj, list):
        return [sanitize_for_prompt(x) for x in obj]
    return obj


def to_one_based_coords(coords: List[Dict[str, int]]) -> List[Dict[str, int]]:
    out = set()
    for h in coords:
        try:
            out.add((int(h["row"]) + 1, int(h["col"]) + 1))
        except Exception:
            continue
    return [{"row": r, "col": c} for r, c in sorted(out)]


def coords_to_set(coords: List[Dict[str, int]]) -> set:
    out = set()
    for h in coords or []:
        try:
            out.add((int(h["row"]), int(h["col"])))
        except Exception:
            continue
    return out


def extract_true_coords_from_cells(cells: List[Dict[str, Any]]) -> List[Dict[str, int]]:
    coords = set()
    for cell in cells or []:
        if bool(cell.get("is_column_header")) or bool(cell.get("is_projected_row_header")) or bool(cell.get("is_spanning")):
            for r in cell.get("row_nums", []) or []:
                for c in cell.get("column_nums", []) or []:
                    try:
                        coords.add((int(r), int(c)))
                    except Exception:
                        continue
    return [{"row": r, "col": c} for r, c in sorted(coords)]


def extract_type_coords_from_cells(cells: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, int]]]:
    column_coords = set()
    projected_coords = set()
    spanning_coords = set()

    column_cell_count = 0
    projected_cell_count = 0
    spanning_cell_count = 0

    for cell in cells or []:
        cell_coords = set()
        for r in cell.get("row_nums", []) or []:
            for c in cell.get("column_nums", []) or []:
                try:
                    cell_coords.add((int(r), int(c)))
                except Exception:
                    continue

        if cell.get("is_column_header"):
            column_coords.update(cell_coords)
            column_cell_count += 1

        if cell.get("is_projected_row_header"):
            projected_coords.update(cell_coords)
            projected_cell_count += 1

        if cell.get("is_spanning"):
            spanning_coords.update(cell_coords)
            spanning_cell_count += 1

    return {
        "column_headers": [{"row": r, "col": c} for r, c in sorted(column_coords)],
        "projected_row_headers": [{"row": r, "col": c} for r, c in sorted(projected_coords)],
        "spanning": [{"row": r, "col": c} for r, c in sorted(spanning_coords)],
        "column_header_cell_count": column_cell_count,
        "projected_row_header_cell_count": projected_cell_count,
        "spanning_cell_count": spanning_cell_count,
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


def dataframe_friendly(records: List[Dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for rec in records:
        row = dict(rec)
        for key in [
            "true_headers_raw",
            "true_headers",
            "true_headers_by_type_raw",
            "true_headers_by_type",
            "parsed_headers",
            "raw_request_table",
        ]:
            if key in row and isinstance(row[key], (list, dict)):
                row[key] = json.dumps(row[key], ensure_ascii=False)
        rows.append(row)
    return pd.DataFrame(rows)


def classify_api_error(msg: str) -> str:
    msg = msg.lower()
    if any(x in msg for x in ["maximum context length", "context length", "too many tokens", "token limit"]):
        return "context_length_exceeded"
    if "timeout" in msg or "timed out" in msg:
        return "timeout"
    if "out of memory" in msg or "oom" in msg or "cuda" in msg:
        return "oom"
    if "connection refused" in msg or ("connect" in msg and "error" in msg):
        return "connection_error"
    if "rate limit" in msg or "429" in msg:
        return "rate_limit"
    if "bad request" in msg or "invalid request" in msg or "400" in msg:
        return "bad_request"
    return "api_error"


def rows_bin(n: int) -> str:
    if n <= 10:
        return "<=10"
    if n <= 25:
        return "11-25"
    return ">25"


def count_bin(n: int) -> str:
    if n == 0:
        return "0"
    if n <= 2:
        return "1-2"
    if n <= 5:
        return "3-5"
    return "6+"


def evaluate_coord_sets(true_set: set, pred_set: set) -> Dict[str, Any]:
    support = len(true_set)
    pred_count = len(pred_set)
    tp = len(true_set & pred_set)
    fp = len(pred_set - true_set)
    fn = len(true_set - pred_set)

    if support == 0 and pred_count == 0:
        precision = 1.0
        recall = 1.0
        f1 = 1.0
        jaccard = 1.0
        exact = True
        partial = True
    else:
        precision = tp / pred_count if pred_count else 0.0
        recall = tp / support if support else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        union = len(true_set | pred_set)
        jaccard = tp / union if union else 1.0
        exact = true_set == pred_set
        partial = exact or (support > 0 and (tp / support) >= 0.5)

    return {
        "support": support,
        "pred_count": pred_count,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "jaccard": jaccard,
        "exact_match": exact,
        "partial_match": partial,
        "header_coverage": recall,
    }


def parse_json_array_output(raw_text: str) -> Tuple[bool, List[Dict[str, int]], str]:
    if raw_text is None or not str(raw_text).strip():
        return False, [], "empty_output"

    text = str(raw_text).strip()

    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text, flags=re.IGNORECASE).strip()

    pairs = re.findall(r"\[\s*(\d+)\s*,\s*(\d+)\s*\]", text)
    if pairs:
        seen = set()
        coords = []
        for r_str, c_str in pairs:
            key = (int(r_str), int(c_str))
            if key not in seen:
                seen.add(key)
                coords.append({"row": key[0], "col": key[1]})
        coords.sort(key=lambda x: (x["row"], x["col"]))
        return True, coords, ""

    if re.fullmatch(r"\[\s*\]", text):
        return True, [], ""

    parens = re.findall(r"\(\s*(\d+)\s*,\s*(\d+)\s*\)", text)
    if parens:
        seen = set()
        coords = []
        for r_str, c_str in parens:
            key = (int(r_str), int(c_str))
            if key not in seen:
                seen.add(key)
                coords.append({"row": key[0], "col": key[1]})
        coords.sort(key=lambda x: (x["row"], x["col"]))
        return True, coords, "fallback_paren_format"

    return False, [], "no_parseable_coordinates"


def chunk_cells_table(table_obj: Dict, chunk_start: int, chunk_end: int) -> Tuple[Dict, int]:
    cells = table_obj.get("cells", [])
    chunk_cells = []
    for cell in cells:
        row_nums = cell.get("row_nums", []) or []
        col_nums = cell.get("column_nums", []) or []
        in_chunk = [r for r in row_nums if chunk_start <= r < chunk_end]
        if not in_chunk:
            continue
        new_cell = {k: v for k, v in cell.items() if k != "row_nums"}
        new_cell["row_nums"] = [r - chunk_start for r in in_chunk]
        new_cell["column_nums"] = col_nums
        chunk_cells.append(new_cell)
    result = {k: v for k, v in table_obj.items() if k != "cells"}
    result["cells"] = chunk_cells
    return result, chunk_start


def chunk_matrix_table(table_obj: Dict, chunk_start: int, chunk_end: int) -> Tuple[Dict, int]:
    data_rows = table_obj.get("data", [])
    result = {k: v for k, v in table_obj.items() if k != "data"}
    result["data"] = data_rows[chunk_start:chunk_end]
    return result, chunk_start


def make_chunks(table_json: str, table_rows: int, table_kind: str) -> List[Tuple[str, int]]:
    try:
        table_obj = json.loads(table_json)
    except Exception:
        return [(table_json, 0)]

    chunks = []
    start = 0
    while start < table_rows:
        end = min(start + CHUNK_SIZE, table_rows)
        try:
            if table_kind == "cells":
                chunk_obj, offset = chunk_cells_table(table_obj, start, end)
            elif table_kind == "matrix":
                chunk_obj, offset = chunk_matrix_table(table_obj, start, end)
            else:
                if "cells" in table_obj:
                    chunk_obj, offset = chunk_cells_table(table_obj, start, end)
                elif "data" in table_obj:
                    chunk_obj, offset = chunk_matrix_table(table_obj, start, end)
                else:
                    chunks.append((table_json, 0))
                    break

            chunk_json = json.dumps(chunk_obj, ensure_ascii=False, separators=(",", ":"))
            chunks.append((chunk_json, offset))
        except Exception as e:
            logging.warning(f"Chunk error at start={start}: {e}, sending full table")
            chunks.append((table_json, 0))
            break

        if end >= table_rows:
            break
        start = start + CHUNK_SIZE - CHUNK_OVERLAP

    return chunks if chunks else [(table_json, 0)]


def merge_chunk_predictions(chunk_results: List[Tuple[List[Dict], int]]) -> List[Dict[str, int]]:
    seen = set()
    merged = []
    for headers, row_offset in chunk_results:
        for h in headers:
            orig_row = h["row"] + row_offset
            orig_col = h["col"]
            key = (orig_row, orig_col)
            if key not in seen:
                seen.add(key)
                merged.append({"row": orig_row, "col": orig_col})
    merged.sort(key=lambda x: (x["row"], x["col"]))
    return merged


# =========================
# ASYNC API CALL
# =========================
async def async_api_call(
    client: httpx.AsyncClient,
    model: str,
    messages: List[Dict[str, str]],
) -> Dict[str, Any]:
    url = f"{VLLM_BASE_URL}/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "temperature": TEMPERATURE,
        "guided_regex": GUIDED_REGEX,
        "guided_decoding_backend": "lm-format-enforcer",
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {VLLM_API_KEY}",
    }

    last_error = ""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            t0 = time.time()
            resp = await client.post(url, json=payload, headers=headers)
            resp.raise_for_status()
            data_resp = resp.json()
            duration = time.time() - t0

            raw_response = data_resp["choices"][0]["message"]["content"] or ""
            parse_success, parsed_headers, parse_error = parse_json_array_output(raw_response)

            usage = data_resp.get("usage", {}) or {}
            tokens_info = {
                "prompt": usage.get("prompt_tokens"),
                "completion": usage.get("completion_tokens"),
                "total": usage.get("total_tokens"),
            } if usage else None

            return {
                "api_success": True,
                "raw_response": raw_response,
                "parse_success": parse_success,
                "parsed_headers": parsed_headers,
                "parse_error": parse_error,
                "duration_sec": duration,
                "retry_attempts": attempt,
                "tokens_used": tokens_info,
                "error_type": "",
                "error_message": "",
            }

        except Exception as e:
            last_error = str(e)
            logging.warning(f"Attempt {attempt}/{MAX_RETRIES} failed: {last_error[:120]}")
            if attempt < MAX_RETRIES:
                await asyncio.sleep(min(RETRY_BACKOFF_BASE ** (attempt - 1), 30.0))

    return {
        "api_success": False,
        "raw_response": "",
        "parse_success": False,
        "parsed_headers": [],
        "parse_error": "",
        "duration_sec": None,
        "retry_attempts": MAX_RETRIES,
        "tokens_used": None,
        "error_type": classify_api_error(last_error),
        "error_message": last_error,
    }


# =========================
# MAIN CLASS
# =========================
class ResponseCollector:
    def __init__(self, output_dir: str = OUTPUT_DIR):
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self.base_output_dir = Path(output_dir)
        self.run_dir = self.base_output_dir / f"run_{self.run_id}"
        self.logs_dir = self.run_dir / "logs"
        self.results_dir = self.run_dir / "results"
        self.metrics_dir = self.run_dir / "metrics"
        self.checkpoints_dir = self.run_dir / "checkpoints"

        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)

        self.log_file = self.logs_dir / f"experiment_{self.run_id}.log"
        self._setup_file_logging()

        self.system_prompt = self._load_system_prompt()
        self.prompts = self._load_prompt_configs()
        self.table_map: Dict[str, List[Dict[str, Any]]] = self._load_table_records()

        self.responses: List[Dict[str, Any]] = []
        self.valid_responses: List[Dict[str, Any]] = []
        self.api_failed_requests: List[Dict[str, Any]] = []
        self.parse_failed_requests: List[Dict[str, Any]] = []

        self.start_time: Optional[datetime] = None
        self.completed_count = 0
        self._lock = asyncio.Lock()

        logging.info(f"Run directory: {self.run_dir}")

    def _setup_file_logging(self):
        root = logging.getLogger()
        for h in list(root.handlers):
            if isinstance(h, logging.FileHandler):
                root.removeHandler(h)
                try:
                    h.close()
                except Exception:
                    pass

        file_handler = logging.FileHandler(self.log_file, encoding="utf-8")
        file_handler.setLevel(log_level_value)
        file_handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        root.addHandler(file_handler)
        root.setLevel(log_level_value)

    def _load_system_prompt(self) -> str:
        path = PROMPTS_DIR / "system.txt"
        if not path.exists():
            raise FileNotFoundError(f"System prompt not found: {path}")
        return path.read_text(encoding="utf-8").strip()

    def _load_prompt_configs(self) -> List[Dict[str, Any]]:
        prompt_files = [
            p for p in sorted(PROMPTS_DIR.glob("*.txt"), key=lambda p: p.name)
            if p.name not in EXCLUDE_PROMPT_FILES
        ]
        if not prompt_files:
            raise FileNotFoundError(f"No prompt .txt files found in {PROMPTS_DIR}")

        prompts = []
        for path in prompt_files:
            text = path.read_text(encoding="utf-8").strip()
            if text:
                prompts.append({"name": path.stem, "file": str(path), "user": text})
        return prompts

    def _should_skip_path(self, path: Path) -> bool:
        return (
            any(part in EXCLUDE_DIR_NAMES for part in path.parts)
            or path.name.startswith(EXCLUDE_FILE_PREFIXES)
        )

    def _iter_json_files(self, root: Path):
        seen = set()
        for path in sorted(root.rglob("*.json"), key=lambda p: str(p)):
            if self._should_skip_path(path):
                continue
            resolved = str(path.resolve())
            if resolved not in seen:
                seen.add(resolved)
                yield path

    def _table_dimensions_from_cells(self, cells):
        rows, cols = set(), set()
        for cell in cells or []:
            for r in cell.get("row_nums", []) or []:
                try:
                    rows.add(int(r))
                except Exception:
                    pass
            for c in cell.get("column_nums", []) or []:
                try:
                    cols.add(int(c))
                except Exception:
                    pass
        return (max(rows) + 1 if rows else 0, max(cols) + 1 if cols else 0)

    def _table_dimensions_from_matrix(self, data_rows):
        if not isinstance(data_rows, list):
            return 0, 0
        cols = max((len(r) for r in data_rows if isinstance(r, list)), default=0)
        return len(data_rows), cols

    def _expand_table_records(self, filepath, raw_data, source_group):
        records = []

        def make_record(item, table_index):
            if not isinstance(item, dict):
                return None

            prompt_table = sanitize_for_prompt(item)

            if "cells" in item:
                type_info = extract_type_coords_from_cells(item.get("cells", []))
                true_headers_0 = extract_true_coords_from_cells(item.get("cells", []))
                table_kind = "cells"
                table_rows, table_cols = self._table_dimensions_from_cells(item.get("cells", []))
                has_type_info = True
            elif "data" in item:
                true_headers_0 = extract_true_coords_from_headers(item.get("headers", []))
                table_kind = "matrix"
                table_rows, table_cols = self._table_dimensions_from_matrix(item.get("data", []))
                type_info = {
                    "column_headers": [],
                    "projected_row_headers": [],
                    "spanning": [],
                    "column_header_cell_count": 0,
                    "projected_row_header_cell_count": 0,
                    "spanning_cell_count": 0,
                }
                has_type_info = False
            else:
                true_headers_0 = []
                table_kind = "generic"
                table_rows, table_cols = 0, 0
                type_info = {
                    "column_headers": [],
                    "projected_row_headers": [],
                    "spanning": [],
                    "column_header_cell_count": 0,
                    "projected_row_header_cell_count": 0,
                    "spanning_cell_count": 0,
                }
                has_type_info = False

            table_json = json.dumps(prompt_table, ensure_ascii=False, separators=(",", ":"))
            table_hash = stable_hash(table_json, 12)

            true_headers_by_type_raw = {
                "column_headers": type_info["column_headers"],
                "projected_row_headers": type_info["projected_row_headers"],
                "spanning": type_info["spanning"],
            }
            true_headers_by_type = {
                k: to_one_based_coords(v) for k, v in true_headers_by_type_raw.items()
            }

            return {
                "source_group": source_group,
                "source_file": str(filepath),
                "source_stem": filepath.stem,
                "table_index": table_index,
                "table_kind": table_kind,
                "table_rows": table_rows,
                "table_cols": table_cols,
                "table_rows_bin": rows_bin(table_rows),
                "table_hash": table_hash,
                "table_json": table_json,
                "true_headers_raw": true_headers_0,
                "true_headers": to_one_based_coords(true_headers_0),
                "true_headers_count": len(true_headers_0),
                "true_headers_count_bin": count_bin(len(true_headers_0)),
                "true_headers_by_type_raw": true_headers_by_type_raw,
                "true_headers_by_type": true_headers_by_type,
                "has_type_info": has_type_info,
                "column_header_cell_count": type_info["column_header_cell_count"],
                "projected_row_header_cell_count": type_info["projected_row_header_cell_count"],
                "spanning_cell_count": type_info["spanning_cell_count"],
                "spanning_cell_count_bin": count_bin(type_info["spanning_cell_count"]),
            }

        if isinstance(raw_data, dict):
            rec = make_record(raw_data, 0)
            if rec:
                records.append(rec)
            return records

        if isinstance(raw_data, list):
            extracted_any = False
            for idx, item in enumerate(raw_data):
                if isinstance(item, dict) and ("cells" in item or "data" in item):
                    rec = make_record(item, idx)
                    if rec:
                        records.append(rec)
                        extracted_any = True
            if not extracted_any:
                for idx, item in enumerate(raw_data):
                    if isinstance(item, dict):
                        rec = make_record(item, idx)
                        if rec:
                            records.append(rec)
        return records

    def collect_from_root(self, root, source_group, limit):
        if not root.exists():
            raise FileNotFoundError(f"Table root does not exist: {root}")
        records = []
        for filepath in self._iter_json_files(root):
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                records.extend(self._expand_table_records(filepath, raw, source_group))
                if len(records) >= limit:
                    break
            except json.JSONDecodeError as e:
                logging.error(f"JSON parse error in {filepath}: {e}")
            except Exception as e:
                logging.error(f"Error reading {filepath}: {e}")
        if len(records) < limit:
            raise ValueError(f"Root {root} produced only {len(records)} records, need {limit}.")
        return records[:limit]

    def _load_table_records(self) -> Dict[str, List[Dict[str, Any]]]:
        table_map = {}
        total = 0
        for src in EXPERIMENT_PLAN:
            records = self.collect_from_root(src["root"], src["name"], int(src["limit"]))
            logging.info(f"Loaded {len(records)} table records from {src['name']}: {src['root']}")
            table_map[src["name"]] = records
            total += len(records)
        logging.info(f"Loaded total {total} table records")
        return table_map

    def _prepare_messages(self, prompt_config, table_json, chunk_info: str = "") -> List[Dict[str, str]]:
        user_template = str(prompt_config.get("user", ""))
        user_prompt = (
            user_template
            .replace("{table_json}", table_json)
            .replace("{table_text}", table_json)
            .replace("{table}", table_json)
        )
        if user_prompt == user_template:
            user_prompt = f"{user_template}\n\nTABLE:\n{table_json}"
        if chunk_info:
            user_prompt += f"\n\n[NOTE: {chunk_info}]"
        return [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def _build_request_id(self, prompt_name, table_record):
        base = (
            f"{prompt_name}__{table_record['source_group']}__{table_record['source_stem']}"
            f"__t{table_record['table_index']}__{table_record['table_hash']}"
        )
        return slugify(base)

    def _flatten_metric_prefix(self, metrics: Dict[str, Any], prefix: str) -> Dict[str, Any]:
        out = {}
        for k, v in metrics.items():
            out[f"{prefix}_{k}"] = v
        return out

    def _make_result(self, prompt_idx, prompt_config, table_record, api_result,
                     chunked: bool = False, n_chunks: int = 1) -> Dict[str, Any]:
        prompt_name = str(prompt_config.get("name", f"prompt_{prompt_idx}"))

        if api_result["api_success"] and api_result["parse_success"]:
            pred_set = coords_to_set(api_result["parsed_headers"])
        else:
            pred_set = set()

        true_set = coords_to_set(table_record["true_headers"])
        overall_metrics = evaluate_coord_sets(true_set, pred_set)

        result = {
            "request_id": self._build_request_id(prompt_name, table_record),
            "timestamp": datetime.now().isoformat(),
            "model": MODEL_NAME,
            "prompt_idx": prompt_idx,
            "prompt_name": prompt_name,
            "prompt_file": prompt_config.get("file", ""),
            "source_group": table_record["source_group"],
            "source_file": table_record["source_file"],
            "source_stem": table_record["source_stem"],
            "table_index": table_record["table_index"],
            "table_kind": table_record["table_kind"],
            "table_rows": table_record["table_rows"],
            "table_cols": table_record["table_cols"],
            "table_rows_bin": table_record["table_rows_bin"],
            "table_hash": table_record["table_hash"],
            "true_headers_raw": table_record["true_headers_raw"],
            "true_headers": table_record["true_headers"],
            "true_headers_count": table_record["true_headers_count"],
            "true_headers_count_bin": table_record["true_headers_count_bin"],
            "true_headers_by_type_raw": table_record["true_headers_by_type_raw"],
            "true_headers_by_type": table_record["true_headers_by_type"],
            "has_type_info": table_record["has_type_info"],
            "column_header_cell_count": table_record["column_header_cell_count"],
            "projected_row_header_cell_count": table_record["projected_row_header_cell_count"],
            "spanning_cell_count": table_record["spanning_cell_count"],
            "spanning_cell_count_bin": table_record["spanning_cell_count_bin"],
            "chunked": chunked,
            "n_chunks": n_chunks,
            "api_success": api_result["api_success"],
            "parse_success": api_result["parse_success"],
            "status": (
                "api_failed"
                if not api_result["api_success"]
                else ("ok" if api_result["parse_success"] else "parse_failed")
            ),
            "raw_response": api_result["raw_response"],
            "parsed_headers": api_result["parsed_headers"],
            "parse_error": api_result["parse_error"],
            "error_type": api_result["error_type"],
            "error_message": api_result["error_message"],
            "duration_sec": api_result["duration_sec"],
            "retry_attempts": api_result["retry_attempts"],
            "tokens_used": api_result["tokens_used"],
            "prompt_tokens": None,
            "completion_tokens": None,
            "total_tokens": None,
            "token_efficiency": None,
            "system_prompt_file": "prompts/system.txt",
        }

        tokens_used = api_result.get("tokens_used") or {}
        if tokens_used:
            result["prompt_tokens"] = tokens_used.get("prompt")
            result["completion_tokens"] = tokens_used.get("completion")
            result["total_tokens"] = tokens_used.get("total")

        if result["completion_tokens"] is not None and overall_metrics["f1"] and overall_metrics["f1"] > 0:
            result["token_efficiency"] = result["completion_tokens"] / overall_metrics["f1"]

        result.update(overall_metrics)

        # Type metrics
        for type_name in ["column_headers", "projected_row_headers", "spanning"]:
            if table_record["has_type_info"]:
                type_true_set = coords_to_set(table_record["true_headers_by_type"][type_name])
                type_metrics = evaluate_coord_sets(type_true_set, pred_set)
                result.update(self._flatten_metric_prefix(type_metrics, type_name))
            else:
                for m in ["support", "pred_count", "tp", "fp", "fn", "precision", "recall", "f1", "jaccard", "exact_match", "partial_match", "header_coverage"]:
                    result[f"{type_name}_{m}"] = None

        return result

    async def _process_one(
        self,
        client: httpx.AsyncClient,
        semaphore: asyncio.Semaphore,
        prompt_idx: int,
        prompt_config: Dict[str, Any],
        table_record: Dict[str, Any],
    ) -> Dict[str, Any]:
        needs_chunking = table_record["table_rows"] > CHUNK_THRESHOLD

        if not needs_chunking:
            messages = self._prepare_messages(prompt_config, table_record["table_json"])
            async with semaphore:
                api_result = await async_api_call(client, MODEL_NAME, messages)
            return self._make_result(prompt_idx, prompt_config, table_record, api_result,
                                     chunked=False, n_chunks=1)

        chunks = make_chunks(
            table_record["table_json"],
            table_record["table_rows"],
            table_record["table_kind"],
        )
        logging.info(
            f"Chunking {table_record['source_stem']} "
            f"({table_record['table_rows']} rows) into {len(chunks)} chunks"
        )

        chunk_results: List[Tuple[List[Dict], int]] = []
        total_duration = 0.0
        total_prompt_tokens = 0
        total_completion_tokens = 0
        any_api_failure = False
        all_raw_responses = []
        total_retries = 0

        for chunk_idx, (chunk_json, row_offset) in enumerate(chunks):
            chunk_info = (
                f"This is chunk {chunk_idx+1}/{len(chunks)} of a large table. "
                f"Row 1 in this chunk corresponds to row {row_offset+1} of the full table."
            )
            messages = self._prepare_messages(prompt_config, chunk_json, chunk_info)
            async with semaphore:
                api_result = await async_api_call(client, MODEL_NAME, messages)

            if api_result["api_success"] and api_result["parse_success"]:
                chunk_results.append((api_result["parsed_headers"], row_offset))
            elif not api_result["api_success"]:
                any_api_failure = True
                logging.warning(
                    f"Chunk {chunk_idx+1}/{len(chunks)} of {table_record['source_stem']} "
                    f"api_failed: {api_result['error_message'][:80]}"
                )

            if api_result.get("duration_sec"):
                total_duration += api_result["duration_sec"]
            total_retries += api_result.get("retry_attempts", 1)
            all_raw_responses.append(api_result.get("raw_response", ""))
            if api_result.get("tokens_used"):
                total_prompt_tokens += api_result["tokens_used"].get("prompt", 0) or 0
                total_completion_tokens += api_result["tokens_used"].get("completion", 0) or 0

        merged_headers = merge_chunk_predictions(chunk_results)

        combined_api_result = {
            "api_success": not any_api_failure or bool(chunk_results),
            "parse_success": bool(merged_headers) or not any_api_failure,
            "parsed_headers": merged_headers,
            "parse_error": "" if chunk_results else "all_chunks_failed",
            "raw_response": " | ".join(r for r in all_raw_responses if r)[:500],
            "error_type": "partial_chunk_failure" if any_api_failure and chunk_results else ("api_error" if any_api_failure else ""),
            "error_message": f"{len(chunks)-len(chunk_results)}/{len(chunks)} chunks failed" if any_api_failure else "",
            "duration_sec": total_duration,
            "retry_attempts": total_retries,
            "tokens_used": {
                "prompt": total_prompt_tokens,
                "completion": total_completion_tokens,
                "total": total_prompt_tokens + total_completion_tokens,
            },
        }
        return self._make_result(prompt_idx, prompt_config, table_record, combined_api_result,
                                 chunked=True, n_chunks=len(chunks))

    def _register_result(self, result: Dict[str, Any]):
        if result["api_success"]:
            self.responses.append(result)
            if result["parse_success"]:
                self.valid_responses.append(result)
            else:
                self.parse_failed_requests.append(result)
        else:
            self.api_failed_requests.append(result)
        self.completed_count += 1

    async def _run_tasks(
        self,
        tasks: List[Tuple[int, Dict, Dict]],
        timestamp: str,
    ):
        total_tasks = len(tasks)
        semaphore = asyncio.Semaphore(CONCURRENCY)
        since_checkpoint = 0

        limits = httpx.Limits(max_connections=CONCURRENCY, max_keepalive_connections=CONCURRENCY)

        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT_SECONDS, limits=limits) as client:
            futures = [
                asyncio.ensure_future(self._process_one(client, semaphore, pi, pc, tr))
                for pi, pc, tr in tasks
            ]

            for coro in asyncio.as_completed(futures):
                result = await coro

                async with self._lock:
                    self._register_result(result)
                    since_checkpoint += 1

                    dur_str = f"{result['duration_sec']:.1f}s" if result.get("duration_sec") else "n/a"
                    chunk_str = f" [chunked×{result.get('n_chunks',1)}]" if result.get("chunked") else ""
                    logging.info(
                        f"[{self.completed_count}/{total_tasks}] "
                        f"{result['prompt_name']} | {result['source_stem']}"
                        f"{chunk_str} | status={result['status']} | dur={dur_str}"
                    )

                    if since_checkpoint >= CHECKPOINT_EVERY:
                        self._save_checkpoint(timestamp)
                        since_checkpoint = 0

                    if REQUEST_PAUSE_EVERY > 0 and self.completed_count % REQUEST_PAUSE_EVERY == 0 and self.completed_count < total_tasks:
                        await asyncio.sleep(REQUEST_PAUSE_SECONDS)

        self._save_checkpoint(timestamp)

    def _save_checkpoint(self, timestamp: str):
        path = self.checkpoints_dir / f"checkpoint_{timestamp}.json"
        payload = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "model": MODEL_NAME,
                "completed_count": self.completed_count,
                "responses": len(self.responses),
                "api_failed": len(self.api_failed_requests),
                "parse_failed": len(self.parse_failed_requests),
            },
            "responses": self.responses,
            "api_failed_requests": self.api_failed_requests,
            "parse_failed_requests": self.parse_failed_requests,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        logging.info(f"Checkpoint saved ({self.completed_count} done): {path}")

    def _summarize_table(self, df: pd.DataFrame, group_cols: Optional[List[str]] = None, metric_prefix: str = "") -> pd.DataFrame:
        if df.empty:
            return pd.DataFrame()

        metric_names = [
            "support", "pred_count", "tp", "fp", "fn",
            "precision", "recall", "f1", "jaccard",
            "exact_match", "partial_match", "header_coverage",
        ]

        rows = []
        if group_cols:
            grouped = df.groupby(group_cols, dropna=False, sort=True)
            iterable = grouped
        else:
            iterable = [((), df)]

        for key, sub in iterable:
            row = {}
            if group_cols:
                if len(group_cols) == 1:
                    row[group_cols[0]] = key
                else:
                    for col, val in zip(group_cols, key):
                        row[col] = val

            row["count"] = int(len(sub))
            if "api_success" in sub.columns:
                row["api_success_rate"] = float(pd.to_numeric(sub["api_success"], errors="coerce").mean())
            if "parse_success" in sub.columns:
                row["parse_success_rate"] = float(pd.to_numeric(sub["parse_success"], errors="coerce").mean())

            for metric in metric_names:
                col = f"{metric_prefix}_{metric}" if metric_prefix else metric
                if col in sub.columns:
                    s = pd.to_numeric(sub[col], errors="coerce")
                    if s.notna().any():
                        row[f"{metric}_mean"] = float(s.mean())
                        row[f"{metric}_median"] = float(s.median())

            for col in ["duration_sec", "prompt_tokens", "completion_tokens", "total_tokens", "token_efficiency"]:
                if col in sub.columns:
                    s = pd.to_numeric(sub[col], errors="coerce").dropna()
                    if len(s) > 0:
                        row[f"{col}_mean"] = float(s.mean())
                        row[f"{col}_median"] = float(s.median())
                        if col == "duration_sec":
                            row["duration_sec_p90"] = float(s.quantile(0.90))
                            row["duration_sec_p99"] = float(s.quantile(0.99))

            rows.append(row)

        return pd.DataFrame(rows)

    def _build_metrics_artifacts(self, timestamp: str):
        all_records = self.responses + self.api_failed_requests
        if not all_records:
            return

        df_all = pd.DataFrame(all_records)

        # binning
        if "table_rows" in df_all.columns:
            df_all["table_rows_bin"] = df_all["table_rows"].apply(rows_bin)
        if "true_headers_count" in df_all.columns:
            df_all["true_headers_count_bin"] = df_all["true_headers_count"].apply(count_bin)
        if "spanning_cell_count" in df_all.columns:
            df_all["spanning_cell_count_bin"] = df_all["spanning_cell_count"].apply(count_bin)

        # per-sample exports
        all_json_path = self.results_dir / f"all_results_{timestamp}.json"
        with open(all_json_path, "w", encoding="utf-8") as f:
            json.dump(all_records, f, ensure_ascii=False, indent=2)

        all_csv_path = self.results_dir / f"all_results_{timestamp}.csv"
        dataframe_friendly(all_records).to_csv(all_csv_path, index=False, encoding="utf-8-sig")

        # summaries
        overall_df = self._summarize_table(df_all)
        by_prompt_df = self._summarize_table(df_all, ["prompt_name"])
        by_source_df = self._summarize_table(df_all, ["source_group"])
        by_prompt_source_df = self._summarize_table(df_all, ["prompt_name", "source_group"])
        by_rows_bin_df = self._summarize_table(df_all, ["table_rows_bin"])
        by_headers_bin_df = self._summarize_table(df_all, ["true_headers_count_bin"])
        by_spanning_bin_df = self._summarize_table(df_all, ["spanning_cell_count_bin"])

        # type summaries
        type_frames = []
        for header_type in ["column_headers", "projected_row_headers", "spanning"]:
            col_name = f"{header_type}_precision"
            if col_name in df_all.columns and pd.to_numeric(df_all[col_name], errors="coerce").notna().any():
                tmp = self._summarize_table(df_all, ["prompt_name"], metric_prefix=header_type)
                if not tmp.empty:
                    tmp.insert(1, "header_type", header_type)
                    type_frames.append(tmp)
        by_prompt_type_df = pd.concat(type_frames, ignore_index=True) if type_frames else pd.DataFrame()

        # save csv
        overall_df.to_csv(self.metrics_dir / f"metrics_overall_{timestamp}.csv", index=False, encoding="utf-8-sig")
        by_prompt_df.to_csv(self.metrics_dir / f"metrics_by_prompt_{timestamp}.csv", index=False, encoding="utf-8-sig")
        by_source_df.to_csv(self.metrics_dir / f"metrics_by_source_{timestamp}.csv", index=False, encoding="utf-8-sig")
        by_prompt_source_df.to_csv(self.metrics_dir / f"metrics_by_prompt_source_{timestamp}.csv", index=False, encoding="utf-8-sig")
        by_rows_bin_df.to_csv(self.metrics_dir / f"metrics_by_rows_bin_{timestamp}.csv", index=False, encoding="utf-8-sig")
        by_headers_bin_df.to_csv(self.metrics_dir / f"metrics_by_true_headers_bin_{timestamp}.csv", index=False, encoding="utf-8-sig")
        by_spanning_bin_df.to_csv(self.metrics_dir / f"metrics_by_spanning_bin_{timestamp}.csv", index=False, encoding="utf-8-sig")
        if not by_prompt_type_df.empty:
            by_prompt_type_df.to_csv(self.metrics_dir / f"metrics_by_prompt_type_{timestamp}.csv", index=False, encoding="utf-8-sig")

        # save xlsx
        metrics_xlsx = self.metrics_dir / f"metrics_{timestamp}.xlsx"
        with pd.ExcelWriter(metrics_xlsx, engine="openpyxl") as writer:
            overall_df.to_excel(writer, sheet_name="overall", index=False)
            by_prompt_df.to_excel(writer, sheet_name="by_prompt", index=False)
            by_source_df.to_excel(writer, sheet_name="by_source", index=False)
            by_prompt_source_df.to_excel(writer, sheet_name="by_prompt_source", index=False)
            by_rows_bin_df.to_excel(writer, sheet_name="by_rows_bin", index=False)
            by_headers_bin_df.to_excel(writer, sheet_name="by_true_headers_bin", index=False)
            by_spanning_bin_df.to_excel(writer, sheet_name="by_spanning_bin", index=False)
            if not by_prompt_type_df.empty:
                by_prompt_type_df.to_excel(writer, sheet_name="by_prompt_type", index=False)

        # structured JSON
        metrics_json = {
            "overall": overall_df.to_dict(orient="records"),
            "by_prompt": by_prompt_df.to_dict(orient="records"),
            "by_source": by_source_df.to_dict(orient="records"),
            "by_prompt_source": by_prompt_source_df.to_dict(orient="records"),
            "by_rows_bin": by_rows_bin_df.to_dict(orient="records"),
            "by_true_headers_bin": by_headers_bin_df.to_dict(orient="records"),
            "by_spanning_bin": by_spanning_bin_df.to_dict(orient="records"),
            "by_prompt_type": by_prompt_type_df.to_dict(orient="records") if not by_prompt_type_df.empty else [],
        }
        with open(self.metrics_dir / f"metrics_{timestamp}.json", "w", encoding="utf-8") as f:
            json.dump(metrics_json, f, ensure_ascii=False, indent=2)

        # summary text
        total = len(df_all)
        api_successes = int(df_all["api_success"].sum()) if "api_success" in df_all.columns else 0
        parse_successes = int(df_all["parse_success"].sum()) if "parse_success" in df_all.columns else 0
        valid_rows = df_all[(df_all["api_success"] == True) & (df_all["parse_success"] == True)] if "api_success" in df_all.columns else pd.DataFrame()

        api_success_rate = (api_successes / total * 100) if total else 0.0
        parse_success_rate_of_api = (parse_successes / api_successes * 100) if api_successes else 0.0
        end_to_end_parse_rate = (parse_successes / total * 100) if total else 0.0

        duration_s = pd.to_numeric(df_all["duration_sec"], errors="coerce").dropna() if "duration_sec" in df_all.columns else pd.Series(dtype=float)
        median_dur = float(duration_s.median()) if len(duration_s) else None
        p90_dur = float(duration_s.quantile(0.90)) if len(duration_s) else None
        p99_dur = float(duration_s.quantile(0.99)) if len(duration_s) else None

        overall_row = overall_df.iloc[0].to_dict() if not overall_df.empty else {}

        summary_path = self.metrics_dir / f"metrics_summary_{timestamp}.txt"
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write("METRICS SUMMARY\n")
            f.write("=" * 100 + "\n\n")
            f.write(f"Run dir: {self.run_dir}\n")
            f.write(f"Model: {MODEL_NAME}\n")
            f.write(f"vLLM URL: {VLLM_BASE_URL}\n")
            f.write(f"Total tasks: {total}\n")
            f.write(f"API success rate: {api_success_rate:.2f}%\n")
            f.write(f"Parse success rate (of API successes): {parse_success_rate_of_api:.2f}%\n")
            f.write(f"Parse success rate (of all tasks): {end_to_end_parse_rate:.2f}%\n\n")

            f.write("Core quality metrics\n")
            f.write("-" * 100 + "\n")
            f.write(f"Cell-level Precision:   {overall_row.get('precision_mean', 0):.4f}\n")
            f.write(f"Cell-level Recall:      {overall_row.get('recall_mean', 0):.4f}\n")
            f.write(f"Cell-level F1:          {overall_row.get('f1_mean', 0):.4f}\n")
            f.write(f"Exact Match Rate:       {overall_row.get('exact_match_mean', 0):.4f}\n")
            f.write(f"Partial Match Rate:     {overall_row.get('partial_match_mean', 0):.4f}\n")
            f.write(f"Jaccard Index:          {overall_row.get('jaccard_mean', 0):.4f}\n")
            f.write(f"Header Coverage:        {overall_row.get('header_coverage_mean', 0):.4f}\n")
            f.write(f"Mean completion tokens:  {overall_row.get('completion_tokens_mean', 0):.2f}\n")
            f.write(f"Token efficiency (comp/F1, mean): {overall_row.get('token_efficiency_mean', 0):.2f}\n")
            f.write(f"Response time median:    {median_dur:.2f}s\n" if median_dur is not None else "Response time median:    n/a\n")
            f.write(f"Response time p90:       {p90_dur:.2f}s\n" if p90_dur is not None else "Response time p90:       n/a\n")
            f.write(f"Response time p99:       {p99_dur:.2f}s\n" if p99_dur is not None else "Response time p99:       n/a\n")

            f.write("\nPrompt comparison (overall)\n")
            f.write("-" * 100 + "\n")
            if not by_prompt_df.empty:
                prompt_show = by_prompt_df.sort_values(by=["f1_mean", "api_success_rate"], ascending=False)
                f.write(prompt_show.to_string(index=False))
                f.write("\n")

            f.write("\nType-specific metrics by prompt\n")
            f.write("-" * 100 + "\n")
            if not by_prompt_type_df.empty:
                f.write(by_prompt_type_df.sort_values(by=["header_type", "prompt_name"]).to_string(index=False))
                f.write("\n")

            f.write("\nComplexity breakdown\n")
            f.write("-" * 100 + "\n")
            f.write("By table_rows_bin:\n")
            f.write(by_rows_bin_df.to_string(index=False) + "\n\n")
            f.write("By true_headers_count_bin:\n")
            f.write(by_headers_bin_df.to_string(index=False) + "\n\n")
            f.write("By spanning_cell_count_bin:\n")
            f.write(by_spanning_bin_df.to_string(index=False) + "\n\n")

        logging.info(f"Metrics saved to {self.metrics_dir}")
        logging.info(f"Overall F1: {overall_row.get('f1_mean', 0):.4f}")
        logging.info(f"Exact Match Rate: {overall_row.get('exact_match_mean', 0):.4f}")
        logging.info(f"Parse success rate: {parse_success_rate_of_api:.2f}%")

    def _save_final_results(self, timestamp: str):
        base = self.results_dir
        total = self.completed_count

        for name, d in [
            (f"responses_{timestamp}.json", self.responses),
            (f"api_failed_{timestamp}.json", self.api_failed_requests),
            (f"parse_failed_{timestamp}.json", self.parse_failed_requests),
        ]:
            with open(base / name, "w", encoding="utf-8") as f:
                json.dump(d, f, ensure_ascii=False, indent=2)

        for name, d in [
            (f"responses_{timestamp}.csv", self.responses),
            (f"api_failed_{timestamp}.csv", self.api_failed_requests),
            (f"parse_failed_{timestamp}.csv", self.parse_failed_requests),
        ]:
            if d:
                dataframe_friendly(d).to_csv(base / name, index=False, encoding="utf-8-sig")

        try:
            xlsx_path = base / f"results_{timestamp}.xlsx"
            with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
                dataframe_friendly(self.responses).to_excel(writer, sheet_name="responses", index=False)
                dataframe_friendly(self.api_failed_requests).to_excel(writer, sheet_name="api_failed", index=False)
                dataframe_friendly(self.parse_failed_requests).to_excel(writer, sheet_name="parse_failed", index=False)
        except Exception as e:
            logging.warning(f"Could not save XLSX: {e}")

        api_err_cnt = Counter(r.get("error_type", "unknown") for r in self.api_failed_requests)
        parse_err_cnt = Counter(r.get("parse_error", "unknown") for r in self.parse_failed_requests)
        chunk_cnt = sum(1 for r in self.responses if r.get("chunked"))

        with open(base / f"summary_{timestamp}.txt", "w", encoding="utf-8") as f:
            f.write("EXPERIMENT SUMMARY\n" + "=" * 100 + "\n")
            f.write(f"Run dir:     {self.run_dir}\n")
            f.write(f"Model:       {MODEL_NAME}\n")
            f.write(f"vLLM URL:    {VLLM_BASE_URL}\n")
            f.write(f"Concurrency: {CONCURRENCY}\n")
            f.write(f"Timeout:     {REQUEST_TIMEOUT_SECONDS}s\n")
            f.write(f"Chunk thresh:{CHUNK_THRESHOLD} rows\n")
            f.write(f"Start: {self.start_time.isoformat() if self.start_time else 'n/a'}\n")
            f.write(f"End:   {datetime.now().isoformat()}\n")
            f.write(f"Total processed:     {total}\n")
            f.write(f"API successes:       {len(self.responses)}\n")
            f.write(f"  of which chunked:  {chunk_cnt}\n")
            f.write(f"Valid parsed:        {len(self.valid_responses)}\n")
            f.write(f"API failures:        {len(self.api_failed_requests)}\n")
            f.write(f"Parse failures:      {len(self.parse_failed_requests)}\n")
            if total > 0:
                f.write(f"API success rate: {len(self.responses)/total*100:.2f}%\n")
                f.write(f"Valid parse rate: {len(self.valid_responses)/total*100:.2f}%\n")
            f.write("\nAPI failure reasons:\n")
            for k, v in api_err_cnt.most_common():
                f.write(f"  - {k}: {v}\n")
            f.write("\nParse failure reasons:\n")
            for k, v in parse_err_cnt.most_common():
                f.write(f"  - {k}: {v}\n")

        logging.info(f"Final results saved to {base}")

    async def _run_async(self):
        self.start_time = datetime.now()
        timestamp = self.start_time.strftime("%Y%m%d_%H%M%S")

        tasks = []
        for src in EXPERIMENT_PLAN:
            allowed = src["prompts"]
            for pi, pc in enumerate(self.prompts):
                if allowed is None or pc["name"] in allowed:
                    for tr in self.table_map[src["name"]]:
                        tasks.append((pi, pc, tr))

        total_tasks = len(tasks)
        for src in EXPERIMENT_PLAN:
            allowed = src["prompts"]
            n_p = len(self.prompts) if allowed is None else len([p for p in self.prompts if p["name"] in allowed])
            n_t = len(self.table_map[src["name"]])
            logging.info(f"  {src['name']}: {n_p} prompts × {n_t} tables = {n_p*n_t} requests")
        logging.info(
            f"Starting: {total_tasks} total requests"
            f" | concurrency={CONCURRENCY}"
            f" | timeout={REQUEST_TIMEOUT_SECONDS}s"
            f" | chunk_threshold={CHUNK_THRESHOLD} rows"
            f" | pause_every={REQUEST_PAUSE_EVERY}"
        )

        await self._run_tasks(tasks, timestamp)
        self._save_final_results(timestamp)
        self._build_metrics_artifacts(timestamp)

    def run(self):
        asyncio.run(self._run_async())

    async def _run_retry_async(self, checkpoint_path: str):
        self.start_time = datetime.now()
        timestamp = self.start_time.strftime("%Y%m%d_%H%M%S") + "_retry"

        with open(checkpoint_path, "r", encoding="utf-8") as f:
            ckpt = json.load(f)

        self.responses = ckpt.get("responses", [])
        self.parse_failed_requests = ckpt.get("parse_failed_requests", [])
        self.valid_responses = [r for r in self.responses if r.get("parse_success")]
        failed_records = ckpt.get("api_failed_requests", [])

        logging.info(
            f"Retry mode: checkpoint={checkpoint_path} | "
            f"restoring {len(self.responses)} ok, {len(self.parse_failed_requests)} parse_failed | "
            f"retrying {len(failed_records)} api_failed"
        )

        if not failed_records:
            logging.info("Nothing to retry.")
            return

        prompt_by_name = {pc["name"]: (pi, pc) for pi, pc in enumerate(self.prompts)}
        table_lookup: Dict[Tuple, Dict] = {}
        for src_name, records in self.table_map.items():
            for tr in records:
                key = (tr["source_group"], tr["source_stem"], tr["table_index"])
                table_lookup[key] = tr

        tasks = []
        skipped = 0
        for rec in failed_records:
            pname = rec.get("prompt_name", "")
            if pname not in prompt_by_name:
                logging.warning(f"Prompt '{pname}' not found, skipping {rec.get('request_id')}")
                skipped += 1
                continue
            pi, pc = prompt_by_name[pname]
            key = (rec.get("source_group"), rec.get("source_stem"), rec.get("table_index", 0))
            tr = table_lookup.get(key)
            if tr is None:
                logging.warning(f"Table not found for key {key}, skipping")
                skipped += 1
                continue
            tasks.append((pi, pc, tr))

        logging.info(f"Rebuilt {len(tasks)} retry tasks (skipped {skipped})")
        self.completed_count = len(self.responses) + len(self.parse_failed_requests)
        await self._run_tasks(tasks, timestamp)
        self._save_final_results(timestamp)
        self._build_metrics_artifacts(timestamp)

    def run_retry(self, checkpoint_path: str):
        asyncio.run(self._run_retry_async(checkpoint_path))


# =========================
# ENTRY POINT
# =========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Table header detection experiment")
    parser.add_argument(
        "--retry",
        metavar="CHECKPOINT_PATH",
        default=None,
        help="Path to checkpoint JSON. Retries only the api_failed requests from that checkpoint.",
    )
    args = parser.parse_args()

    collector = ResponseCollector(output_dir=OUTPUT_DIR)

    if args.retry:
        logging.info(f"=== RETRY MODE: {args.retry} ===")
        collector.run_retry(args.retry)
    else:
        collector.run()