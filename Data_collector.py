import os
import json
import time
import logging
import re
import hashlib
import asyncio
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
LOG_FILE = os.getenv("LOG_FILE", "experiment.log")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.0"))

CONCURRENCY = int(os.getenv("CONCURRENCY", "5"))
CHECKPOINT_EVERY = int(os.getenv("CHECKPOINT_EVERY", "5"))

REQUEST_TIMEOUT_SECONDS = float(os.getenv("REQUEST_TIMEOUT_SECONDS", "600.0"))
RETRY_BACKOFF_BASE = float(os.getenv("RETRY_BACKOFF_BASE", "2.0"))

# Structured output schema: array of [row, col] integer pairs
# vLLM enforces this via guided decoding — model CANNOT output anything else
GUIDED_JSON_SCHEMA = {
    "type": "array",
    "items": {
        "type": "array",
        "items": {"type": "integer"},
        "minItems": 2,
        "maxItems": 2,
    },
}

TABLE_SOURCES = [
    {
        "name": "pubtables_complex_top500",
        "root": PROJECT_ROOT / "Get_500_Tables_from_PubTables" / "JSON_Complex_TOP500_normalized",
        "limit": 500,
    },
    {
        "name": "expert_viewpoint",
        "root": PROJECT_ROOT / "Convert_from_xlsx_to_Json" / "expert_viewpoint_converted_json",
        "limit": 500,
    },
    {
        "name": "maximum_viewpoint",
        "root": PROJECT_ROOT / "Convert_from_xlsx_to_Json" / "maximum_viewpoint_converted_json",
        "limit": 500,
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
    handlers=[
        logging.FileHandler(LOG_FILE, encoding="utf-8"),
        logging.StreamHandler(),
    ],
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
    out = []
    for h in coords:
        try:
            out.append({"row": int(h["row"]) + 1, "col": int(h["col"]) + 1})
        except Exception:
            continue
    out = sorted({(x["row"], x["col"]) for x in out})
    return [{"row": r, "col": c} for r, c in out]


def extract_true_coords_from_cells(cells: List[Dict[str, Any]]) -> List[Dict[str, int]]:
    coords = set()
    for cell in cells or []:
        if bool(cell.get("is_column_header")) or bool(cell.get("is_projected_row_header")):
            for r in cell.get("row_nums", []) or []:
                for c in cell.get("column_nums", []) or []:
                    try:
                        coords.add((int(r), int(c)))
                    except Exception:
                        continue
    return [{"row": r, "col": c} for r, c in sorted(coords)]


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
        for key in ["true_headers_raw", "true_headers", "parsed_headers", "raw_request_table"]:
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
    if "bad request" in msg or "invalid request" in msg:
        return "bad_request"
    return "api_error"


# =========================
# JSON OUTPUT PARSER
# =========================
def parse_json_array_output(raw_text: str) -> Tuple[bool, List[Dict[str, int]], str]:
    """
    Parse structured output: [[row, col], [row, col], ...]
    With guided_json enabled, the model is forced to output valid JSON —
    but we keep fallbacks for safety.
    """
    if not raw_text or not raw_text.strip():
        return True, [], ""  # empty = no headers, valid

    text = raw_text.strip()

    # Strip markdown fences (just in case)
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text, flags=re.IGNORECASE).strip()

    # Primary: direct JSON parse
    try:
        parsed = json.loads(text)
        if not isinstance(parsed, list):
            return False, [], f"expected list, got {type(parsed).__name__}"
        coords = []
        seen = set()
        for item in parsed:
            if not isinstance(item, list) or len(item) != 2:
                return False, [], f"item not [row,col]: {item}"
            r, c = item
            if not isinstance(r, int) or not isinstance(c, int):
                return False, [], f"non-integer coords: {item}"
            key = (r, c)
            if key not in seen:
                seen.add(key)
                coords.append({"row": r, "col": c})
        coords.sort(key=lambda x: (x["row"], x["col"]))
        return True, coords, ""
    except json.JSONDecodeError:
        pass

    # Fallback 1: extract embedded JSON array
    match = re.search(r"\[\s*\[.*?\]\s*\]", text, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group())
            if isinstance(parsed, list):
                coords = []
                seen = set()
                for item in parsed:
                    if isinstance(item, list) and len(item) == 2:
                        r, c = item[0], item[1]
                        if isinstance(r, int) and isinstance(c, int):
                            key = (r, c)
                            if key not in seen:
                                seen.add(key)
                                coords.append({"row": r, "col": c})
                coords.sort(key=lambda x: (x["row"], x["col"]))
                return True, coords, ""
        except json.JSONDecodeError:
            pass

    # Fallback 2: old (row,col) format for backwards compatibility
    matches = re.findall(r"\(\s*(\d+)\s*,\s*(\d+)\s*\)", text)
    if matches:
        coords = sorted({(int(r), int(c)) for r, c in matches})
        return True, [{"row": r, "col": c} for r, c in coords], "fallback_paren_format"

    return False, [], "no_parseable_coordinates"


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
        # Structured output: vLLM guided decoding forces valid JSON schema
        # Eliminates all reasoning text, preambles, and format errors
        "guided_json": GUIDED_JSON_SCHEMA,
        "guided_decoding_backend": "outlines",  # alternative: "lm-format-enforcer"
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
            data = resp.json()
            duration = time.time() - t0

            raw_response = data["choices"][0]["message"]["content"] or ""
            parse_success, parsed_headers, parse_error = parse_json_array_output(raw_response)

            usage = data.get("usage", {})
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
            logging.warning(f"Attempt {attempt}/{MAX_RETRIES} failed: {last_error}")
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
        self.output_dir = output_dir
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

        self.system_prompt = self._load_system_prompt()
        self.prompts = self._load_prompt_configs()
        self.tables = self._load_table_records()

        self.responses: List[Dict[str, Any]] = []
        self.valid_responses: List[Dict[str, Any]] = []
        self.api_failed_requests: List[Dict[str, Any]] = []
        self.parse_failed_requests: List[Dict[str, Any]] = []

        self.start_time: Optional[datetime] = None
        self.completed_count = 0
        self._lock = asyncio.Lock()

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
                try: rows.add(int(r))
                except: pass
            for c in cell.get("column_nums", []) or []:
                try: cols.add(int(c))
                except: pass
        return (max(rows) + 1 if rows else 0, max(cols) + 1 if cols else 0)

    def _table_dimensions_from_matrix(self, data_rows):
        if not isinstance(data_rows, list):
            return 0, 0
        rows = len(data_rows)
        cols = max((len(r) for r in data_rows if isinstance(r, list)), default=0)
        return rows, cols

    def _expand_table_records(self, filepath, raw_data, source_group):
        records = []

        def make_record(item, table_index):
            if not isinstance(item, dict):
                return None
            prompt_table = sanitize_for_prompt(item)
            if "cells" in item:
                true_headers_0 = extract_true_coords_from_cells(item.get("cells", []))
                table_kind = "cells"
                table_rows, table_cols = self._table_dimensions_from_cells(item.get("cells", []))
            elif "data" in item:
                true_headers_0 = extract_true_coords_from_headers(item.get("headers", []))
                table_kind = "matrix"
                table_rows, table_cols = self._table_dimensions_from_matrix(item.get("data", []))
            else:
                true_headers_0 = []
                table_kind = "generic"
                table_rows, table_cols = 0, 0

            table_json = json.dumps(prompt_table, ensure_ascii=False, separators=(",", ":"))
            table_hash = stable_hash(table_json, 12)
            return {
                "source_group": source_group,
                "source_file": str(filepath),
                "source_stem": filepath.stem,
                "table_index": table_index,
                "table_kind": table_kind,
                "table_rows": table_rows,
                "table_cols": table_cols,
                "table_hash": table_hash,
                "table_json": table_json,
                "true_headers_raw": true_headers_0,
                "true_headers": to_one_based_coords(true_headers_0),
                "true_headers_count": len(true_headers_0),
            }

        if isinstance(raw_data, dict):
            rec = make_record(raw_data, 0)
            if rec:
                records.append(rec)
        elif isinstance(raw_data, list):
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

    def _load_table_records(self):
        all_records = []
        for src in TABLE_SOURCES:
            records = self.collect_from_root(src["root"], src["name"], int(src["limit"]))
            logging.info(f"Loaded {len(records)} table records from {src['name']}: {src['root']}")
            all_records.extend(records)
        logging.info(f"Loaded total {len(all_records)} table records")
        return all_records

    def _prepare_messages(self, prompt_config, table_json):
        user_template = str(prompt_config.get("user", ""))
        user_prompt = (
            user_template
            .replace("{table_json}", table_json)
            .replace("{table_text}", table_json)
            .replace("{table}", table_json)
        )
        if user_prompt == user_template:
            user_prompt = f"{user_template}\n\nTABLE:\n{table_json}"
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

    def _make_result(self, prompt_idx, prompt_config, table_record, api_result):
        prompt_name = str(prompt_config.get("name", f"prompt_{prompt_idx}"))
        return {
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
            "table_hash": table_record["table_hash"],
            "true_headers_raw": table_record["true_headers_raw"],
            "true_headers": table_record["true_headers"],
            "true_headers_count": table_record["true_headers_count"],
            "api_success": api_result["api_success"],
            "parse_success": api_result["parse_success"],
            "status": (
                "api_failed" if not api_result["api_success"]
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
            "system_prompt_file": "prompts/system.txt",
        }

    def _save_checkpoint(self, timestamp: str):
        path = Path(self.output_dir) / f"checkpoint_{timestamp}.json"
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

    def _save_final_results(self, timestamp: str):
        base = Path(self.output_dir)
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

        api_error_counter = Counter(r.get("error_type", "unknown") for r in self.api_failed_requests)
        parse_error_counter = Counter(r.get("parse_error", "unknown") for r in self.parse_failed_requests)

        with open(base / f"summary_{timestamp}.txt", "w", encoding="utf-8") as f:
            f.write("EXPERIMENT SUMMARY\n" + "=" * 80 + "\n")
            f.write(f"Model: {MODEL_NAME}\n")
            f.write(f"vLLM base URL: {VLLM_BASE_URL}\n")
            f.write(f"Concurrency: {CONCURRENCY}\n")
            f.write(f"Timeout: {REQUEST_TIMEOUT_SECONDS}s\n")
            f.write(f"Start: {self.start_time.isoformat() if self.start_time else 'n/a'}\n")
            f.write(f"End: {datetime.now().isoformat()}\n")
            f.write(f"Total processed: {total}\n")
            f.write(f"API successes: {len(self.responses)}\n")
            f.write(f"Valid parsed: {len(self.valid_responses)}\n")
            f.write(f"API failures: {len(self.api_failed_requests)}\n")
            f.write(f"Parse failures: {len(self.parse_failed_requests)}\n")
            if total > 0:
                f.write(f"API success rate: {len(self.responses)/total*100:.2f}%\n")
                f.write(f"Valid parse rate: {len(self.valid_responses)/total*100:.2f}%\n")
            f.write("\nAPI failure reasons:\n")
            for k, v in api_error_counter.most_common():
                f.write(f"  - {k}: {v}\n")
            f.write("\nParse failure reasons:\n")
            for k, v in parse_error_counter.most_common():
                f.write(f"  - {k}: {v}\n")

        logging.info(f"Final results saved to {base}")

    async def _run_async(self):
        self.start_time = datetime.now()
        timestamp = self.start_time.strftime("%Y%m%d_%H%M%S")

        tasks = [
            (pi, pc, tr)
            for pi, pc in enumerate(self.prompts)
            for tr in self.tables
        ]
        total_tasks = len(tasks)
        logging.info(
            f"Starting: {len(self.prompts)} prompts × {len(self.tables)} tables = {total_tasks} requests"
            f" | concurrency={CONCURRENCY} | timeout={REQUEST_TIMEOUT_SECONDS}s"
        )

        semaphore = asyncio.Semaphore(CONCURRENCY)
        since_last_checkpoint = 0

        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT_SECONDS) as client:

            async def process_one(prompt_idx, prompt_config, table_record):
                messages = self._prepare_messages(prompt_config, table_record["table_json"])
                async with semaphore:
                    api_result = await async_api_call(client, MODEL_NAME, messages)
                return self._make_result(prompt_idx, prompt_config, table_record, api_result)

            futures = [
                asyncio.ensure_future(process_one(pi, pc, tr))
                for pi, pc, tr in tasks
            ]

            for coro in asyncio.as_completed(futures):
                result = await coro

                async with self._lock:
                    if result["api_success"]:
                        self.responses.append(result)
                        if result["parse_success"]:
                            self.valid_responses.append(result)
                        else:
                            self.parse_failed_requests.append(result)
                    else:
                        self.api_failed_requests.append(result)

                    self.completed_count += 1
                    since_last_checkpoint += 1

                    dur_str = f"{result['duration_sec']:.1f}s" if result.get("duration_sec") else "n/a"
                    logging.info(
                        f"[{self.completed_count}/{total_tasks}] "
                        f"{result['prompt_name']} | {result['source_stem']} | "
                        f"status={result['status']} | dur={dur_str}"
                    )

                    if since_last_checkpoint >= CHECKPOINT_EVERY:
                        self._save_checkpoint(timestamp)
                        since_last_checkpoint = 0

        self._save_checkpoint(timestamp)
        self._save_final_results(timestamp)

    def run(self):
        asyncio.run(self._run_async())


if __name__ == "__main__":
    collector = ResponseCollector(output_dir=OUTPUT_DIR)
    collector.run()