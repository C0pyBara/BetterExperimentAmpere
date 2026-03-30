import os
import json
import time
import asyncio
import hashlib
import logging
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import pandas as pd
from openai import AsyncOpenAI

from prompts import PROMPTS


# =========================
# CONFIG (env with sensible defaults)
# =========================
VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://127.0.0.1:9092/v1")
VLLM_API_KEY = os.getenv("VLLM_API_KEY", "EMPTY")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen3-VL-30B-A3B-Thinking")

# Comma-separated roots where JSON tables live in the repo
TABLES_ROOTS = [
    p.strip() for p in os.getenv(
        "TABLES_ROOTS",
        "./Convert_from_xlsx_to_Json,./Get_500_Tables_from_PubTables"
    ).split(",")
    if p.strip()
]

OUTPUT_DIR = os.getenv("OUTPUT_DIR", "results")
LOG_FILE = os.getenv("LOG_FILE", "experiment.log")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))
REQUEST_DELAY = float(os.getenv("REQUEST_DELAY", "0"))

TEMPERATURE = float(os.getenv("TEMPERATURE", "0.0"))

CONCURRENCY = int(os.getenv("CONCURRENCY", "16"))
CHECKPOINT_INTERVAL = int(os.getenv("CHECKPOINT_INTERVAL", "100"))

EXCLUDE_DIR_NAMES = {
    "results", "raw_responses", "test_responses", "__pycache__",
    ".git", ".venv", "venv", "_logs"
}
EXCLUDE_FILE_PREFIXES = (
    "responses_", "failed_", "checkpoint_", "summary_", "parsed_responses_"
)

LABEL_KEYS = {
    "is_column_header",
    "is_projected_row_header",
    "is_metadata",
    "is_row_header",
    "is_spanning",
    "label",
    "labels",
    "gold",
    "answer",
    "answers",
    "target",
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
        logging.StreamHandler()
    ]
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
    """
    Remove label/leakage fields but keep the table in JSON form.
    """
    if isinstance(obj, dict):
        cleaned = {}
        for k, v in obj.items():
            if k in LABEL_KEYS or k.startswith("is_"):
                continue
            cleaned[k] = sanitize_for_prompt(v)
        return cleaned
    if isinstance(obj, list):
        return [sanitize_for_prompt(x) for x in obj]
    return obj


def extract_true_coords_from_cells(cells: List[Dict[str, Any]]) -> List[Dict[str, int]]:
    """
    Expand spans: if a header cell covers multiple row_nums / column_nums,
    include every covered coordinate.
    """
    coords = set()
    for cell in cells:
        if bool(cell.get("is_column_header")) or bool(cell.get("is_projected_row_header")):
            row_nums = cell.get("row_nums", [])
            col_nums = cell.get("column_nums", [])
            for r in row_nums:
                for c in col_nums:
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
    """
    Convert nested fields to JSON strings for CSV/XLSX compatibility.
    """
    rows = []
    for rec in records:
        row = dict(rec)
        for key in [
            "true_headers",
            "parsed_headers",
            "raw_request_table",
            "error_details",
        ]:
            if key in row and isinstance(row[key], (list, dict)):
                row[key] = json.dumps(row[key], ensure_ascii=False)
        rows.append(row)
    return pd.DataFrame(rows)


def classify_api_error(exc: Exception) -> str:
    msg = str(exc).lower()

    if any(x in msg for x in [
        "maximum context length",
        "context length",
        "prompt too long",
        "too many tokens",
        "input too long",
        "token limit",
        "max context"
    ]):
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
# MAIN CLASS
# =========================
class ResponseCollector:
    def __init__(self, json_roots: List[str], output_dir: str = OUTPUT_DIR):
        self.json_roots = json_roots
        self.output_dir = output_dir
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

        self.client = AsyncOpenAI(
            base_url=VLLM_BASE_URL,
            api_key=VLLM_API_KEY,
        )

        self.responses: List[Dict[str, Any]] = []          # API success (parse ok + parse fail)
        self.valid_responses: List[Dict[str, Any]] = []    # parse ok only
        self.api_failed_requests: List[Dict[str, Any]] = []
        self.parse_failed_requests: List[Dict[str, Any]] = []

        self.start_time: Optional[datetime] = None
        self.completed_count = 0
        self._last_checkpoint_count = 0

    def create_system_prompt(self) -> str:
        return (
            "You are a deterministic table header detection system for a research experiment.\n\n"
            "TASK:\n"
            "Identify all header cells in the given table.\n\n"
            "INPUT:\n"
            "- The table is provided as a sanitized JSON object.\n"
            "- Each cell may contain row_nums and column_nums arrays.\n"
            "- A cell can span multiple rows or columns.\n\n"
            "OUTPUT RULES (STRICT):\n"
            "- Output ONLY valid JSON.\n"
            "- Use EXACTLY this schema: {\"headers\": [{\"row\": 0, \"col\": 0}]}\n"
            "- Use 0-based indexing.\n"
            "- If a header cell spans multiple coordinates, include EVERY covered coordinate.\n"
            "- No explanations, no markdown, no extra keys, no code fences.\n"
            "- If there are no header cells, return exactly: {\"headers\": []}\n"
        )

    def load_system_prompt(self) -> str:
        path = Path(__file__).resolve().parent / "prompts" / "system.txt"
        try:
            text = path.read_text(encoding="utf-8").strip()
            if text:
                return text
        except OSError as e:
            logging.warning("Could not read system prompt from %s: %s", path, e)
        return self.create_system_prompt()

    def _should_skip_path(self, path: Path) -> bool:
        if any(part in EXCLUDE_DIR_NAMES for part in path.parts):
            return True
        if path.name.startswith(EXCLUDE_FILE_PREFIXES):
            return True
        return False

    def _iter_json_files(self):
        seen = set()
        for root in self.json_roots:
            root_path = Path(root)
            if not root_path.exists():
                logging.warning(f"Root does not exist: {root_path}")
                continue
            for path in root_path.rglob("*.json"):
                if self._should_skip_path(path):
                    continue
                resolved = str(path.resolve())
                if resolved in seen:
                    continue
                seen.add(resolved)
                yield path

    def _table_dimensions_from_cells(self, cells: List[Dict[str, Any]]) -> Tuple[int, int]:
        rows = set()
        cols = set()
        for cell in cells:
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

    def _expand_table_records(self, filepath: Path, raw_data: Any) -> List[Dict[str, Any]]:
        """
        Supports:
        - dict with "cells"
        - dict with "data"
        - list of table dicts
        """
        records = []

        def make_record(item: Dict[str, Any], table_index: int) -> Optional[Dict[str, Any]]:
            if not isinstance(item, dict):
                return None

            if "cells" in item:
                prompt_table = sanitize_for_prompt(item)
                true_headers = extract_true_coords_from_cells(item.get("cells", []))
                table_kind = "cells"
                rows, cols = self._table_dimensions_from_cells(item.get("cells", []))
            elif "data" in item:
                prompt_table = sanitize_for_prompt(item)
                true_headers = extract_true_coords_from_headers(item.get("headers", []))
                table_kind = "matrix"
                data_rows = item.get("data") or []
                rows = len(data_rows)
                if not data_rows:
                    cols = 0
                else:
                    first = data_rows[0]
                    cols = len(first) if isinstance(first, (list, tuple)) else 0
            else:
                prompt_table = sanitize_for_prompt(item)
                true_headers = []
                table_kind = "generic"
                rows, cols = 0, 0

            table_json = json.dumps(prompt_table, ensure_ascii=False, separators=(",", ":"))
            table_hash = stable_hash(table_json, 12)

            return {
                "source_file": str(filepath),
                "source_stem": filepath.stem,
                "table_index": table_index,
                "table_kind": table_kind,
                "table_rows": rows,
                "table_cols": cols,
                "table_hash": table_hash,
                "table_json": table_json,
                "true_headers": true_headers,
                "true_headers_count": len(true_headers),
            }

        if isinstance(raw_data, dict):
            rec = make_record(raw_data, 0)
            if rec is not None:
                records.append(rec)
            return records

        if isinstance(raw_data, list):
            # Common case in this repo: a list of table objects.
            extracted_any = False
            for idx, item in enumerate(raw_data):
                if isinstance(item, dict) and ("cells" in item or "data" in item):
                    rec = make_record(item, idx)
                    if rec is not None:
                        records.append(rec)
                        extracted_any = True
            if extracted_any:
                return records

            # Fallback: treat list items that are dicts as separate objects
            for idx, item in enumerate(raw_data):
                if isinstance(item, dict):
                    rec = make_record(item, idx)
                    if rec is not None:
                        records.append(rec)

        return records

    def load_json_tables(self) -> List[Dict[str, Any]]:
        tables = []
        for filepath in self._iter_json_files():
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    raw = json.load(f)
                expanded = self._expand_table_records(filepath, raw)
                tables.extend(expanded)
            except json.JSONDecodeError as e:
                logging.error(f"JSON parse error in {filepath}: {e}")
            except Exception as e:
                logging.error(f"Error reading {filepath}: {e}")

        logging.info(f"Loaded {len(tables)} table records")
        return tables

    def prepare_messages(self, prompt_config: Dict[str, Any], table_json: str) -> List[Dict[str, str]]:
    
        # грузим system prompt из файла
        system_prompt = self.load_system_prompt()

        # добавляем доп. system инструкции (если есть)
        extra_system = prompt_config.get("system")
        if extra_system:
            if isinstance(extra_system, list):
                extra_text = "\n".join(str(x) for x in extra_system)
            else:
                extra_text = str(extra_system)

            system_prompt = f"{system_prompt}\n\n{extra_text}"

        user_template = str(prompt_config.get("user", ""))

        # универсальная подстановка
        user_prompt = (
            user_template
            .replace("{table_json}", table_json)
            .replace("{table_text}", table_json)
            .replace("{table}", table_json)
        )

        if user_prompt == user_template:
            user_prompt = f"{user_template}\n\nTABLE_JSON:\n{table_json}"

        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

    def parse_model_output(self, raw_text: str) -> Tuple[bool, List[Dict[str, int]], str]:
        """
        Strict JSON parser with a small cleanup for code fences.
        """
        if not raw_text:
            return False, [], "empty_output"

        text = raw_text.strip()

        # Remove ```json ... ``` fences if present
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
            text = re.sub(r"\s*```$", "", text, flags=re.IGNORECASE).strip()

        candidate = text

        # Try to isolate the first JSON object if model added text around it
        if not candidate.startswith("{"):
            start = candidate.find("{")
            end = candidate.rfind("}")
            if start != -1 and end != -1 and end > start:
                candidate = candidate[start:end + 1].strip()

        try:
            obj = json.loads(candidate)
        except Exception as e:
            return False, [], f"invalid_json: {e}"

        if not isinstance(obj, dict) or "headers" not in obj or not isinstance(obj["headers"], list):
            return False, [], "missing_or_invalid_headers_key"

        coords = []
        for h in obj["headers"]:
            if isinstance(h, dict) and "row" in h and "col" in h:
                try:
                    coords.append((int(h["row"]), int(h["col"])))
                except Exception:
                    continue

        coords = sorted(set(coords))
        parsed_headers = [{"row": r, "col": c} for r, c in coords]

        return True, parsed_headers, ""

    async def make_api_call(self, model: str, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """
        API call with retry logic.
        No input token precheck and no truncation.
        """
        last_exc = None

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                start_time = time.time()

                # Try JSON mode first, then fallback.
                try:
                    completion = await self.client.chat.completions.create(
                        model=model,
                        messages=messages,
                        temperature=TEMPERATURE,
                        response_format={"type": "json_object"},
                    )
                except Exception:
                    completion = await self.client.chat.completions.create(
                        model=model,
                        messages=messages,
                        temperature=TEMPERATURE,
                    )

                duration = time.time() - start_time
                raw_response = completion.choices[0].message.content or ""

                parse_success, parsed_headers, parse_error = self.parse_model_output(raw_response)

                tokens_used = None
                if hasattr(completion, "usage") and completion.usage:
                    tokens_used = {
                        "prompt": getattr(completion.usage, "prompt_tokens", None),
                        "completion": getattr(completion.usage, "completion_tokens", None),
                        "total": getattr(completion.usage, "total_tokens", None),
                    }

                return {
                    "api_success": True,
                    "raw_response": raw_response,
                    "parse_success": parse_success,
                    "parsed_headers": parsed_headers,
                    "parse_error": parse_error,
                    "duration_sec": duration,
                    "retry_attempts": attempt,
                    "tokens_used": tokens_used,
                    "error_type": "",
                    "error_message": "",
                }

            except Exception as e:
                last_exc = e
                logging.warning(
                    f"Attempt {attempt}/{MAX_RETRIES} failed for {model}: {e}"
                )
                if attempt < MAX_RETRIES:
                    await asyncio.sleep(min(2 ** (attempt - 1), 30))

        return {
            "api_success": False,
            "raw_response": "",
            "parse_success": False,
            "parsed_headers": [],
            "parse_error": "",
            "duration_sec": None,
            "retry_attempts": MAX_RETRIES,
            "tokens_used": None,
            "error_type": classify_api_error(last_exc) if last_exc else "api_error",
            "error_message": str(last_exc) if last_exc else "Unknown error",
        }

    def build_request_id(self, prompt_name: str, table_record: Dict[str, Any]) -> str:
        base = f"{prompt_name}__{table_record['source_stem']}__t{table_record['table_index']}__{table_record['table_hash']}"
        return slugify(base)

    async def process_one(
        self,
        prompt_idx: int,
        prompt_config: Dict[str, Any],
        table_record: Dict[str, Any],
    ) -> Dict[str, Any]:
        prompt_name = str(prompt_config.get("name", f"prompt_{prompt_idx}"))
        request_id = self.build_request_id(prompt_name, table_record)

        messages = self.prepare_messages(prompt_config, table_record["table_json"])
        api_result = await self.make_api_call(MODEL_NAME, messages)

        record = {
            "request_id": request_id,
            "timestamp": datetime.now().isoformat(),
            "model": MODEL_NAME,
            "prompt_idx": prompt_idx,
            "prompt_name": prompt_name,
            "source_file": table_record["source_file"],
            "source_stem": table_record["source_stem"],
            "table_index": table_record["table_index"],
            "table_kind": table_record["table_kind"],
            "table_rows": table_record["table_rows"],
            "table_cols": table_record["table_cols"],
            "table_hash": table_record["table_hash"],
            "true_headers": table_record["true_headers"],
            "true_headers_count": table_record["true_headers_count"],
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
        }

        return record

    async def worker(self, queue: asyncio.Queue, timestamp: str):
        while True:
            job = await queue.get()
            try:
                if job is None:
                    return

                record = await self.process_one(
                    prompt_idx=job["prompt_idx"],
                    prompt_config=job["prompt_config"],
                    table_record=job["table_record"],
                )

                # Store records
                if record["api_success"]:
                    self.responses.append(record)
                    if record["parse_success"]:
                        self.valid_responses.append(record)
                    else:
                        self.parse_failed_requests.append(record)
                else:
                    self.api_failed_requests.append(record)

                self.completed_count += 1

                if self.completed_count - self._last_checkpoint_count >= CHECKPOINT_INTERVAL:
                    self._last_checkpoint_count = self.completed_count
                    self.save_checkpoint(timestamp)

                if REQUEST_DELAY > 0:
                    await asyncio.sleep(REQUEST_DELAY)

            finally:
                queue.task_done()

    def save_checkpoint(self, timestamp: str):
        checkpoint_path = Path(self.output_dir) / f"checkpoint_{timestamp}.json"
        payload = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "model": MODEL_NAME,
                "completed_count": self.completed_count,
                "responses": len(self.responses),
                "valid_responses": len(self.valid_responses),
                "api_failed": len(self.api_failed_requests),
                "parse_failed": len(self.parse_failed_requests),
            },
            "responses": self.responses,
            "api_failed_requests": self.api_failed_requests,
            "parse_failed_requests": self.parse_failed_requests,
        }
        with open(checkpoint_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        logging.info(f"Checkpoint saved: {checkpoint_path}")

    def _save_records_json_csv(self, records: List[Dict[str, Any]], stem: str, timestamp: str):
        json_path = Path(self.output_dir) / f"{stem}_{timestamp}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)

        if records:
            csv_path = Path(self.output_dir) / f"{stem}_{timestamp}.csv"
            df = dataframe_friendly(records)
            df.to_csv(csv_path, index=False, encoding="utf-8-sig")
            return json_path, csv_path

        return json_path, None

    def _save_records_xlsx(self, timestamp: str):
        xlsx_path = Path(self.output_dir) / f"results_{timestamp}.xlsx"
        try:
            with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
                dataframe_friendly(self.responses).to_excel(writer, sheet_name="responses", index=False)
                dataframe_friendly(self.valid_responses).to_excel(writer, sheet_name="parsed_only", index=False)
                dataframe_friendly(self.api_failed_requests).to_excel(writer, sheet_name="api_failed", index=False)
                dataframe_friendly(self.parse_failed_requests).to_excel(writer, sheet_name="parse_failed", index=False)
            return xlsx_path
        except Exception as e:
            logging.warning(f"Could not save XLSX: {e}")
            return None

    def save_final_results(self, timestamp: str):
        total = self.completed_count

        # Main records
        responses_json, responses_csv = self._save_records_json_csv(self.responses, "responses", timestamp)
        api_failed_json, api_failed_csv = self._save_records_json_csv(self.api_failed_requests, "api_failed", timestamp)
        parse_failed_json, parse_failed_csv = self._save_records_json_csv(self.parse_failed_requests, "parse_failed", timestamp)

        xlsx_path = self._save_records_xlsx(timestamp)

        # Summary
        summary_path = Path(self.output_dir) / f"summary_{timestamp}.txt"
        api_error_counter = Counter(r.get("error_type", "unknown") for r in self.api_failed_requests)
        parse_error_counter = Counter(r.get("parse_error", "unknown") for r in self.parse_failed_requests)

        with open(summary_path, "w", encoding="utf-8") as f:
            f.write("EXPERIMENT SUMMARY\n")
            f.write("=" * 80 + "\n")
            f.write(f"Model: {MODEL_NAME}\n")
            f.write(f"vLLM base URL: {VLLM_BASE_URL}\n")
            f.write(f"Start time: {self.start_time.isoformat() if self.start_time else 'n/a'}\n")
            f.write(f"End time: {datetime.now().isoformat()}\n")
            f.write(f"Total processed: {total}\n")
            f.write(f"API successes: {len(self.responses)}\n")
            f.write(f"Valid parsed outputs: {len(self.valid_responses)}\n")
            f.write(f"API failures: {len(self.api_failed_requests)}\n")
            f.write(f"Parse failures: {len(self.parse_failed_requests)}\n")
            if total > 0:
                f.write(f"API success rate: {len(self.responses) / total * 100:.2f}%\n")
                f.write(f"Valid parse rate: {len(self.valid_responses) / total * 100:.2f}%\n")
            f.write("\nAPI failure reasons:\n")
            for k, v in api_error_counter.most_common():
                f.write(f"  - {k}: {v}\n")
            f.write("\nParse failure reasons:\n")
            for k, v in parse_error_counter.most_common():
                f.write(f"  - {k}: {v}\n")
            f.write("\nSample failed request ids:\n")
            for rec in (self.api_failed_requests[:20] + self.parse_failed_requests[:20]):
                f.write(f"  - {rec.get('request_id')} | {rec.get('status')} | {rec.get('error_type') or rec.get('parse_error')}\n")

        all_payload = {
            "metadata": {
                "model": MODEL_NAME,
                "vllm_base_url": VLLM_BASE_URL,
                "start_time": self.start_time.isoformat() if self.start_time else None,
                "end_time": datetime.now().isoformat(),
                "total_processed": total,
                "api_successes": len(self.responses),
                "valid_parsed_outputs": len(self.valid_responses),
                "api_failures": len(self.api_failed_requests),
                "parse_failures": len(self.parse_failed_requests),
            },
            "responses": self.responses,
            "api_failed_requests": self.api_failed_requests,
            "parse_failed_requests": self.parse_failed_requests,
        }
        all_json_path = Path(self.output_dir) / f"all_results_{timestamp}.json"
        with open(all_json_path, "w", encoding="utf-8") as f:
            json.dump(all_payload, f, ensure_ascii=False, indent=2)

        logging.info(f"Saved: {responses_json}")
        if responses_csv:
            logging.info(f"Saved: {responses_csv}")
        logging.info(f"Saved: {api_failed_json}")
        if api_failed_csv:
            logging.info(f"Saved: {api_failed_csv}")
        logging.info(f"Saved: {parse_failed_json}")
        if parse_failed_csv:
            logging.info(f"Saved: {parse_failed_csv}")
        if xlsx_path:
            logging.info(f"Saved: {xlsx_path}")
        logging.info(f"Saved: {summary_path}")
        logging.info(f"Saved: {all_json_path}")

    async def run(self, prompts: List[Dict[str, Any]]):
        self.start_time = datetime.now()

        tables = self.load_json_tables()
        if not tables:
            logging.error("No table files found.")
            return

        total_tasks = len(tables) * len(prompts)
        timestamp = self.start_time.strftime("%Y%m%d_%H%M%S")

        logging.info(
            f"Starting experiment: {len(prompts)} prompts × {len(tables)} tables = {total_tasks} requests"
        )

        queue: asyncio.Queue = asyncio.Queue()

        for prompt_idx, prompt_config in enumerate(prompts):
            for table_record in tables:
                await queue.put({
                    "prompt_idx": prompt_idx,
                    "prompt_config": prompt_config,
                    "table_record": table_record,
                })

        # Sentinels for workers
        for _ in range(CONCURRENCY):
            await queue.put(None)

        workers = [asyncio.create_task(self.worker(queue, timestamp)) for _ in range(CONCURRENCY)]

        await queue.join()
        for w in workers:
            await w

        self.save_final_results(timestamp)


if __name__ == "__main__":
    collector = ResponseCollector(json_roots=TABLES_ROOTS, output_dir=OUTPUT_DIR)
    asyncio.run(collector.run(PROMPTS))