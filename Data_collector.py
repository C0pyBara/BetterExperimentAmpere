import os
import json
import time
import logging
from pathlib import Path
from typing import List, Dict

from openai import OpenAI

# =========================
# НАСТРОЙКИ (минимум)
# =========================
BASE_URL = "http://localhost:9092/v1"
MODEL_NAME = "Qwen/Qwen3-VL-30B-A3B-Thinking"

TEMPERATURE = 0.0
MAX_TOKENS = 2048
MAX_RETRIES = 3

INPUT_DIR = "./json"       # папка с таблицами
OUTPUT_DIR = "./results"   # куда сохраняем

# =========================
# ЛОГИ
# =========================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

# =========================
# КЛИЕНТ vLLM
# =========================
client = OpenAI(
    base_url=BASE_URL,
    api_key="EMPTY"
)

# =========================
# ОСНОВНОЙ КЛАСС
# =========================
class ResponseCollector:
    def __init__(self):
        Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    # -------------------------
    # API вызов
    # -------------------------
    def make_api_call(self, messages: List[Dict]) -> Dict:
        for attempt in range(MAX_RETRIES):
            try:
                start_time = time.time()

                response = client.chat.completions.create(
                    model=MODEL_NAME,
                    messages=messages,
                    temperature=TEMPERATURE,
                    max_tokens=MAX_TOKENS
                )

                text = response.choices[0].message.content

                return {
                    "success": True,
                    "response": text,
                    "duration": time.time() - start_time,
                    "attempt": attempt + 1
                }

            except Exception as e:
                logging.warning(f"Ошибка (попытка {attempt+1}): {e}")
                time.sleep(2 ** attempt)

        return {"success": False, "error": "Max retries exceeded"}

    # -------------------------
    # Построение промпта
    # -------------------------
    def build_prompt(self, table_data: Dict) -> List[Dict]:
        return [
            {
                "role": "user",
                "content": f"""
Определи заголовки таблицы.

Таблица:
{json.dumps(table_data, ensure_ascii=False, indent=2)}

Ответ верни в JSON.
"""
            }
        ]

    # -------------------------
    # Обработка одного файла
    # -------------------------
    def process_file(self, file_path: Path):
        logging.info(f"Обработка: {file_path.name}")

        with open(file_path, "r", encoding="utf-8") as f:
            table_data = json.load(f)

        messages = self.build_prompt(table_data)

        result = self.make_api_call(messages)

        output_path = Path(OUTPUT_DIR) / f"{file_path.stem}_result.json"

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump({
                "file": file_path.name,
                "result": result
            }, f, ensure_ascii=False, indent=2)

    # -------------------------
    # Запуск всего эксперимента
    # -------------------------
    def run(self):
        files = list(Path(INPUT_DIR).glob("*.json"))

        logging.info(f"Найдено файлов: {len(files)}")

        for file_path in files:
            self.process_file(file_path)

        logging.info("Готово")


# =========================
# ЗАПУСК
# =========================
if __name__ == "__main__":
    collector = ResponseCollector()
    collector.run()