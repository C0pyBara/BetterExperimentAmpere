#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
normalize_pubtables_json.py

Приводит PubTables JSON к упрощённому формату,
совместимому с твоим XLSX-конвертером.
"""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional


# =========================
# НАСТРОЙКИ
# =========================

INPUT_DIR = Path(r"C:\Users\YtkaB\Desktop\Dataset_for_betterExperiment\Get_500_Tables_from_PubTables\JSON_Complex_TOP500")
OUTPUT_DIR = Path(r"C:\Users\YtkaB\Desktop\Dataset_for_betterExperiment\Get_500_Tables_from_PubTables\JSON_Complex_TOP500_normalized")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# =========================
# НОРМАЛИЗАЦИЯ ЯЧЕЙКИ
# =========================

def normalize_cell(cell: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Очищает одну ячейку до нужного формата.
    """

    text_xml = cell.get("xml_text_content", "")
    text_pdf = cell.get("pdf_text_content", text_xml)

    # Пропускаем полностью пустые
    if not str(text_xml).strip() and not str(text_pdf).strip():
        return None

    return {
        "row_nums": cell.get("row_nums", []),
        "column_nums": cell.get("column_nums", []),
        "xml_text_content": str(text_xml).strip(),
        "pdf_text_content": str(text_pdf).strip(),
        "is_column_header": bool(cell.get("is_column_header", False)),
        "is_projected_row_header": bool(cell.get("is_projected_row_header", False)),
    }


# =========================
# НОРМАЛИЗАЦИЯ ТАБЛИЦЫ
# =========================

def normalize_table(table: Dict[str, Any]) -> Dict[str, Any]:
    """
    Приводит таблицу к нужному формату.
    """

    cells_in = table.get("cells", [])
    cells_out: List[Dict[str, Any]] = []

    for cell in cells_in:
        new_cell = normalize_cell(cell)
        if new_cell:
            cells_out.append(new_cell)

    return {
        "cells": cells_out,
        "structure_id": table.get("structure_id", "unknown"),
        "pdf_file_name": table.get("pdf_file_name", "unknown.pdf"),
        "split": table.get("split", "unknown"),
    }


# =========================
# ОБРАБОТКА ФАЙЛА
# =========================

def process_file(json_path: Path):
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"[SKIP] Ошибка чтения {json_path}: {e}")
        return

    if not isinstance(data, list):
        print(f"[SKIP] Неверный формат (не список): {json_path}")
        return

    result = []

    for table in data:
        norm_table = normalize_table(table)
        result.append(norm_table)

    out_path = OUTPUT_DIR / json_path.name

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"[OK] {json_path.name}")


# =========================
# ОБРАБОТКА ПАПКИ
# =========================

def process_all():
    files = list(INPUT_DIR.glob("*.json"))

    print(f"Найдено файлов: {len(files)}")

    for f in files:
        process_file(f)

    print("Готово!")


# =========================
# MAIN
# =========================

if __name__ == "__main__":
    process_all()