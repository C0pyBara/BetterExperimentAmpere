from pathlib import Path

_DIR = Path(__file__).resolve().parent


def _read_txt(name: str) -> str:
    return (_DIR / f"{name}.txt").read_text(encoding="utf-8")


PROMPTS = [
    {"name": "zero_min", "user": _read_txt("zero_min")},
    {"name": "zero_max", "user": _read_txt("zero_max")},
    {"name": "sc_min", "user": _read_txt("sc_min")},
    {"name": "sc_max", "user": _read_txt("sc_max")},
    {"name": "reasoning_min", "user": _read_txt("reasoning_min")},
    {"name": "reasoning_max", "user": _read_txt("reasoning_max")},
]
