from __future__ import annotations

from datetime import datetime
from typing import Any


def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "y"}:
            return True
        if normalized in {"false", "0", "no", "n", ""}:
            return False
    return bool(value)


def print_section(title: str, body: str) -> None:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)
    print(body)

