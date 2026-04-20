from __future__ import annotations

from datetime import datetime


def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def print_section(title: str, body: str) -> None:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)
    print(body)

