from __future__ import annotations

import json
import re
from typing import Any


def extract_json_block(text: str) -> dict[str, Any]:
    if not text:
        raise ValueError("入力が空です")

    fenced_json = extract_fenced_json(text)
    if fenced_json is not None:
        return json.loads(fenced_json)

    brace_block = extract_first_json_object(text)
    if brace_block is not None:
        return json.loads(brace_block)

    raise ValueError("JSONブロックが見つかりません")


def extract_fenced_json(text: str) -> str | None:
    patterns = [
        r"```json\s*(\{.*?\})\s*```",
        r"```\s*(\{.*?\})\s*```",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return None


def extract_first_json_object(text: str) -> str | None:
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape = False

    for i in range(start, len(text)):
        ch = text[i]

        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue

        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]

    return None
