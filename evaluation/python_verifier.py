from __future__ import annotations

import ast
import re
from typing import Any


def extract_python_candidate(output: str) -> str | None:
    text = (output or "").strip()
    if not text:
        return None

    fenced = re.search(r"```(?:python|py)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if fenced:
        return fenced.group(1).strip()

    if text.lstrip().startswith(("def ", "class ", "import ", "from ")):
        return text

    return None


def verify_python_output(output: str) -> dict[str, Any]:
    code = extract_python_candidate(output)
    if code is None:
        return {
            "checked": False,
            "ok": True,
            "error": "",
        }

    try:
        ast.parse(code)
    except SyntaxError as e:
        return {
            "checked": True,
            "ok": False,
            "error": f"構文エラー: {e.msg}（行 {e.lineno}）",
        }

    return {
        "checked": True,
        "ok": True,
        "error": "",
    }
