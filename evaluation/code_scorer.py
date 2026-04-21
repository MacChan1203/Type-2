from __future__ import annotations

import ast
import re


# 行頭または空白直後に現れるキーワードだけをコード兆候として認める。
# これにより "1 from 10" のような自然文が誤ってコード判定されない。
CODE_HINT_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"(?:^|\n)\s*def\s"),
    re.compile(r"(?:^|\n)\s*class\s"),
    re.compile(r"(?:^|\n)\s*import\s"),
    re.compile(r"(?:^|\n)\s*from\s+\S+\s+import\s"),
    re.compile(r"```"),
    re.compile(r"\breturn\s+\S"),
    re.compile(r"\bif\s+__name__\b"),
    re.compile(r"(?:^|\n)\s*function\s"),
    re.compile(r"(?:^|\n)\s*const\s+\w+"),
    re.compile(r"(?:^|\n)\s*let\s+\w+"),
    re.compile(r"(?:^|\n)\s*var\s+\w+"),
    re.compile(r"\bSELECT\s+\w", re.IGNORECASE),
    re.compile(r"<html\b", re.IGNORECASE),
    re.compile(r"^#!/bin/", re.MULTILINE),
)


def looks_like_code(text: str) -> bool:
    return any(pattern.search(text) for pattern in CODE_HINT_PATTERNS)


def looks_like_plain_python(text: str) -> bool:
    stripped = text.lstrip()
    if stripped.startswith("```"):
        return False
    return stripped.startswith(("def ", "class ", "import ", "from "))


def score_text(output: str) -> int:
    text = output.strip()
    lower = text.lower()
    score = 100

    if len(text) < 25:
        score -= 35
    if len(text.split()) < 8 and len(text) < 45:
        score -= 20
    if re.search(r"\btodo\b", lower) or re.search(r"\bfixme\b", lower):
        score -= 10
    if text.count("\n") == 0 and len(text) > 800:
        score -= 15
    if "わかりません" in text or "不明です" in text:
        score -= 20

    return max(0, min(100, score))


def score_code(output: str, is_reverse: bool = False) -> int:
    score = 100

    if not output or not isinstance(output, str):
        return 0

    code_lower = output.lower()

    if is_reverse:
        if not looks_like_code(output):
            return 0
        if not re.search(r"\bdef\s", code_lower) and not re.search(r"\bclass\s", code_lower):
            score -= 20
        if "```" in code_lower:
            score -= 10
        if len(output.strip()) < 80:
            score -= 20
        return max(0, min(100, score))

    if not looks_like_code(output):
        return score_text(output)

    # 実用度ヒット: 関数/メソッド呼び出しに限定し、識別子の部分一致を除外
    if re.search(r"\beval\s*\(", output):
        score -= 40

    if re.search(r"\bexec\s*\(", output):
        score -= 40

    if looks_like_plain_python(output):
        try:
            ast.parse(output)
        except SyntaxError:
            score -= 35

    if re.search(r"\bexcept\s+Exception\b", output):
        score -= 20

    # "pass" 文単独の行のみ検出（password 等の部分一致を防ぐ）
    if re.search(r"(?:^|\n)\s*pass\s*(?:#.*)?$", output, re.MULTILINE):
        score -= 10

    if not re.search(r"\bdef\s", code_lower) and not re.search(r"\bclass\s", code_lower):
        score -= 20

    if re.search(r"\btodo\b", code_lower):
        score -= 5

    if re.search(r"\bfixme\b", code_lower):
        score -= 10

    if re.search(r"\basdf\b", code_lower) or re.search(r"\bxxx\b", code_lower):
        score -= 10

    if len(output.strip()) < 20:
        score -= 20

    return max(0, min(100, score))
