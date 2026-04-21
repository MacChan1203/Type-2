from __future__ import annotations

import io
import re
import tokenize

from evaluation.python_verifier import extract_python_candidate


NOISY_REVERSE_MARKERS = (
    "WARNING",
    "注意",
    "意図的",
    "バグ",
    "欠陥",
    "脆弱",
    "セキュリティ",
    "情報漏",
    "実行時エラー",
    "論理",
    "攻撃",
    "パストラバーサル",
    "インジェクション",
    "シミュレーション",
    "デモ",
    "学習",
    "検証目的",
    "設計ミス",
    "エラーハンドリング",
    "ハードコード",
    "誤って",
    "不正",
    "不十分",
    "していない",
    "エスケープ",
    "甘い",
    "リスク",
)


NOISY_REVERSE_PRINT_MARKERS = (
    "warn",
    "warning",
    "fail",
    "critical",
    "vuln",
    "injection",
    "leaked",
    "leakage",
    "bypass",
    "exploit",
    "security",
    "debug",
    "error",
    "password",
    "credential",
    "admin",
    "unsafe",
    "脆弱",
    "漏洩",
    "攻撃",
    "警告",
    "汚染",
    "ロックなし",
    "認証",
    "管理",
)


DEMO_DEFINITION_MARKERS = ("exploit", "attack", "demo", "scenario", "test_", "run_example")


def normalize_reverse_output(output: str) -> str:
    cleaned = re.sub(r"<unused\d+>", "", output)
    code = extract_python_candidate(cleaned)
    if code:
        return sanitize_reverse_code(code)

    lines = []
    for line in cleaned.strip().splitlines():
        stripped = line.strip()
        if stripped.startswith(("###", "解説", "総評", "**総評")):
            break
        if stripped.startswith("<unused"):
            continue
        if should_drop_reverse_comment(stripped):
            continue
        lines.append(line)
    return sanitize_reverse_code("\n".join(lines).strip())


def should_drop_reverse_comment(stripped_line: str) -> bool:
    if not stripped_line.startswith("#"):
        return False
    comment_body = stripped_line.lstrip("#").strip()
    if comment_body and set(comment_body) <= {"=", "-", "*"}:
        return True
    return has_noisy_reverse_marker(stripped_line)


def has_noisy_reverse_marker(text: str) -> bool:
    return any(marker in text for marker in NOISY_REVERSE_MARKERS)


def sanitize_reverse_code(code: str) -> str:
    without_noisy_docstrings = strip_reverse_docstrings(code)
    without_comments = strip_python_comments(without_noisy_docstrings)
    kept_lines = []
    for line in without_comments.splitlines():
        stripped = line.strip()
        if stripped.startswith("if __name__") and "__main__" in stripped:
            break
        if is_noisy_reverse_print(stripped):
            continue
        kept_lines.append(line.rstrip())
    return trim_trailing_comment_only_block(
        compact_reverse_blank_lines(
            remove_top_level_reverse_demo("\n".join(kept_lines).strip())
        )
    )


def compact_reverse_blank_lines(code: str) -> str:
    lines = code.splitlines()
    compacted: list[str] = []

    for index, line in enumerate(lines):
        if line.strip():
            compacted.append(line)
            continue

        previous = next((x for x in reversed(compacted) if x.strip()), "")
        next_line = next((x for x in lines[index + 1:] if x.strip()), "")
        previous_indent = len(previous) - len(previous.lstrip(" ")) if previous else 0
        next_indent = len(next_line) - len(next_line.lstrip(" ")) if next_line else 0

        if previous_indent > 0 or next_indent > 0:
            continue
        if compacted and compacted[-1].strip() == "":
            continue
        compacted.append(line)

    return "\n".join(compacted).strip()


def strip_reverse_docstrings(code: str) -> str:
    return re.sub(r'(?s)([ \t]*)(("""|\'\'\'))(.*?)(\2)', "", code)


def strip_python_comments(code: str) -> str:
    try:
        tokens = []
        for token in tokenize.generate_tokens(io.StringIO(code).readline):
            if token.type == tokenize.COMMENT:
                continue
            tokens.append(token)
        return tokenize.untokenize(tokens)
    except (tokenize.TokenError, SyntaxError):
        return "\n".join(
            line.partition("#")[0].rstrip()
            for line in code.splitlines()
        )


def remove_top_level_reverse_demo(code: str) -> str:
    lines = code.splitlines()
    kept: list[str] = []
    saw_body_definition = False
    dropping_demo = False

    for line in lines:
        stripped = line.strip()
        indent = len(line) - len(line.lstrip(" "))
        is_top_level = indent == 0

        if is_top_level and is_reverse_demo_definition(stripped):
            dropping_demo = True
            continue

        if is_top_level and stripped.startswith(("class ", "def ")):
            saw_body_definition = True
            dropping_demo = False
            kept.append(line)
            continue

        if is_top_level and stripped.startswith(("import ", "from ", "@")):
            if not dropping_demo:
                kept.append(line)
            continue

        if is_top_level and saw_body_definition and stripped:
            dropping_demo = True
            continue

        if not dropping_demo:
            kept.append(line)

    return "\n".join(kept).strip()


def is_reverse_demo_definition(stripped_line: str) -> bool:
    match = re.match(r"(?:async\s+)?def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", stripped_line)
    if not match:
        return False
    name = match.group(1).lower()
    return any(marker in name for marker in DEMO_DEFINITION_MARKERS)


def is_noisy_reverse_print(stripped_line: str) -> bool:
    if not stripped_line.startswith("print("):
        return False
    lowered = stripped_line.lower()
    return any(marker in lowered for marker in NOISY_REVERSE_PRINT_MARKERS)


def trim_trailing_comment_only_block(code: str) -> str:
    lines = code.splitlines()
    while lines and not lines[-1].strip():
        lines.pop()

    while lines and lines[-1].strip().startswith("#"):
        lines.pop()
        while lines and not lines[-1].strip():
            lines.pop()

    return "\n".join(lines).strip()
