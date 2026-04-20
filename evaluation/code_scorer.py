from __future__ import annotations


CODE_HINTS = (
    "def ",
    "class ",
    "import ",
    "from ",
    "```",
    "return ",
    "if __name__",
    "function ",
    "const ",
    "let ",
    "var ",
    "SELECT ",
    "<html",
    "#!/bin/",
)


def looks_like_code(text: str) -> bool:
    lowered = text.lower()
    return any(hint.lower() in lowered for hint in CODE_HINTS)


def score_text(output: str) -> int:
    text = output.strip()
    score = 100

    if len(text) < 25:
        score -= 35
    if len(text.split()) < 8 and len(text) < 45:
        score -= 20
    if "todo" in text.lower() or "fixme" in text.lower():
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
        if "def " not in code_lower and "class " not in code_lower:
            score -= 20
        if "```" in code_lower:
            score -= 10
        if len(output.strip()) < 80:
            score -= 20
        return max(0, min(100, score))

    if not looks_like_code(output):
        return score_text(output)

    if "eval(" in code_lower:
        score -= 40

    if "exec(" in code_lower:
        score -= 40

    if "except exception" in code_lower:
        score -= 20

    if "pass" in code_lower:
        score -= 10

    if "def " not in code_lower and "class " not in code_lower:
        score -= 20

    if "todo" in code_lower:
        score -= 5

    if "fixme" in code_lower:
        score -= 10

    if "asdf" in code_lower or "xxx" in code_lower:
        score -= 10

    if len(output.strip()) < 20:
        score -= 20

    return max(0, min(100, score))
