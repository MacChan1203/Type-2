from __future__ import annotations

import ast
import re
from typing import Any


DEFECT_PATTERNS = {
    "runtime_bug": [
        r"\braise\s+",
        r"\b1\s*/\s*0\b",
        r"\[[^\]]+\]\s*\[[^\]]+\]",
        r"\bNone\.",
    ],
    "logic_bug": [
        r"\brange\([^)]*-\s*1\)",
        r"\bis\s+['\"]",
        r"\bis\s+not\s+['\"]",
        r"\b==\s*None\b",
        r"\b!=\s*None\b",
    ],
    "state_bug": [
        r"\bglobal\s+",
        r"def\s+\w+\([^)]*=\s*\[\]",
        r"def\s+\w+\([^)]*=\s*\{\}",
        r"\brandom\.",
        r"\bdatetime\.now\(",
    ],
    "data_bug": [
        r"\.split\(['\"][,;]['\"]\)",
        r"\bint\(",
        r"\bfloat\(",
        r"\.strip\(\)",
        r"\.lower\(\)",
    ],
    "exception_bug": [
        r"except\s*:",
        r"except\s+Exception",
        r"except\s+.*:\s*\n\s*pass",
        r"except\s+.*:\s*\n\s*return\s+None",
    ],
    "security_bug": [
        r"\beval\(",
        r"\bexec\(",
        r"os\.environ",
        r"subprocess\.[^(]+\([^)]*shell\s*=\s*True",
        r"pickle\.loads?\(",
        r"yaml\.load\(",
        r"open\([^)]*['\"]w",
    ],
}


def score_reverse_output(output_text: str) -> dict[str, Any]:
    text = (output_text or "").strip()
    lower = text.lower()

    score = 0
    tags: list[str] = []
    weaknesses: list[str] = []
    strengths: list[str] = []

    # 1. コード本体の有無
    if "def " in lower or "class " in lower:
        score += 20
        strengths.append("コード本体が含まれている")
    else:
        weaknesses.append("コード本体が弱い")
        tags.append("missing_code")

    # 2. コード上の欠陥シグナル
    matched_categories = []
    for tag, patterns in DEFECT_PATTERNS.items():
        if any(re.search(pattern, text, re.IGNORECASE | re.MULTILINE) for pattern in patterns):
            matched_categories.append(tag)
            tags.append(tag)

    if len(matched_categories) >= 5:
        score += 35
        strengths.append("複数種類の欠陥シグナルがある")
    elif len(matched_categories) >= 3:
        score += 24
        weaknesses.append("欠陥の種類をさらに増やせる")
        tags.append("few_bug_types")
    else:
        score += 8
        weaknesses.append("欠陥シグナルが少ない")
        tags.append("missing_bug_taxonomy")

    # 3. Pythonとして読める形か
    try:
        ast.parse(text)
        score += 20
        strengths.append("Python構文として成立している")
    except SyntaxError:
        weaknesses.append("Python構文として成立していない")
        tags.append("syntax_error")

    # 4. 説明に逃げずコードのみになっているか
    explanatory_markers = [
        "バグ分析",
        "修正手順",
        "再発防止",
        "このコード",
        "以下",
        "```",
    ]
    explanatory_hits = sum(1 for marker in explanatory_markers if marker.lower() in lower)
    if explanatory_hits == 0:
        score += 15
        strengths.append("説明ではなくコードのみで表現されている")
    else:
        weaknesses.append("コード以外の説明が混ざっている")
        tags.append("too_explanatory")

    # 5. ファイル全体としてのまとまり
    if len(text.splitlines()) >= 20 and ("if __name__" in text or text.count("def ") >= 2):
        score += 15
        strengths.append("ファイル全体として読めるまとまりがある")
    else:
        weaknesses.append("成果物が断片的")
        tags.append("weak_structure")

    # 6. 良いコードすぎる兆候への減点
    anti_quality_penalty = 0
    if "production-ready" in lower or "best practice" in lower or "robust" in lower:
        anti_quality_penalty += 10
    if "todo" not in lower and "pass" not in lower and not matched_categories:
        anti_quality_penalty += 15

    if anti_quality_penalty:
        score -= anti_quality_penalty
        weaknesses.append("逆タスクなのに良いコード方向へ寄りすぎている")
        tags.append("too_correct")

    score = max(0, min(100, score))

    return {
        "score": score,
        "strengths": strengths,
        "weaknesses": weaknesses,
        "tags": tags,
        "next_action": "逆タスクでは、説明を混ぜず、コード本文の中に複数種類の自然な欠陥を埋め込むこと。",
    }

