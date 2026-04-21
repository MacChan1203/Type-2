from __future__ import annotations

import ast
import re
from typing import Any


DEFECT_PATTERNS = {
    "runtime_bug": [
        r"\braise\s+",
        r"\b1\s*/\s*0\b",
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
        # split してそのまま固定インデックスを取るのは典型的なデータバグ
        r"\.split\(['\"][,;]['\"]\)\s*\[",
        # 多段インデックス参照（誤った shape 仮定）
        r"\[[^\]]+\]\s*\[[^\]]+\]",
        # 辞書取得値を bool 不定のまま int 変換するパターン
        r"\bint\(\s*[A-Za-z_][A-Za-z0-9_]*\.get\(",
        r"\bfloat\(\s*[A-Za-z_][A-Za-z0-9_]*\.get\(",
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


OVEREXPLAINED_COMMENT_MARKERS = (
    "意図的なバグ",
    "バグの埋め込み",
    "欠陥",
    "セキュリティリスク",
    "情報漏えい",
    "実行時エラー",
    "論理バグ",
    "状態管理",
    "不適切な例外処理",
)


COMMENT_MISMATCH_PATTERNS = (
    (
        r"#.*appendではなく.*インデックス",
        r"\.append\(",
    ),
    (
        r"#.*0で割",
        r"/\s*\([^)]*\+\s*1\)",
    ),
    (
        r"#.*例外を投げる",
        r"return\s+",
    ),
)


def detect_overexplained_bug_comments(text: str) -> bool:
    comment_lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip().startswith("#")
    ]
    hits = 0
    for line in comment_lines:
        if any(marker in line for marker in OVEREXPLAINED_COMMENT_MARKERS):
            hits += 1
    return hits >= 2


def detect_comment_code_mismatch(text: str) -> bool:
    return any(
        re.search(comment_pattern, text, re.IGNORECASE)
        and re.search(code_pattern, text, re.IGNORECASE)
        for comment_pattern, code_pattern in COMMENT_MISMATCH_PATTERNS
    )


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

    # 6. コメントが答え合わせになっていないか、実装と矛盾していないか
    comment_penalty = 0
    if detect_overexplained_bug_comments(text):
        comment_penalty += 10
        weaknesses.append("コメントでバグの場所や種類を説明しすぎている")
        tags.append("overexplained_bug_comments")

    if detect_comment_code_mismatch(text):
        comment_penalty += 12
        weaknesses.append("コメントと実装内容が矛盾している")
        tags.append("comment_code_mismatch")

    if comment_penalty == 0:
        strengths.append("コメントが過剰な答え合わせになっていない")

    # 7. 良いコードすぎる兆候への減点。
    # 以前は "production-ready" 等の英単語部分一致で減点していたが、
    # 逆タスクの説明文に付随して正当に現れうるため誤爆しやすい。
    # ここでは「欠陥が一切検出できない」実体ベースの兆候のみ見る。
    anti_quality_penalty = 0
    if not matched_categories and not re.search(r"\btodo\b|\bfixme\b|\bpass\b", text, re.IGNORECASE):
        anti_quality_penalty += 15

    if anti_quality_penalty:
        score -= anti_quality_penalty
        weaknesses.append("逆タスクなのに良いコード方向へ寄りすぎている")
        tags.append("too_correct")

    score -= comment_penalty

    score = max(0, min(100, score))

    return {
        "score": score,
        "strengths": strengths,
        "weaknesses": weaknesses,
        "tags": tags,
        "next_action": "逆タスクでは、説明を混ぜず、コード本文の中に複数種類の自然な欠陥を埋め込むこと。",
    }
