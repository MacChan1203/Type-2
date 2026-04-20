from __future__ import annotations
from typing import Any, Dict


def merge_reviews(
    intent_review: Dict[str, Any],
    quality_review: Dict[str, Any],
    reverse_review: Dict[str, Any] | None,
    intent: Dict[str, Any],
    code_score: int = 0,
    reverse_rule_score: int = 0,
) -> Dict[str, Any]:
    intent_score = int(intent_review.get("score", 0))
    quality_score = int(quality_review.get("score", 0))
    reverse_score = int(reverse_review.get("score", 0)) if reverse_review else 0

    is_reverse = bool(intent.get("is_reverse_task", False))

    # -------------------------
    # Reverse Task
    # -------------------------
    if is_reverse:
        final_score = int(
            intent_score * 0.35 +
            reverse_score * 0.35 +
            reverse_rule_score * 0.25 +
            code_score * 0.05
        )
    else:
        final_score = int(
            intent_score * 0.50 +
            quality_score * 0.30 +
            code_score * 0.20
        )

    strengths = (
        intent_review.get("strengths", []) +
        quality_review.get("strengths", []) +
        (reverse_review.get("strengths", []) if reverse_review else [])
    )

    weaknesses = (
        intent_review.get("weaknesses", []) +
        quality_review.get("weaknesses", []) +
        (reverse_review.get("weaknesses", []) if reverse_review else [])
    )

    tags = (
        intent_review.get("tags", []) +
        quality_review.get("tags", []) +
        (reverse_review.get("tags", []) if reverse_review else [])
    )

    if is_reverse and reverse_rule_score < 60:
        weaknesses.append("逆タスク専用の構造要件を満たしていない")
        tags.append("reverse_structure_fail")

    # 重複削除
    strengths = list(dict.fromkeys(strengths))
    weaknesses = list(dict.fromkeys(weaknesses))
    tags = list(dict.fromkeys(tags))

    next_action = ""
    if is_reverse and reverse_review:
        next_action = str(
            reverse_review.get("next_action")
            or quality_review.get("next_action")
            or intent_review.get("next_action", "")
        )
    else:
        next_action = str(
            quality_review.get("next_action")
            or intent_review.get("next_action", "")
        )

    return {
        "score": max(0, min(100, final_score)),
        "strengths": strengths,
        "weaknesses": weaknesses,
        "tags": tags,
        "next_action": next_action,
    }

