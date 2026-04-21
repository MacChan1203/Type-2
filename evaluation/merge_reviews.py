from __future__ import annotations
from typing import Any

# Reverse task scoring weights
_W_INTENT_REVERSE = 0.35
_W_REVERSE_LLM = 0.35
_W_REVERSE_RULE = 0.25
_W_CODE_REVERSE = 0.05

# Normal task scoring weights
_W_INTENT_NORMAL = 0.50
_W_QUALITY_NORMAL = 0.30
_W_CODE_NORMAL = 0.20

# Score caps applied by quality gate
_CAP_ALL_PARSE_FAILED = 50
_CAP_PARTIAL_PARSE_FAILED = 75
_CAP_LOW_CODE_SCORE = 65
_CAP_LOW_RULE_SCORE = 69

# Threshold boundaries
_RULE_SCORE_WARN = 60
_RULE_SCORE_PASS = 75
_CODE_SCORE_WARN = 50


def _as_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return list(value)
    return []


def merge_reviews(
    intent_review: dict[str, Any],
    quality_review: dict[str, Any] | None,
    reverse_review: dict[str, Any] | None,
    intent: dict[str, Any],
    code_score: int = 0,
    reverse_rule_score: int = 0,
) -> dict[str, Any]:
    intent_score = int(intent_review.get("score", 0) or 0)
    quality_score = int(quality_review.get("score", 0) or 0) if quality_review else 0
    reverse_score = int(reverse_review.get("score", 0) or 0) if reverse_review else 0

    is_reverse = bool(intent.get("is_reverse_task", False))

    # -------------------------
    # Reverse Task
    # -------------------------
    if is_reverse:
        final_score = int(
            intent_score * _W_INTENT_REVERSE +
            reverse_score * _W_REVERSE_LLM +
            reverse_rule_score * _W_REVERSE_RULE +
            code_score * _W_CODE_REVERSE
        )
    else:
        final_score = int(
            intent_score * _W_INTENT_NORMAL +
            quality_score * _W_QUALITY_NORMAL +
            code_score * _W_CODE_NORMAL
        )

    strengths = (
        _as_list(intent_review.get("strengths"))
        + (_as_list(quality_review.get("strengths")) if quality_review else [])
        + (_as_list(reverse_review.get("strengths")) if reverse_review else [])
    )

    weaknesses = (
        _as_list(intent_review.get("weaknesses"))
        + (_as_list(quality_review.get("weaknesses")) if quality_review else [])
        + (_as_list(reverse_review.get("weaknesses")) if reverse_review else [])
    )

    tags = (
        _as_list(intent_review.get("tags"))
        + (_as_list(quality_review.get("tags")) if quality_review else [])
        + (_as_list(reverse_review.get("tags")) if reverse_review else [])
    )

    if is_reverse and reverse_rule_score < _RULE_SCORE_WARN:
        weaknesses.append("逆タスク専用の構造要件を満たしていない")
        tags.append("reverse_structure_fail")

    if is_reverse and reverse_rule_score >= _RULE_SCORE_PASS:
        strengths.append("ルールベース評価では逆タスクとして十分な欠陥構造を満たしている")
        tags.append("reverse_rule_success")

    # 重複削除
    strengths = list(dict.fromkeys(strengths))
    weaknesses = list(dict.fromkeys(weaknesses))
    tags = list(dict.fromkeys(tags))

    next_action = ""
    quality_next = quality_review.get("next_action") if quality_review else None
    if is_reverse and reverse_review:
        next_action = str(
            reverse_review.get("next_action")
            or quality_next
            or intent_review.get("next_action", "")
        )
    else:
        next_action = str(
            quality_next
            or intent_review.get("next_action", "")
        )

    # parse_error は「どれだけのCriticが解析に失敗したか」で段階化する。
    # 以前は 1 つでも失敗したら min(60) で潰していたため、
    # fallback_review の score=50 と相まって出力が常に 50 近辺で貼り付いていた。
    parse_failed_reviews = sum(
        1
        for r in (intent_review, quality_review, reverse_review)
        if r
        and any(
            t in ("parse_error", "parse_failed") for t in (_as_list(r.get("tags")))
        )
    )
    total_reviews = (
        1
        + (1 if quality_review else 0)
        + (1 if reverse_review else 0)
    )

    if parse_failed_reviews >= total_reviews:
        final_score = min(final_score, _CAP_ALL_PARSE_FAILED)
        weaknesses.append("全てのCritic出力の解析に失敗したため合格扱いにしない")
    elif parse_failed_reviews > 0:
        # 部分失敗は信頼度が下がるが、他Criticが高評価なら合格圏に残せる。
        final_score = min(final_score, _CAP_PARTIAL_PARSE_FAILED)
        weaknesses.append("一部Criticの出力解析に失敗しているため上限を抑えた")

    if not is_reverse and code_score < _CODE_SCORE_WARN:
        final_score = min(final_score, _CAP_LOW_CODE_SCORE)
        weaknesses.append("ルールベースのコード評価が低いため、LLM評価だけでは合格扱いにしない")

    if is_reverse and reverse_rule_score < _RULE_SCORE_WARN:
        final_score = min(final_score, _CAP_LOW_RULE_SCORE)

    # 以前は reverse_rule_score >= 75 のときに final_score を max(85) で
    # 強制的に引き上げていたが、ルール一致だけでLLM評価を覆すのは行き過ぎ。
    # reverse_rule_score はすでに重み 0.25 で最終スコアへ寄与しているため、
    # ここでは追加の上書きを行わない。

    return {
        "score": max(0, min(100, final_score)),
        "strengths": list(dict.fromkeys(strengths)),
        "weaknesses": list(dict.fromkeys(weaknesses)),
        "tags": tags,
        "next_action": next_action,
    }
