from __future__ import annotations

import re
from typing import Any

from type_defs import ReviewDict
from utils.json_utils import extract_json_block


__all__ = [
    "normalize_review",
    "safe_parse_review",
    "safe_parse_improver",
    "fallback_review",
]


def _coerce_str_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x) for x in value if x is not None]
    return [str(value)]


SCORE_QUANTUM = 5


def _snap_score(score: int) -> int:
    """小規模LLMの連続値較正の弱さを吸収するため、5点刻みに丸める。"""
    clamped = max(0, min(100, score))
    return int(round(clamped / SCORE_QUANTUM) * SCORE_QUANTUM)


def normalize_review(review: dict[str, Any]) -> ReviewDict:
    raw_score = review.get("score", 0)

    try:
        score = int(raw_score)
    except Exception:
        score = _coerce_score(raw_score)

    return {
        "score": _snap_score(score),
        "strengths": _coerce_str_list(review.get("strengths")),
        "weaknesses": _coerce_str_list(review.get("weaknesses")),
        "tags": _coerce_str_list(review.get("tags")),
        "next_action": str(review.get("next_action") or "").strip(),
    }



def fallback_review(reason: str) -> ReviewDict:
    return {
        "score": 50,
        "strengths": [],
        "weaknesses": [f"解析失敗: {reason}"],
        "tags": ["parse_error"],
        "next_action": "LLM出力形式を改善してください。",
    }

def safe_parse_review(text: str) -> ReviewDict:
    try:
        review = extract_json_block(text)
    except Exception:
        return fallback_review("json_parse_failed")

    if not isinstance(review, dict) or "score" not in review:
        return fallback_review("invalid_json_output")

    review.setdefault("strengths", [])
    review.setdefault("weaknesses", [])
    review.setdefault("tags", [])
    review.setdefault("next_action", "")

    return normalize_review(review)


def safe_parse_improver(text: str) -> dict[str, Any]:
    if not text or not isinstance(text, str):
        return {
            "priority_fixes": [],
            "keep": [],
            "improved_prompt": "",
            "error": "入力が空、または文字列ではありません",
        }

    try:
        obj = extract_json_block(text.strip())
        if isinstance(obj, dict):
            obj.setdefault("priority_fixes", [])
            obj.setdefault("keep", [])
            obj.setdefault("improved_prompt", "")
            return obj
    except Exception:
        pass

    return {
        "priority_fixes": [],
        "keep": [],
        "improved_prompt": "",
        "error": "parse_failed",
    }



def _coerce_score(raw_score: Any) -> int:
    if raw_score is None:
        return 50

    text = str(raw_score).strip()

    grade_map = {
        "A+": 98,
        "A": 95,
        "A-": 90,
        "B+": 87,
        "B": 83,
        "B-": 80,
        "C+": 77,
        "C": 73,
        "C-": 70,
        "D+": 67,
        "D": 63,
        "D-": 60,
        "F": 40,
    }

    if text in grade_map:
        return grade_map[text]

    m_100 = re.search(r"(\d+(?:\.\d+)?)\s*/\s*100", text)
    if m_100:
        return int(float(m_100.group(1)))

    m_10 = re.search(r"(\d+(?:\.\d+)?)\s*/\s*10", text)
    if m_10:
        return int(float(m_10.group(1)) * 10)

    m_num = re.search(r"\d+(?:\.\d+)?", text)
    if m_num:
        return int(float(m_num.group(0)))

    return 50
