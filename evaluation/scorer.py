from __future__ import annotations

import json
import re
from typing import Any

def extract_json_block(text: str) -> dict:
    if not text:
        raise ValueError("Empty input")

    fenced_json = _extract_fenced_json(text)
    if fenced_json is not None:
        return json.loads(fenced_json)

    brace_block = _extract_first_json_object(text)
    if brace_block is not None:
        return json.loads(brace_block)

    raise ValueError("No JSON block found")


def normalize_review(review: dict[str, Any]) -> dict[str, Any]:
    raw_score = review.get("score", 0)

    try:
        score = int(raw_score)
    except Exception:
        score = _coerce_score(raw_score)

    score = max(0, min(100, score))

    strengths = review.get("strengths", [])
    weaknesses = review.get("weaknesses", [])
    tags = review.get("tags", [])
    next_action = str(review.get("next_action", "")).strip()

    if not isinstance(strengths, list):
        strengths = [str(strengths)]

    if not isinstance(weaknesses, list):
        weaknesses = [str(weaknesses)]

    if not isinstance(tags, list):
        tags = [str(tags)]

    return {
        "score": score,
        "strengths": [str(x) for x in strengths],
        "weaknesses": [str(x) for x in weaknesses],
        "tags": [str(x) for x in tags],
        "next_action": next_action,
    }



def parse_review(text: str) -> dict[str, Any]:
    """
    Critic の出力をできるだけ review dict に変換する。

    優先順位:
    1. そのまま JSON
    2. ```json ... ``` ブロック
    3. 最初の { ... } を抽出
    4. 自然文から heuristic に推定
    """
    if not text or not isinstance(text, str):
        return fallback_review("empty_or_invalid_text")

    raw = text.strip()

    # 1) そのまま JSON
    parsed = _try_json_load(raw)
    if parsed is not None:
        return normalize_review(parsed)

    # 2) ```json ... ``` ブロック
    fenced_json = _extract_fenced_json(raw)
    if fenced_json is not None:
        parsed = _try_json_load(fenced_json)
        if parsed is not None:
            return normalize_review(parsed)

    # 3) 最初の {...} を抽出
    brace_block = _extract_first_json_object(raw)
    if brace_block is not None:
        parsed = _try_json_load(brace_block)
        if parsed is not None:
            return normalize_review(parsed)

    # 4) 最終手段: 自然文から推定
    return normalize_review(heuristic_parse(raw))


def extract_list(text: str, keywords: list[str]) -> list[str]:
    lines = text.split("\n")
    result = []

    for line in lines:
        for kw in keywords:
            if kw.lower() in line.lower():
                result.append(line.strip())

    return result[:5]



def _try_json_load(text: str) -> dict[str, Any] | None:
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
        return None
    except Exception:
        return None


def _extract_fenced_json(text: str) -> str | None:
    """
    ```json ... ``` または ``` ... ``` を抜く
    """
    patterns = [
        r"```json\s*(\{.*?\})\s*```",
        r"```\s*(\{.*?\})\s*```",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return None


def _extract_first_json_object(text: str) -> str | None:
    """
    テキスト内の最初の JSON object っぽい {...} を抽出する。
    中括弧のネストを数えて切り出す。
    """
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


def heuristic_parse(text: str) -> dict[str, Any]:
    """
    JSONで取れない自然文レビューを雑に review dict へ落とす。
    """
    lower = text.lower()

    score = _infer_score(text)
    strengths = _extract_bullets_after_headers(
        text,
        headers=["strength", "strengths", "長所", "良い点", "メリット"]
    )
    weaknesses = _extract_bullets_after_headers(
        text,
        headers=["weakness", "weaknesses", "短所", "弱点", "課題", "issues"]
    )

    # 箇条書きが拾えなかった時の最低限 fallback
    if not strengths:
        if any(word in lower for word in ["excellent", "strong", "good", "優れて", "高い", "秀逸"]):
            strengths = ["全体として肯定的な評価が含まれています。"]
        else:
            strengths = []

    if not weaknesses:
        if any(word in lower for word in ["critical", "issue", "weak", "problem", "致命的", "課題", "問題"]):
            weaknesses = ["本文中に改善点または問題点が含まれています。"]
        else:
            weaknesses = ["構造化された弱点の抽出に失敗しました。"]

    next_action = _infer_next_action(text)
    tags = _infer_tags(text, strengths, weaknesses)

    return {
        "score": score,
        "strengths": strengths[:5],
        "weaknesses": weaknesses[:5],
        "tags": tags[:5],
        "next_action": next_action,
    }


def _infer_score(text: str) -> int:
    """
    score / Final Score / 9.5/10 / 95 などを拾う
    """
    patterns = [
        r'"score"\s*:\s*(\d{1,3})',
        r"\bscore\b\s*[:=]\s*(\d{1,3})",
        r"\bfinal score\b\s*[:=]\s*(\d{1,3})",
        r"(\d+(?:\.\d+)?)\s*/\s*10",
        r"(\d+(?:\.\d+)?)\s*/\s*5",
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if not match:
            continue

        try:
            value = float(match.group(1))
            if "/10" in match.group(0):
                return max(0, min(100, int(value * 10)))
            if "/5" in match.group(0):
                return max(0, min(100, int(value * 20)))
            return max(0, min(100, int(value)))
        except Exception:
            pass

    lower = text.lower()
    # スコアが見つからない場合の雑推定
    if any(w in lower for w in ["critical", "致命的", "重大"]):
        return 55
    if any(w in lower for w in ["excellent", "outstanding", "優れて", "非常に高い"]):
        return 85
    return 70


def _extract_bullets_after_headers(text: str, headers: list[str]) -> list[str]:
    """
    見出しのあとに続く箇条書きをゆるく拾う
    """
    lines = text.splitlines()
    results: list[str] = []
    capture = False

    for line in lines:
        stripped = line.strip()
        lower = stripped.lower()

        # 見出しに反応
        if any(h.lower() in lower for h in headers):
            capture = True
            continue

        # 別見出しっぽいものが来たら終了
        if capture and (
            stripped.startswith("###")
            or stripped.startswith("##")
            or stripped.endswith(":")
            or stripped.endswith("：")
        ):
            if results:
                break

        if capture:
            if stripped.startswith(("-", "*", "•")):
                item = stripped.lstrip("-*•").strip()
                if item:
                    results.append(item)
            elif re.match(r"^\d+\.", stripped):
                item = re.sub(r"^\d+\.\s*", "", stripped).strip()
                if item:
                    results.append(item)

    return results


def _infer_next_action(text: str) -> str:
    candidates = [
        r'"next_action"\s*:\s*"([^"]+)"',
        r"\bnext action\b\s*[:：]\s*(.+)",
        r"\brecommendation[s]?\b\s*[:：]\s*(.+)",
        r"\bsummary\b\s*[:：]\s*(.+)",
    ]

    for pattern in candidates:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            value = match.group(1).strip()
            if value:
                return value[:400]

    # 末尾の数行から雑に拾う
    lines = [x.strip() for x in text.splitlines() if x.strip()]
    if lines:
        return lines[-1][:400]

    return "LLM出力形式を改善してください。"


def _infer_tags(text: str, strengths: list[str], weaknesses: list[str]) -> list[str]:
    lower = text.lower()
    tags: list[str] = []

    if "security" in lower or "脆弱性" in text or "xss" in lower or "sql injection" in lower:
        tags.append("security")
    if "structure" in lower or "構造" in text:
        tags.append("structure")
    if "educational" in lower or "教材" in text or "学習" in text:
        tags.append("educational_value")
    if weaknesses:
        tags.append("has_weaknesses")
    if strengths:
        tags.append("has_strengths")

    # 重複除去
    deduped = []
    for tag in tags:
        if tag not in deduped:
            deduped.append(tag)
    return deduped


def fallback_review(reason: str) -> dict[str, Any]:
    return {
        "score": 50,
        "strengths": [],
        "weaknesses": [f"parse failed: {reason}"],
        "tags": ["parse_error"],
        "next_action": "LLM出力形式を改善してください。",
    }

def safe_parse_review(text: str) -> dict[str, Any]:
    review = parse_review(text)

    # JSONっぽくない場合は失敗扱い
    if not isinstance(review, dict) or "score" not in review:
        return {
            "score": 40,
            "strengths": [],
            "weaknesses": ["invalid_json_output"],
            "tags": ["parse_failed"],
            "next_action": "JSON形式で出力してください"
        }

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
            "error": "empty_or_invalid_text",
        }

    raw = text.strip()

    # 1) そのままJSONとして読む
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            obj.setdefault("priority_fixes", [])
            obj.setdefault("keep", [])
            obj.setdefault("improved_prompt", "")
            return obj
    except Exception:
        pass

    # 2) ```json ... ``` ブロックを読む
    try:
        m = re.search(r"```json\s*(\{.*?\})\s*```", raw, re.DOTALL | re.IGNORECASE)
        if m:
            obj = json.loads(m.group(1))
            if isinstance(obj, dict):
                obj.setdefault("priority_fixes", [])
                obj.setdefault("keep", [])
                obj.setdefault("improved_prompt", "")
                return obj
    except Exception:
        pass

    # 3) ``` ... ``` ブロックを読む
    try:
        m = re.search(r"```\s*(\{.*?\})\s*```", raw, re.DOTALL)
        if m:
            obj = json.loads(m.group(1))
            if isinstance(obj, dict):
                obj.setdefault("priority_fixes", [])
                obj.setdefault("keep", [])
                obj.setdefault("improved_prompt", "")
                return obj
    except Exception:
        pass

    # 4) 最初の {...} を抽出して読む
    try:
        brace_block = _extract_first_json_object(raw)
        if brace_block:
            obj = json.loads(brace_block)
            if isinstance(obj, dict):
                obj.setdefault("priority_fixes", [])
                obj.setdefault("keep", [])
                obj.setdefault("improved_prompt", "")
                return obj
    except Exception:
        pass

    # 5) どうしてもダメなら失敗扱い
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

