from __future__ import annotations

import re
from typing import Any


EXECUTION_TASK_MARKERS = (
    "Hacker News",
    "hacker news",
    "ニュース",
    "最新",
    "現在",
    "今日",
    "取得",
    "検索",
    "API",
    "スクレイピング",
    "クロール",
    "URL",
    "サイト",
    "Web",
    "外部",
    "時刻",
    "になったら",
)

EXECUTION_CLAIM_MARKERS = (
    "取得しました",
    "取得し",
    "実行しました",
    "実行されました",
    "接続し",
    "フェッチ",
    "クロール",
    "スクレイピング",
    "成功",
    "完了しました",
    "翻訳しました",
)

SIMULATION_MARKERS = (
    "シミュレーション",
    "前提として",
    "取得されたことを前提",
    "ダミー",
    "プレースホルダー",
    "xxxxxxxx",
    "ABC123",
    "example.com",
)


def requires_execution_evidence(task: str) -> bool:
    return any(marker in task for marker in EXECUTION_TASK_MARKERS)


def claims_execution(output: str) -> bool:
    return any(marker in output for marker in EXECUTION_CLAIM_MARKERS)


def has_execution_evidence(output: str) -> bool:
    if "ステータス: FAILURE" in output or '"status": "FAILURE"' in output:
        return True

    urls = re.findall(r"https?://[^\s)）>\"]+", output)
    real_urls = [
        url
        for url in urls
        if "example.com" not in url
        and "xxxxxxxx" not in url
        and "ABC123" not in url
    ]
    if real_urls:
        return True

    evidence_patterns = (
        r"\bitem\?id=\d+\b",
        r"\bid[=:]\s*\d+\b",
        r"実行時刻:\s*\d{4}-\d{2}-\d{2}T\d{2}:\d{2}",
        r"理由:\s*\S+",
        r"取得メモ:\s*\S+",
    )
    return any(re.search(pattern, output) for pattern in evidence_patterns)


def detect_unsupported_execution_claim(task: str, output: str) -> bool:
    if not requires_execution_evidence(task):
        return False
    if any(marker in output for marker in SIMULATION_MARKERS):
        return True
    if claims_execution(output) and not has_execution_evidence(output):
        return True
    return False


def apply_execution_claim_review(
    task: str,
    output: str,
    review: dict[str, Any],
) -> dict[str, Any]:
    if not detect_unsupported_execution_claim(task, output):
        return review

    updated = dict(review)
    weaknesses = list(updated.get("weaknesses", []))
    tags = list(updated.get("tags", []))

    weaknesses.append("外部取得や実行を成功したように述べているが、実URL・取得ID・失敗理由などの検証可能な証拠がない")
    tags.append("unsupported_execution_claim")

    updated["weaknesses"] = list(dict.fromkeys(weaknesses))
    updated["tags"] = list(dict.fromkeys(tags))
    updated["score"] = min(int(updated.get("score", 50)), 35)
    updated["next_action"] = (
        "外部取得・時刻実行・APIアクセスはToolRouter経由で実行し、成功時は実URLやID、失敗時は理由を必ず出力する"
    )
    return updated
