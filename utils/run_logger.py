from __future__ import annotations

from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
from typing import Any
from uuid import uuid4

from config import settings


_SUCCESS_STATUSES: frozenset[str] = frozenset({
    "completed",
    "reflective_completed",
    "direct_tool_completed",
    "reverse_best_effort_completed",
})

SENSITIVE_KV_PATTERN = re.compile(
    r"(?i)\b(api[_-]?key|access[_-]?key|secret[_-]?key|client[_-]?secret|auth[_-]?token"
    r"|token|secret|password|passwd|passphrase|credential|private[_-]?key"
    r"|x[_-]?api[_-]?key|aws[_-]?secret[_-]?access[_-]?key|aws[_-]?access[_-]?key[_-]?id"
    r"|db[_-]?password|database[_-]?url|connection[_-]?string|dsn)"
    r"\s*[:=]\s*['\"]?([^\s,'\"]+)"
)

BEARER_PATTERN = re.compile(r"(?i)\b(bearer)\s+([A-Za-z0-9._~+/=-]{12,})")

SENSITIVE_SINGLE_PATTERNS = (
    # OpenAI / Anthropic 風
    re.compile(r"\b(sk-(?:ant-)?[A-Za-z0-9_\-]{12,})\b"),
    # AWS access key ID
    re.compile(r"\b(AKIA[0-9A-Z]{16})\b"),
    # GitHub tokens
    re.compile(r"\b(gh[pousr]_[A-Za-z0-9]{30,})\b"),
    # Slack bot / user / app tokens
    re.compile(r"\b(xox[baprs]-[A-Za-z0-9-]{10,})\b"),
    # Stripe live keys
    re.compile(r"\b((?:sk|pk|rk)_live_[0-9A-Za-z]{16,})\b"),
    # Google API key
    re.compile(r"\b(AIza[0-9A-Za-z\-_]{30,})\b"),
    # JWT (ヘッダ.ペイロード.署名)
    re.compile(r"\b(eyJ[A-Za-z0-9_\-]{6,}\.[A-Za-z0-9_\-]{6,}\.[A-Za-z0-9_\-.+/=]{6,})\b"),
    # PEM private keys (ブロック全体)
    re.compile(
        r"-----BEGIN (?:RSA |DSA |EC |OPENSSH |PGP |ENCRYPTED )?PRIVATE KEY-----"
        r".*?"
        r"-----END (?:RSA |DSA |EC |OPENSSH |PGP |ENCRYPTED )?PRIVATE KEY-----",
        re.DOTALL,
    ),
    # Azure Storage AccountKey / SAS
    re.compile(r"AccountKey=[A-Za-z0-9+/=]{20,}"),
    re.compile(r"SharedAccessKey=[A-Za-z0-9+/=]{20,}"),
)


def redact_sensitive_text(text: str) -> str:
    redacted = SENSITIVE_KV_PATTERN.sub(
        lambda m: f"{m.group(1)}: [REDACTED]",
        text,
    )
    redacted = BEARER_PATTERN.sub(
        lambda m: f"{m.group(1)}: [REDACTED]",
        redacted,
    )
    for pattern in SENSITIVE_SINGLE_PATTERNS:
        redacted = pattern.sub("[REDACTED]", redacted)
    return redacted


def _sanitize_log_text(text: str) -> str:
    sanitized = redact_sensitive_text(str(text))
    limit = max(0, settings.max_log_text_chars)
    if limit and len(sanitized) > limit:
        omitted = len(sanitized) - limit
        sanitized = f"{sanitized[:limit]}\n... ({omitted} chars omitted)"
    return sanitized


def _safe_json_value(value: Any) -> Any:
    if isinstance(value, str):
        return _sanitize_log_text(value)
    if isinstance(value, list):
        return [_safe_json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _safe_json_value(item) for key, item in value.items()}

    try:
        json.dumps(value, ensure_ascii=False)
        return value
    except TypeError:
        return str(value)


def build_run_log(
    *,
    task: str,
    mode: str,
    status: str,
    output: str,
    plan: str = "",
    review: dict[str, Any] | None = None,
    intent: dict[str, Any] | None = None,
    tool_name: str | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    started_at = datetime.now().isoformat(timespec="seconds")
    task_hash = hashlib.sha256(task.encode("utf-8")).hexdigest()[:16]

    return {
        "run_id": f"{started_at.replace(':', '').replace('-', '')}-{uuid4().hex[:8]}",
        "created_at": started_at,
        "task_hash": task_hash,
        "mode": mode,
        "status": status,
        "model": settings.ollama_model,
        "tool_name": tool_name,
        "task": _sanitize_log_text(task),
        "plan": _sanitize_log_text(plan),
        "output": _sanitize_log_text(output),
        "review": _safe_json_value(review or {}),
        "intent": _safe_json_value(intent or {}),
        "metadata": _safe_json_value(metadata or {}),
    }


def save_run_log(
    *,
    task: str,
    mode: str,
    status: str,
    output: str,
    plan: str = "",
    review: dict[str, Any] | None = None,
    intent: dict[str, Any] | None = None,
    tool_name: str | None = None,
    metadata: dict[str, Any] | None = None,
    runs_path: str | None = None,
) -> Path:
    log = build_run_log(
        task=task,
        mode=mode,
        status=status,
        output=output,
        plan=plan,
        review=review,
        intent=intent,
        tool_name=tool_name,
        metadata=metadata,
    )

    target_dir = Path(runs_path or settings.runs_path)
    target_dir.mkdir(parents=True, exist_ok=True)
    path = target_dir / f"{log['run_id']}.json"
    path.write_text(
        json.dumps(log, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return path


def prune_run_logs(*, keep: int, runs_path: str | None = None) -> list[str]:
    """古いログを削除し、削除したファイル名の一覧を返す。"""
    keep = max(0, keep)
    target_dir = Path(runs_path or settings.runs_path)
    if not target_dir.exists():
        return []

    logs = sorted(
        target_dir.glob("*.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    removed: list[str] = []
    for path in logs[keep:]:
        path.unlink()
        removed.append(path.name)
    return removed


def summarize_run_logs(*, runs_path: str | None = None) -> dict[str, Any]:
    """runs/ 配下のログを集計して成功率・スコア分布などを返す。"""
    target_dir = Path(runs_path or settings.runs_path)
    empty: dict[str, Any] = {
        "total": 0,
        "success_rate_pct": 0.0,
        "avg_score": 0.0,
        "avg_attempts": None,
        "statuses": {},
        "modes": {},
    }
    if not target_dir.exists():
        return empty

    logs = list(target_dir.glob("*.json"))
    total = len(logs)
    if total == 0:
        return empty

    statuses: dict[str, int] = {}
    scores: list[int] = []
    attempts: list[int] = []
    modes: dict[str, int] = {}

    for path in logs:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue

        status = str(data.get("status", "unknown"))
        statuses[status] = statuses.get(status, 0) + 1

        mode = str(data.get("mode", "unknown"))
        modes[mode] = modes.get(mode, 0) + 1

        review = data.get("review", {})
        if isinstance(review, dict):
            score = review.get("score")
            if isinstance(score, (int, float)):
                scores.append(int(score))

        metadata = data.get("metadata", {})
        if isinstance(metadata, dict):
            attempt = metadata.get("attempt")
            if isinstance(attempt, int):
                attempts.append(attempt)

    completed = sum(statuses.get(s, 0) for s in _SUCCESS_STATUSES)
    return {
        "total": total,
        "success_rate_pct": round(completed / total * 100, 1),
        "avg_score": round(sum(scores) / len(scores), 1) if scores else 0.0,
        "avg_attempts": round(sum(attempts) / len(attempts), 2) if attempts else None,
        "statuses": statuses,
        "modes": modes,
    }
