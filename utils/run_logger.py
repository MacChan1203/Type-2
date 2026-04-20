from __future__ import annotations

from datetime import datetime
import hashlib
import json
from pathlib import Path
from typing import Any
from uuid import uuid4

from config import settings


def _safe_json_value(value: Any) -> Any:
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
        "task": task,
        "plan": plan,
        "output": output,
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
