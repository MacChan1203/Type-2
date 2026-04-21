from __future__ import annotations

from dataclasses import dataclass
import re
import threading
from typing import Callable

from tools.external_requests import is_external_execution_task, run_unsupported_external_task
from tools.hacker_news import is_hacker_news_task, run_hacker_news_task


CanHandle = Callable[[str], bool]
RunTool = Callable[[str, bool], str]


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    can_handle: CanHandle
    run: RunTool


@dataclass(frozen=True)
class ToolResult:
    tool_name: str
    output: str
    status: str = "SUCCESS"
    evidence: tuple[str, ...] = ()
    metadata: dict[str, str] | None = None


def infer_tool_status(output: str) -> str:
    failure_markers = (
        "ステータス: 失敗",
        "ステータス: FAILURE",
        '"status": "FAILURE"',
        '"status": "失敗"',
    )
    return "FAILURE" if any(marker in output for marker in failure_markers) else "SUCCESS"


def extract_evidence_urls(output: str) -> tuple[str, ...]:
    urls = re.findall(r"https?://[^\s)）>\"]+", output)
    return tuple(dict.fromkeys(urls))


class ToolRouter:
    def __init__(self, tools: list[ToolSpec] | None = None) -> None:
        self._tools = list(tools or [])

    def register(self, tool: ToolSpec) -> None:
        if any(existing.name == tool.name for existing in self._tools):
            raise ValueError(f"ツールはすでに登録されています: {tool.name}")
        self._tools.append(tool)

    @property
    def tools(self) -> tuple[ToolSpec, ...]:
        return tuple(self._tools)

    def match(self, task: str) -> ToolSpec | None:
        for tool in self._tools:
            if tool.can_handle(task):
                return tool
        return None

    def run(self, task: str, verbose: bool = True) -> ToolResult | None:
        tool = self.match(task)
        if tool is None:
            return None
        output = tool.run(task, verbose)
        return ToolResult(
            tool_name=tool.name,
            output=output,
            status=infer_tool_status(output),
            evidence=extract_evidence_urls(output),
            metadata={"description": tool.description},
        )


def build_default_router() -> ToolRouter:
    return ToolRouter(
        tools=[
            ToolSpec(
                name="hacker_news_top_japanese",
                description="Hacker Newsのトップ記事を取得し、本文を取得して日本語で表示します。",
                can_handle=is_hacker_news_task,
                run=run_hacker_news_task,
            ),
            ToolSpec(
                name="unsupported_external_execution",
                description="専用の実行ツールがない外部実行依頼を失敗として扱います。",
                can_handle=is_external_execution_task,
                run=run_unsupported_external_task,
            )
        ]
    )


_DEFAULT_ROUTER: ToolRouter | None = None
_DEFAULT_ROUTER_LOCK = threading.Lock()


def get_default_router() -> ToolRouter:
    global _DEFAULT_ROUTER
    if _DEFAULT_ROUTER is None:
        with _DEFAULT_ROUTER_LOCK:
            if _DEFAULT_ROUTER is None:
                _DEFAULT_ROUTER = build_default_router()
    return _DEFAULT_ROUTER
