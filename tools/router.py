from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

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


class ToolRouter:
    def __init__(self, tools: list[ToolSpec] | None = None) -> None:
        self._tools = list(tools or [])

    def register(self, tool: ToolSpec) -> None:
        if any(existing.name == tool.name for existing in self._tools):
            raise ValueError(f"Tool already registered: {tool.name}")
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
        return ToolResult(tool_name=tool.name, output=tool.run(task, verbose))


def build_default_router() -> ToolRouter:
    return ToolRouter(
        tools=[
            ToolSpec(
                name="hacker_news_top_japanese",
                description="Fetch the top Hacker News story, retrieve article text, and display it in Japanese.",
                can_handle=is_hacker_news_task,
                run=run_hacker_news_task,
            )
        ]
    )


_DEFAULT_ROUTER: ToolRouter | None = None


def get_default_router() -> ToolRouter:
    global _DEFAULT_ROUTER
    if _DEFAULT_ROUTER is None:
        _DEFAULT_ROUTER = build_default_router()
    return _DEFAULT_ROUTER
