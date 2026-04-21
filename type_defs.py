from __future__ import annotations

from typing import Any, NotRequired, TypedDict


class ReviewDict(TypedDict):
    score: int
    strengths: list[str]
    weaknesses: list[str]
    tags: list[str]
    next_action: str
    reflection: NotRequired[dict[str, Any]]


class IntentDict(TypedDict):
    intent: str
    is_reverse_task: bool
    success_criteria: str


class ReflectiveControlDict(TypedDict):
    objective: str
    assumptions: list[str]
    subgoals: list[str]
    risks: list[str]
    success_criteria: list[str]
    worker_instruction: str


class ReflectiveReflectionDict(TypedDict):
    score: int
    done: bool
    strengths: list[str]
    weaknesses: list[str]
    next_instruction: str


class ImproverResultDict(TypedDict):
    priority_fixes: list[str]
    keep: list[str]
    improved_prompt: str
    error: NotRequired[str]
