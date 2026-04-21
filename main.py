from __future__ import annotations

import argparse
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import redirect_stdout
import io
import json
import logging
import sys
from typing import Any

_logger = logging.getLogger("type2")


def _setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
        level=logging.DEBUG if verbose else logging.WARNING,
        force=True,
    )

from agents.crew_setup import (
    run_reflective_controller,
    run_reflective_reflector,
    run_planner,
    run_worker,
    run_intent_checker,
    run_intent_critic,
    run_quality_critic,
    run_reverse_critic,
    run_improver,
)
from config import settings
from memory.chroma_store import MemoryStore, MemoryStoreProtocol

from evaluation.code_scorer import score_code
from evaluation.execution_claims import apply_execution_claim_review
from evaluation.merge_reviews import merge_reviews
from evaluation.python_verifier import extract_python_candidate, verify_python_output
from evaluation.reverse_fallback import (
    apply_reverse_warning,
    build_reverse_fallback_output,
    build_reverse_fallback_review,
    should_use_reverse_fallback,
)
from evaluation.reverse_metrics import score_reverse_output
from evaluation.reverse_normalize import normalize_reverse_output
from evaluation.scorer import (
    fallback_review,
    safe_parse_review,
    safe_parse_improver,
)
from tools.router import ToolResult, get_default_router
from evaluation.thresholds import SCORE_SUCCESS_THRESHOLD
from utils.helpers import coerce_bool
from utils.json_utils import extract_json_block
from utils.run_logger import prune_run_logs, save_run_log, summarize_run_logs
from type_defs import (
    ReflectiveControlDict,
    ReflectiveReflectionDict,
    ImproverResultDict,
    IntentDict,
    ReviewDict,
)


_MAX_INTENT_CHARS = 300
_MAX_NEXT_ACTION_CHARS = 500
_MAX_IMPROVED_PROMPT_CHARS = 300


class DisabledMemoryStore:
    def search_success_memories(self, query: str, n_results: int | None = None) -> list[str]:
        return []

    def search_failure_memories(self, query: str, n_results: int | None = None) -> list[str]:
        return []

    def format_memories_for_prompt(self, query: str, include_failures: bool = True) -> str:
        return ""

    def add_memory(self, **kwargs: Any) -> str:
        return "memory-disabled"

    def clear_all_memories(self) -> int:
        return 0


def build_memory_store(verbose: bool = True) -> MemoryStoreProtocol:
    try:
        return MemoryStore()
    except Exception as e:
        _logger.warning("メモリ初期化エラー: %s", e)
        return DisabledMemoryStore()


def run_direct_tool_if_supported(task: str, verbose: bool = True) -> ToolResult | None:
    result = get_default_router().run(task, verbose=verbose)
    return result


def record_run_log(
    *,
    task: str,
    mode: str,
    status: str,
    output: str,
    plan: str = "",
    review: ReviewDict | None = None,
    intent: IntentDict | None = None,
    tool_name: str | None = None,
    metadata: dict[str, Any] | None = None,
    verbose: bool = True,
) -> None:
    try:
        path = save_run_log(
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
        if verbose:
            _logger.info("実行ログ: %s", path)
    except Exception as e:
        _logger.warning("実行ログエラー: %s", e)


def safe_parse_reflective_control(text: str) -> ReflectiveControlDict:
    try:
        obj = extract_json_block(text)
        if isinstance(obj, dict):
            obj.setdefault("objective", "")
            obj.setdefault("assumptions", [])
            obj.setdefault("subgoals", [])
            obj.setdefault("risks", [])
            obj.setdefault("success_criteria", [])
            obj.setdefault("worker_instruction", "")
            return obj
    except Exception:
        pass

    return {
        "objective": "",
        "assumptions": [],
        "subgoals": [],
        "risks": ["反省モード制御JSONの解析に失敗しました"],
        "success_criteria": [],
        "worker_instruction": text.strip() if isinstance(text, str) else "",
    }


def safe_parse_reflective_reflection(text: str) -> ReflectiveReflectionDict:
    try:
        obj = extract_json_block(text)
        if isinstance(obj, dict):
            obj.setdefault("score", 50)
            obj.setdefault("done", False)
            obj.setdefault("strengths", [])
            obj.setdefault("weaknesses", [])
            obj.setdefault("next_instruction", "")
            obj["score"] = max(0, min(100, int(obj.get("score", 50))))
            obj["done"] = coerce_bool(obj.get("done", False))
            return obj
    except Exception:
        pass

    return {
        "score": 50,
        "done": False,
        "strengths": [],
        "weaknesses": ["反省JSONの解析に失敗しました"],
        "next_instruction": "現在の依頼に直接答え、過去タスクへ逸脱しないこと。",
    }


def default_intent(task: str) -> IntentDict:
    return {
        "intent": task[:_MAX_INTENT_CHARS],
        "is_reverse_task": has_reverse_hint(task),
        "success_criteria": "ユーザーの依頼に沿った成果物を返すこと",
    }


STRONG_REVERSE_MARKERS = (
    "逆タスク",
    "わざとバグ",
    "わざと欠陥",
    "わざと壊",
    "わざと失敗",
    "わざと動かない",
    "意図的なバグ",
    "意図的な欠陥",
    "レビュー教材",
    "デバッグ教材",
    "悪いコード例",
    "失敗例を作",
)

REVERSE_MARKERS = (
    "わざと",
    "意図的",
    "あえて",
    "故意",
    "失敗例",
    "悪い例",
    "失敗する実装",
    "壊れたコード",
    "意図的に壊",
)

NEGATIVE_TARGETS = (
    "バグ",
    "壊",
    "悪く",
    "不完全",
    "欠陥",
    "失敗",
    "動かない",
    "誤動作",
    "クラッシュ",
    "脆弱",
    "危険",
)


def has_reverse_hint(task: str) -> bool:
    if any(marker in task for marker in STRONG_REVERSE_MARKERS):
        return True
    has_marker = any(marker in task for marker in REVERSE_MARKERS)
    has_target = any(target in task for target in NEGATIVE_TARGETS)
    return has_marker and has_target


def should_use_memory(task: str, is_reverse: bool = False) -> bool:
    if is_reverse:
        return True

    simple_markers = ("書いて", "作って", "実装", "出力", "サンプル")
    simple_targets = ("FizzBuzz", "hello", "足し算", "関数", "コード")
    if any(marker in task for marker in simple_markers) and any(
        target in task for target in simple_targets
    ):
        return False

    return True


def parse_intent(task: str, verbose: bool = True) -> IntentDict:
    try:
        intent_raw = run_intent_checker(task_text=task)
        intent = extract_json_block(intent_raw)
        if not isinstance(intent, dict):
            return default_intent(task)
    except Exception as e:
        _logger.warning("意図解析エラー: %s", e)
        return default_intent(task)

    intent.setdefault("intent", task[:_MAX_INTENT_CHARS])
    intent.setdefault("success_criteria", "ユーザーの依頼に沿った成果物を返すこと")
    intent["is_reverse_task"] = bool(intent.get("is_reverse_task", False))

    if has_reverse_hint(task):
        intent["is_reverse_task"] = True

    return intent

# =========================
# Retry判定
# =========================
def should_retry(
    score: int,
    threshold: int = 80,
    is_reverse: bool = False,
    tags: list[str] | None = None,
) -> bool:
    tags = tags or []

    if is_reverse:
        if "missing_bug_taxonomy" in tags:
            return True
        if "missing_code" in tags:
            return True
        if "reverse_structure_fail" in tags:
            return True
        if "too_correct" in tags:
            return True
        if "python_syntax_error" in tags:
            return True
        if "overexplained_bug_comments" in tags:
            return True
        if "comment_code_mismatch" in tags:
            return True
        if "reverse_rule_success" in tags:
            return False
        return score < 90

    return score < threshold


def format_retry_message(is_reverse: bool, tags: list[str]) -> str:
    if is_reverse:
        if "parse_error" in tags or "parse_failed" in tags:
            return "🔁 逆タスクサンプルを改善します（Critic出力が構造化されていません）"
        return "🔁 逆タスクサンプルを改善します"
    return "🔁 再試行します"


def max_attempt_status(is_reverse: bool, output: str, tags: list[str]) -> str:
    if is_reverse:
        if "missing_code" in tags or not extract_python_candidate(output):
            return "reverse_max_attempts_no_code"
        if "python_syntax_error" in tags:
            return "reverse_max_attempts_syntax_failed"
        if "reverse_structure_fail" in tags:
            return "reverse_max_attempts_structure_failed"
        if output.strip():
            return "reverse_best_effort_completed"
    return "max_attempts_reached"


def format_max_attempt_message(is_reverse: bool, status: str) -> str:
    if is_reverse and status == "reverse_best_effort_completed":
        return "✅ 最良の逆タスクサンプルとして完了"
    if is_reverse and status == "reverse_local_fallback_completed":
        return "⚠️ 逆タスクの生成が不安定だったため、ローカルのフォールバックコードを返します"
    if is_reverse:
        return "⚠️ 逆タスクの試行は残課題ありで終了"
    return "⚠️ 最大試行回数に達しました"


# =========================
# review整理
# =========================
def _as_str_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x) for x in value if x is not None]
    return [str(value)]


def clean_review(review: ReviewDict, limit: int | None = None) -> ReviewDict:
    strengths = _as_str_list(review.get("strengths"))
    weaknesses = _as_str_list(review.get("weaknesses"))
    tags = _as_str_list(review.get("tags"))

    if limit is not None:
        strengths = strengths[:limit]
        weaknesses = weaknesses[:limit]
        tags = tags[:limit]

    return {
        "score": int(review.get("score", 50) or 50),
        "strengths": strengths,
        "weaknesses": weaknesses,
        "tags": tags,
        "next_action": str(review.get("next_action") or "")[:_MAX_NEXT_ACTION_CHARS],
    }


# =========================
# fallback改善
# =========================
def build_improvement_note(review: ReviewDict) -> str:
    weaknesses = "\n".join(f"- {w}" for w in review.get("weaknesses", []))
    next_action = review.get("next_action", "")

    return f"""
【前回の弱点】
{weaknesses}

【改善】
{next_action}
""".strip()


# =========================
# improver結果反映
# =========================
def build_improvement_note_from_improver(improver_result: ImproverResultDict) -> str:
    improved = str(improver_result.get("improved_prompt", ""))[:_MAX_IMPROVED_PROMPT_CHARS]
    return f"""
【改善】
{improved}
""".strip()


def build_current_task(base_task: str, improvement_note: str) -> str:
    if not improvement_note:
        return base_task

    return f"""
{base_task}

{improvement_note}
""".strip()


def save_attempt_memory(
    store: MemoryStoreProtocol,
    task: str,
    plan: str,
    output: str,
    review: ReviewDict,
    improvement_note: str,
    is_reverse_task: bool = False,
    verbose: bool = True,
) -> None:
    try:
        store.add_memory(
            task=task,
            plan=plan,
            output=output,
            review=review,
            improvement_note=improvement_note,
            is_reverse_task=is_reverse_task,
            model=settings.ollama_model,
        )
    except Exception as e:
        _logger.warning("メモリ保存エラー: %s", e)


def format_reflective_control(control: ReflectiveControlDict) -> str:
    assumptions = "\n".join(f"- {x}" for x in control.get("assumptions", []))
    subgoals = "\n".join(f"- {x}" for x in control.get("subgoals", []))
    risks = "\n".join(f"- {x}" for x in control.get("risks", []))
    criteria = "\n".join(f"- {x}" for x in control.get("success_criteria", []))

    return f"""
目的:
{control.get("objective", "")}

前提:
{assumptions}

サブゴール:
{subgoals}

リスク:
{risks}

成功条件:
{criteria}

Worker指示:
{control.get("worker_instruction", "")}
""".strip()


def _collect_critic(fut: Future[str], label: str) -> ReviewDict:
    try:
        return clean_review(safe_parse_review(fut.result()))
    except Exception as e:
        _logger.warning("Critic失敗 (%s): %s", label, e)
        return clean_review(fallback_review(f"{label}_failed"))


def evaluate_output(
    *,
    task_text: str,
    output: str,
    intent: IntentDict,
    verbose: bool = False,
) -> ReviewDict:
    is_reverse = bool(intent.get("is_reverse_task", False))
    code_score = score_code(output, is_reverse=is_reverse)
    if verbose:
        _logger.debug("コードスコア: %s", code_score)

    # Criticは互いに独立しているため並列実行できる。
    # ただしOllamaがデフォルトで直列処理する設定の場合はキューイングされるだけで
    # 実速度は OLLAMA_NUM_PARALLEL の設定に依存する。
    with ThreadPoolExecutor(max_workers=3) as executor:
        fut_intent = executor.submit(
            run_intent_critic, task_text=task_text, output_text=output
        )
        fut_quality = (
            None
            if is_reverse
            else executor.submit(
                run_quality_critic,
                task_text=task_text,
                output_text=output,
                is_reverse=False,
            )
        )
        fut_reverse = (
            executor.submit(run_reverse_critic, task_text=task_text, output_text=output)
            if is_reverse
            else None
        )
        intent_review = _collect_critic(fut_intent, "intent_critic")
        quality_review = (
            _collect_critic(fut_quality, "quality_critic") if fut_quality else None
        )
        reverse_review = (
            _collect_critic(fut_reverse, "reverse_critic") if fut_reverse else None
        )

    reverse_rule_score = score_reverse_output(output)["score"] if is_reverse else 0

    review = merge_reviews(
        intent_review=intent_review,
        quality_review=quality_review,
        reverse_review=reverse_review,
        intent=intent,
        code_score=code_score,
        reverse_rule_score=reverse_rule_score,
    )
    review = apply_execution_claim_review(task_text, output, review)
    review = clean_review(review)

    verification = verify_python_output(output)
    if verification["checked"] and not verification["ok"]:
        weaknesses = list(review.get("weaknesses", []))
        tags = list(review.get("tags", []))
        weaknesses.append(f"Python構文検証に失敗: {verification['error']}")
        tags.append("python_syntax_error")
        review["weaknesses"] = list(dict.fromkeys(weaknesses))
        review["tags"] = list(dict.fromkeys(tags))
        review["score"] = min(int(review.get("score", 50)), 65)
        review["next_action"] = "Pythonコードを返す場合は、少なくとも構文として成立する形に修正してください。"

    return review


# =========================
# Pipeline本体
# =========================
def run_pipeline(task: str, verbose: bool = True) -> str:
    direct_result = run_direct_tool_if_supported(task, verbose=verbose)
    if direct_result is not None:
        record_run_log(
            task=task,
            mode="standard",
            status="direct_tool_completed" if direct_result.status == "SUCCESS" else "direct_tool_failed",
            output=direct_result.output,
            tool_name=direct_result.tool_name,
            metadata={
                "tool_router": True,
                "tool_status": direct_result.status,
                "evidence": list(direct_result.evidence),
                **(direct_result.metadata or {}),
            },
            verbose=verbose,
        )
        return direct_result.output

    store = build_memory_store(verbose=verbose)

    # ---- Intent ----
    intent = parse_intent(task, verbose=verbose)
    improvement_note = ""

    if verbose:
        _logger.info("=== 意図 ===\n%s", intent)

    # ---- Loop ----
    max_attempts = settings.max_pipeline_attempts
    output = ""
    for attempt in range(1, max_attempts + 1):
        current_task = build_current_task(task, improvement_note)

        if verbose:
            _logger.info("試行 %d / %d", attempt, max_attempts)

        # ---- Memory ----
        if should_use_memory(current_task, bool(intent.get("is_reverse_task", False))):
            try:
                memory_text = store.format_memories_for_prompt(
                    query=current_task,
                    include_failures=attempt > 1,
                ) or ""
            except Exception as e:
                _logger.warning("メモリエラー: %s", e)
                memory_text = ""
        else:
            memory_text = ""

        executed_task_text = f"""
{current_task}

=== 過去の知見 ===
{memory_text}
""".strip()

        # ---- Planner ----
        if intent.get("is_reverse_task", False):
            plan = "依頼に沿う壊れ方を決め、複数種類の欠陥を自然に埋め込む。"
        else:
            plan = run_planner(
                task_text=executed_task_text,
                memory_text=memory_text,
            )

        if verbose:
            _logger.info("計画: %s", plan)

        # ---- Worker ----
        output = run_worker(
            task_text=executed_task_text,
            memory_text=memory_text,
            plan_text=plan,
            is_reverse=bool(intent.get("is_reverse_task", False)),
        )
        if intent.get("is_reverse_task", False):
            output = normalize_reverse_output(output)
        if verbose:
            _logger.info("出力: %s", output)

        is_reverse = bool(intent.get("is_reverse_task", False))
        review = evaluate_output(
            task_text=current_task,
            output=output,
            intent=intent,
            verbose=verbose,
        )

        if verbose:
            _logger.info("評価: %s", clean_review(review, limit=5))

        # ---- Retry ----
        if should_retry(
            score=review["score"],
            is_reverse=is_reverse,
            tags=review.get("tags", []),
        ):
            if verbose:
                _logger.info("%s", format_retry_message(is_reverse, review.get("tags", [])))

            if attempt >= max_attempts:
                status = max_attempt_status(
                    is_reverse=is_reverse,
                    output=output,
                    tags=review.get("tags", []),
                )
                if should_use_reverse_fallback(status):
                    output = build_reverse_fallback_output(task)
                    review = build_reverse_fallback_review(review)
                    status = "reverse_local_fallback_completed"
                if verbose:
                    _logger.info("%s", format_max_attempt_message(is_reverse, status))
                save_attempt_memory(
                    store=store,
                    task=task,
                    plan=plan,
                    output=output,
                    review=review,
                    improvement_note=improvement_note or status,
                    is_reverse_task=is_reverse,
                    verbose=verbose,
                )
                record_run_log(
                    task=task,
                    mode="standard",
                    status=status,
                    output=output,
                    plan=plan,
                    review=review,
                    intent=intent,
                    metadata={"attempt": attempt, "is_reverse_task": is_reverse},
                    verbose=verbose,
                )
                return apply_reverse_warning(output) if is_reverse else output

            if settings.use_llm_improver:
                try:
                    raw_improver = run_improver(
                        task_text=current_task,
                        review_text=str(review),
                    )
                    improver_result = safe_parse_improver(raw_improver)
                    if improver_result.get("error"):
                        improvement_note = build_improvement_note(review)
                    else:
                        improvement_note = build_improvement_note_from_improver(improver_result)
                except Exception as e:
                    _logger.warning("改善生成エラー: %s", e)
                    improvement_note = build_improvement_note(review)
            else:
                # USE_LLM_IMPROVER=false のときは LLM 呼び出しを省略して決定論的に組む
                improvement_note = build_improvement_note(review)

            continue

        else:
            if verbose:
                _logger.info("✅ 完了")
            save_attempt_memory(
                store=store,
                task=task,
                plan=plan,
                output=output,
                review=review,
                improvement_note=improvement_note or "completed",
                is_reverse_task=is_reverse,
                verbose=verbose,
            )
            record_run_log(
                task=task,
                mode="standard",
                status="completed",
                output=output,
                plan=plan,
                review=review,
                intent=intent,
                metadata={"attempt": attempt, "is_reverse_task": is_reverse},
                verbose=verbose,
            )
            return apply_reverse_warning(output) if is_reverse else output

    return apply_reverse_warning(output) if is_reverse else output


def run_reflective_pipeline(task: str, verbose: bool = True) -> str:
    direct_result = run_direct_tool_if_supported(task, verbose=verbose)
    if direct_result is not None:
        record_run_log(
            task=task,
            mode="reflective",
            status="direct_tool_completed" if direct_result.status == "SUCCESS" else "direct_tool_failed",
            output=direct_result.output,
            tool_name=direct_result.tool_name,
            metadata={
                "tool_router": True,
                "tool_status": direct_result.status,
                "evidence": list(direct_result.evidence),
                **(direct_result.metadata or {}),
            },
            verbose=verbose,
        )
        return direct_result.output

    store = build_memory_store(verbose=verbose)
    intent = parse_intent(task, verbose=verbose)
    is_reverse = bool(intent.get("is_reverse_task", False))
    memory_text = ""

    if should_use_memory(task, is_reverse):
        try:
            memory_text = store.format_memories_for_prompt(
                query=task,
                include_failures=False,
            ) or ""
        except Exception as e:
            _logger.warning("メモリエラー: %s", e)

    current_instruction = task
    best_output = ""
    best_review: ReviewDict = {"score": 0, "strengths": [], "weaknesses": [], "tags": [], "next_action": ""}
    best_plan = ""

    if verbose:
        _logger.info("=== 反省モード ===")
        _logger.info("注意: これは真のAGIではなく、目標分析・実行・反省を回すAGI風モードです。")
        _logger.info("=== 意図 ===\n%s", intent)

    for cycle in range(1, settings.max_reflective_cycles + 1):
        if verbose:
            _logger.info("反省サイクル %d", cycle)

        control_raw = run_reflective_controller(
            task_text=current_instruction,
            memory_text=memory_text,
        )
        control = safe_parse_reflective_control(control_raw)
        plan = format_reflective_control(control)

        if verbose:
            _logger.info("制御: %s", control)

        worker_task = f"""
現在の依頼:
{task}

反省モード制御指示:
{control.get("worker_instruction", "")}

重要:
- 現在の依頼を最優先する
- 過去タスクや参考記憶の題材へ逸脱しない
- 真のAGIであるとは主張しない
""".strip()

        output = run_worker(
            task_text=worker_task,
            memory_text=memory_text,
            plan_text=plan,
            is_reverse=is_reverse,
        )
        if is_reverse:
            output = normalize_reverse_output(output)

        review = evaluate_output(
            task_text=task,
            output=output,
            intent=intent,
        )

        reflection_raw = run_reflective_reflector(
            task_text=task,
            plan_text=plan,
            output_text=output,
            review_text=str(review),
        )
        reflection = safe_parse_reflective_reflection(reflection_raw)

        combined_score = int(review["score"] * 0.7 + reflection["score"] * 0.3)
        if combined_score > int(best_review.get("score", 0)):
            best_output = output
            best_plan = plan
            best_review = {
                **review,
                "score": combined_score,
                "reflection": reflection,
            }

        if verbose:
            _logger.info("出力: %s", output)
            _logger.info("評価: %s", clean_review(review, limit=5))
            _logger.info("反省: %s", reflection)

        if reflection.get("done") and combined_score >= SCORE_SUCCESS_THRESHOLD:
            break

        next_instruction = str(reflection.get("next_instruction", "")).strip()
        if not next_instruction:
            next_instruction = review.get("next_action", "")

        current_instruction = f"""
{task}

【前回の反省】
{next_instruction}
""".strip()

    save_attempt_memory(
        store=store,
        task=task,
        plan=best_plan,
        output=best_output,
        review=best_review,
        improvement_note="reflective_completed",
        is_reverse_task=is_reverse,
        verbose=verbose,
    )
    record_run_log(
        task=task,
        mode="reflective",
        status="reflective_completed",
        output=best_output,
        plan=best_plan,
        review=best_review,
        intent=intent,
        metadata={"max_cycles": settings.max_reflective_cycles, "is_reverse_task": is_reverse},
        verbose=verbose,
    )
    return apply_reverse_warning(best_output) if is_reverse else best_output


# =========================
# CLI
# =========================
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task")
    parser.add_argument("--quiet", action="store_true", help="ログを抑えて最終結果だけを出力する")
    parser.add_argument("--prune-runs", type=int, help="最新N件だけ残して runs の古いJSONログを削除する")
    parser.add_argument("--clear-memory", action="store_true", help="ChromaDB上の成功・失敗記憶を削除する")
    parser.add_argument("--stats", action="store_true", help="runs/ の実行ログ集計（成功率・平均スコア等）を表示する")
    parser.add_argument(
        "--mode",
        choices=("standard", "agi", "reflective"),
        default="standard",
        help="standard は通常パイプライン、reflective は目標分析・反省サイクルを使う。agi は互換用エイリアス",
    )
    args = parser.parse_args()
    if not args.task and args.prune_runs is None and not args.clear_memory and not args.stats:
        parser.error("--task、--prune-runs、--clear-memory のいずれかを指定してください")
    return args


def execute_task(task: str, mode: str, quiet: bool = False) -> str:
    def _run() -> str:
        if mode in ("agi", "reflective"):
            return run_reflective_pipeline(task=task, verbose=not quiet)
        return run_pipeline(task=task, verbose=not quiet)

    if not quiet:
        return _run()

    stdout_buffer = io.StringIO()
    try:
        with redirect_stdout(stdout_buffer):
            return _run()
    except Exception as e:
        captured_stdout = stdout_buffer.getvalue().strip()
        if captured_stdout:
            raise RuntimeError(f"{e}\n\n捕捉した標準出力:\n{captured_stdout}") from e
        raise


def main():
    args = parse_args()
    _setup_logging(verbose=not getattr(args, "quiet", False))

    if getattr(args, "stats", False):
        summary = summarize_run_logs()
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    maintenance_messages = []
    if args.prune_runs is not None:
        removed = prune_run_logs(keep=args.prune_runs)
        maintenance_messages.append(f"実行ログ削除数: {len(removed)}")
        for name in removed:
            _logger.info("削除: %s", name)
    if args.clear_memory:
        store = build_memory_store(verbose=not args.quiet)
        removed = store.clear_all_memories()
        maintenance_messages.append(f"記憶削除数: {removed}")
    if not args.task:
        print("\n".join(maintenance_messages))
        return 0

    result = execute_task(task=args.task, mode=args.mode, quiet=args.quiet)
    if maintenance_messages and not args.quiet:
        print("\n".join(maintenance_messages))
    print(result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
