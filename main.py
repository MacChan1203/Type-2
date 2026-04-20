from __future__ import annotations

import argparse
from contextlib import redirect_stderr, redirect_stdout
import io
import sys
from typing import Any

from agents.crew_setup import (
    run_agi_controller,
    run_agi_reflector,
    run_planner,
    run_worker,
    run_intent_checker,
    run_intent_critic,
    run_quality_critic,
    run_reverse_critic,
    run_improver,
)
from config import settings
from memory.chroma_store import MemoryStore

from evaluation.code_scorer import score_code
from evaluation.execution_claims import apply_execution_claim_review
from evaluation.merge_reviews import merge_reviews
from evaluation.reverse_metrics import score_reverse_output
from evaluation.scorer import (
    extract_json_block,
    safe_parse_review,
    safe_parse_improver,
)
from tools.router import ToolResult, get_default_router
from utils.run_logger import save_run_log


def run_direct_tool_if_supported(task: str, verbose: bool = True) -> ToolResult | None:
    result = get_default_router().run(task, verbose=verbose)
    return result


def run_direct_task_if_supported(task: str, verbose: bool = True) -> str | None:
    result = run_direct_tool_if_supported(task, verbose=verbose)
    return result.output if result is not None else None


def record_run_log(
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
            print("RUN LOG:", path)
    except Exception as e:
        if verbose:
            print("RUN LOG ERROR:", e)


def safe_parse_agi_control(text: str) -> dict[str, Any]:
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
        "risks": ["AGI controller JSON parse failed"],
        "success_criteria": [],
        "worker_instruction": text.strip() if isinstance(text, str) else "",
    }


def safe_parse_agi_reflection(text: str) -> dict[str, Any]:
    try:
        obj = extract_json_block(text)
        if isinstance(obj, dict):
            obj.setdefault("score", 50)
            obj.setdefault("done", False)
            obj.setdefault("strengths", [])
            obj.setdefault("weaknesses", [])
            obj.setdefault("next_instruction", "")
            obj["score"] = max(0, min(100, int(obj.get("score", 50))))
            obj["done"] = bool(obj.get("done", False))
            return obj
    except Exception:
        pass

    return {
        "score": 50,
        "done": False,
        "strengths": [],
        "weaknesses": ["AGI reflector JSON parse failed"],
        "next_instruction": "現在の依頼に直接答え、過去タスクへ逸脱しないこと。",
    }


def default_intent(task: str) -> dict[str, Any]:
    return {
        "intent": task[:300],
        "is_reverse_task": has_reverse_hint(task),
        "success_criteria": "ユーザーの依頼に沿った成果物を返すこと",
    }


def has_reverse_hint(task: str) -> bool:
    reverse_markers = (
        "わざと",
        "意図的",
        "あえて",
        "故意",
        "逆タスク",
        "失敗例",
        "悪い例",
        "失敗する実装",
        "レビュー教材",
    )
    negative_targets = ("バグ", "壊", "悪く", "不完全", "欠陥", "失敗")
    return any(marker in task for marker in reverse_markers) and any(
        target in task for target in negative_targets
    )


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


def parse_intent(task: str, verbose: bool = True) -> dict[str, Any]:
    try:
        intent_raw = run_intent_checker(task_text=task)
        intent = extract_json_block(intent_raw)
        if not isinstance(intent, dict):
            return default_intent(task)
    except Exception as e:
        if verbose:
            print("INTENT ERROR:", e)
        return default_intent(task)

    intent.setdefault("intent", task[:300])
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
        return score < 90

    return score < threshold


# =========================
# review整理
# =========================
def clean_review(review: dict, limit: int | None = None) -> dict:
    strengths = review.get("strengths", [])
    weaknesses = review.get("weaknesses", [])
    tags = review.get("tags", [])

    if limit is not None:
        strengths = strengths[:limit]
        weaknesses = weaknesses[:limit]
        tags = tags[:limit]

    return {
        "score": int(review.get("score", 50)),
        "strengths": strengths,
        "weaknesses": weaknesses,
        "tags": tags,
        "next_action": str(review.get("next_action", ""))[:500],
    }


# =========================
# fallback改善
# =========================
def build_improvement_note(review: dict) -> str:
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
def build_improvement_note_from_improver(improver_result: dict) -> str:
    return f"""
【改善】
{improver_result.get("improved_prompt", "")}
""".strip()


def build_current_task(base_task: str, improvement_note: str) -> str:
    if not improvement_note:
        return base_task

    return f"""
{base_task}

{improvement_note}
""".strip()


def save_attempt_memory(
    store: MemoryStore,
    task: str,
    plan: str,
    output: str,
    review: dict,
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
        if verbose:
            print("MEMORY SAVE ERROR:", e)


def format_agi_control(control: dict[str, Any]) -> str:
    assumptions = "\n".join(f"- {x}" for x in control.get("assumptions", []))
    subgoals = "\n".join(f"- {x}" for x in control.get("subgoals", []))
    risks = "\n".join(f"- {x}" for x in control.get("risks", []))
    criteria = "\n".join(f"- {x}" for x in control.get("success_criteria", []))

    return f"""
Objective:
{control.get("objective", "")}

Assumptions:
{assumptions}

Subgoals:
{subgoals}

Risks:
{risks}

Success Criteria:
{criteria}

Worker Instruction:
{control.get("worker_instruction", "")}
""".strip()


# =========================
# Pipeline本体
# =========================
def run_pipeline(task: str, verbose: bool = True) -> str:
    direct_result = run_direct_tool_if_supported(task, verbose=verbose)
    if direct_result is not None:
        record_run_log(
            task=task,
            mode="standard",
            status="direct_tool_completed",
            output=direct_result.output,
            tool_name=direct_result.tool_name,
            metadata={"tool_router": True},
            verbose=verbose,
        )
        return direct_result.output

    store = MemoryStore()

    # ---- Intent ----
    intent = parse_intent(task, verbose=verbose)
    improvement_note = ""

    if verbose:
        print("=== INTENT ===")
        print(intent)

    # ---- Loop ----
    for attempt in range(1, settings.max_improvement_loops + 2):
        current_task = build_current_task(task, improvement_note)

        if verbose:
            print("=" * 50)
            print(f"試行 {attempt}")
            print("=" * 50)

        # ---- Memory ----
        if should_use_memory(current_task, bool(intent.get("is_reverse_task", False))):
            try:
                memory_text = store.format_memories_for_prompt(
                    query=current_task,
                    include_failures=attempt > 1,
                ) or ""
            except Exception as e:
                if verbose:
                    print("MEMORY ERROR:", e)
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
            print("PLAN:", plan)

        # ---- Worker ----
        output = run_worker(
            task_text=executed_task_text,
            memory_text=memory_text,
            plan_text=plan,
            is_reverse=bool(intent.get("is_reverse_task", False)),
        )
        if verbose:
            print("OUTPUT:", output)

        # ---- Score ----
        is_reverse = bool(intent.get("is_reverse_task", False))
        code_score = score_code(output, is_reverse=is_reverse)
        if verbose:
            print("CODE SCORE:", code_score)

        # ---- Critics ----
        intent_review = clean_review(
            safe_parse_review(
                run_intent_critic(task_text=current_task, output_text=output)
            )
        )

        quality_review = clean_review(
            safe_parse_review(
                run_quality_critic(
                    task_text=current_task,
                    output_text=output,
                    is_reverse=is_reverse,
                )
            )
        )

        reverse_review = None
        reverse_rule_score = 0

        if is_reverse:
            reverse_review = clean_review(
                safe_parse_review(
                    run_reverse_critic(task_text=current_task, output_text=output)
                )
            )

            reverse_rule_score = score_reverse_output(output)["score"]

        # ---- Merge ----
        review = merge_reviews(
            intent_review=intent_review,
            quality_review=quality_review,
            reverse_review=reverse_review,
            intent=intent,
            code_score=code_score,
            reverse_rule_score=reverse_rule_score,
        )
        review = apply_execution_claim_review(current_task, output, review)

        review = clean_review(review)

        if verbose:
            print("REVIEW:", clean_review(review, limit=5))

        # ---- Retry ----
        if should_retry(
            score=review["score"],
            is_reverse=intent.get("is_reverse_task", False),
            tags=review.get("tags", []),
        ):
            if verbose:
                print("🔁 Retry")

            if attempt >= settings.max_improvement_loops + 1:
                if verbose:
                    print("⚠️ Max attempts reached")
                save_attempt_memory(
                    store=store,
                    task=task,
                    plan=plan,
                    output=output,
                    review=review,
                    improvement_note=improvement_note or "max_attempts_reached",
                    is_reverse_task=is_reverse,
                    verbose=verbose,
                )
                record_run_log(
                    task=task,
                    mode="standard",
                    status="max_attempts_reached",
                    output=output,
                    plan=plan,
                    review=review,
                    intent=intent,
                    metadata={"attempt": attempt, "is_reverse_task": is_reverse},
                    verbose=verbose,
                )
                return output

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
                if verbose:
                    print("IMPROVER ERROR:", e)
                improvement_note = build_improvement_note(review)

            continue

        else:
            if verbose:
                print("✅ Completed")
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
            return output

    return output


def run_agi_pipeline(task: str, verbose: bool = True) -> str:
    direct_result = run_direct_tool_if_supported(task, verbose=verbose)
    if direct_result is not None:
        record_run_log(
            task=task,
            mode="agi",
            status="direct_tool_completed",
            output=direct_result.output,
            tool_name=direct_result.tool_name,
            metadata={"tool_router": True},
            verbose=verbose,
        )
        return direct_result.output

    store = MemoryStore()
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
            if verbose:
                print("MEMORY ERROR:", e)

    current_instruction = task
    best_output = ""
    best_review: dict[str, Any] = {"score": 0, "strengths": [], "weaknesses": [], "tags": []}
    best_plan = ""

    if verbose:
        print("=== AGI MODE ===")
        print("注意: これは真のAGIではなく、目標分析・実行・反省を回すAGI風モードです。")
        print("=== INTENT ===")
        print(intent)

    for cycle in range(1, settings.max_agi_cycles + 1):
        if verbose:
            print("=" * 50)
            print(f"AGI cycle {cycle}")
            print("=" * 50)

        control_raw = run_agi_controller(
            task_text=current_instruction,
            memory_text=memory_text,
        )
        control = safe_parse_agi_control(control_raw)
        plan = format_agi_control(control)

        if verbose:
            print("CONTROL:", control)

        worker_task = f"""
現在の依頼:
{task}

AGI-mode制御指示:
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

        intent_review = clean_review(
            safe_parse_review(
                run_intent_critic(task_text=task, output_text=output)
            )
        )
        quality_review = clean_review(
            safe_parse_review(
                run_quality_critic(
                    task_text=task,
                    output_text=output,
                    is_reverse=is_reverse,
                )
            )
        )
        reverse_review = None
        reverse_rule_score = 0
        if is_reverse:
            reverse_review = clean_review(
                safe_parse_review(
                    run_reverse_critic(task_text=task, output_text=output)
                )
            )
            reverse_rule_score = score_reverse_output(output)["score"]

        code_score = score_code(output, is_reverse=is_reverse)
        review = clean_review(
            merge_reviews(
                intent_review=intent_review,
                quality_review=quality_review,
                reverse_review=reverse_review,
                intent=intent,
                code_score=code_score,
                reverse_rule_score=reverse_rule_score,
            )
        )
        review = clean_review(apply_execution_claim_review(task, output, review))

        reflection_raw = run_agi_reflector(
            task_text=task,
            plan_text=plan,
            output_text=output,
            review_text=str(review),
        )
        reflection = safe_parse_agi_reflection(reflection_raw)

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
            print("OUTPUT:", output)
            print("REVIEW:", clean_review(review, limit=5))
            print("REFLECTION:", reflection)

        if reflection.get("done") and combined_score >= 85:
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
        improvement_note="agi_mode_completed",
        is_reverse_task=is_reverse,
        verbose=verbose,
    )
    record_run_log(
        task=task,
        mode="agi",
        status="agi_mode_completed",
        output=best_output,
        plan=best_plan,
        review=best_review,
        intent=intent,
        metadata={"max_cycles": settings.max_agi_cycles, "is_reverse_task": is_reverse},
        verbose=verbose,
    )
    return best_output


# =========================
# CLI
# =========================
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--quiet", action="store_true", help="ログを抑えて最終結果だけを出力する")
    parser.add_argument(
        "--mode",
        choices=("standard", "agi"),
        default="standard",
        help="standard は通常パイプライン、agi はAGI風の目標分析・反省サイクルを使う",
    )
    return parser.parse_args()


def execute_task(task: str, mode: str, quiet: bool = False) -> str:
    def _run() -> str:
        if mode == "agi":
            return run_agi_pipeline(task=task, verbose=not quiet)
        return run_pipeline(task=task, verbose=not quiet)

    if not quiet:
        return _run()

    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()
    with redirect_stdout(stdout_buffer), redirect_stderr(stderr_buffer):
        return _run()


def main():
    args = parse_args()
    result = execute_task(task=args.task, mode=args.mode, quiet=args.quiet)
    print(result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
