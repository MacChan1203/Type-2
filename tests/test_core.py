from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime
import io
import json
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from evaluation.code_scorer import score_code
from evaluation.execution_claims import (
    apply_execution_claim_review,
    detect_unsupported_execution_claim,
)
from evaluation.merge_reviews import merge_reviews
from evaluation.reverse_metrics import score_reverse_output
from evaluation.scorer import safe_parse_review
from main import (
    build_current_task,
    execute_task,
    has_reverse_hint,
    main as cli_main,
    parse_intent,
    run_agi_pipeline,
    run_pipeline,
    safe_parse_agi_control,
    safe_parse_agi_reflection,
    should_use_memory,
)
from memory.chroma_store import MemoryStore
from tools.hacker_news import is_hacker_news_task, parse_target_time
from tools.router import ToolRouter, ToolSpec, build_default_router
from utils.run_logger import save_run_log


class CoreTests(unittest.TestCase):
    def test_reverse_hint_requires_reverse_and_negative_words(self) -> None:
        self.assertTrue(has_reverse_hint("わざとバグを入れて"))
        self.assertFalse(has_reverse_hint("このバグを直して"))

    def test_build_current_task_does_not_accumulate_old_notes(self) -> None:
        base = "FizzBuzzを書いて"
        first = build_current_task(base, "改善A")
        second = build_current_task(base, "改善B")
        self.assertIn("改善A", first)
        self.assertNotIn("改善A", second)
        self.assertIn("改善B", second)

    def test_review_parser_accepts_fenced_json(self) -> None:
        parsed = safe_parse_review(
            '```json\n{"score": "82", "strengths": ["ok"], "weaknesses": [], '
            '"tags": ["x"], "next_action": "go"}\n```'
        )
        self.assertEqual(parsed["score"], 82)
        self.assertEqual(parsed["strengths"], ["ok"])

    def test_code_score_does_not_penalize_non_code_normal_output(self) -> None:
        self.assertGreater(score_code("これは設計方針の説明です。十分な長さの通常回答です。", is_reverse=False), 60)
        self.assertEqual(score_code("これは設計方針の説明です。", is_reverse=True), 0)

    def test_reverse_score_rewards_code_with_defects(self) -> None:
        code = """
import os
STATE = []

def load_value(items=[]):
    global STATE
    try:
        value = int(os.environ.get("TOKEN"))
        STATE.append(eval(str(value)))
    except Exception:
        pass
    return STATE[99]

def run():
    for i in range(len(STATE) - 1):
        if i is "0":
            return load_value()
    return None.missing

if __name__ == "__main__":
    print(run())
"""
        result = score_reverse_output(code)
        self.assertGreaterEqual(result["score"], 70)
        self.assertIn("security_bug", result["tags"])

    def test_merge_reviews_uses_reverse_rule_score_for_reverse_tasks(self) -> None:
        review = merge_reviews(
            intent_review={"score": 80, "strengths": [], "weaknesses": [], "tags": []},
            quality_review={"score": 80, "strengths": [], "weaknesses": [], "tags": []},
            reverse_review={"score": 80, "strengths": [], "weaknesses": [], "tags": []},
            intent={"is_reverse_task": True},
            code_score=80,
            reverse_rule_score=20,
        )
        self.assertIn("reverse_structure_fail", review["tags"])

    def test_execution_claim_detector_rejects_fake_external_success(self) -> None:
        task = "Hacker Newsの最初のニュースを日本語にして表示して"
        output = "取得しました。翻訳しました。URL: https://example.com/hn-story"

        self.assertTrue(detect_unsupported_execution_claim(task, output))

    def test_execution_claim_detector_accepts_real_evidence_or_failure(self) -> None:
        task = "Hacker Newsの最初のニュースを日本語にして表示して"
        success_output = "取得しました。\nURL:\nhttps://news.ycombinator.com/item?id=123456"
        failure_output = "ステータス: FAILURE\n理由:\nAPI timeout"

        self.assertFalse(detect_unsupported_execution_claim(task, success_output))
        self.assertFalse(detect_unsupported_execution_claim(task, failure_output))

    def test_execution_claim_review_caps_score_and_tags_issue(self) -> None:
        review = {
            "score": 92,
            "strengths": ["形式は整っている"],
            "weaknesses": [],
            "tags": ["practical"],
            "next_action": "done",
        }

        updated = apply_execution_claim_review(
            task="Hacker Newsの最初のニュースを日本語にして表示して",
            output="HN APIから取得しました。翻訳しました。（リンク: https://example.com/item?id=xxxxxxxx）",
            review=review,
        )

        self.assertEqual(updated["score"], 35)
        self.assertIn("unsupported_execution_claim", updated["tags"])
        self.assertIn("ToolRouter", updated["next_action"])

    def test_parse_intent_falls_back_when_llm_fails(self) -> None:
        def boom(task_text: str) -> str:
            raise RuntimeError("llm down")

        with patch("main.run_intent_checker", boom):
            intent = parse_intent("わざとバグを入れて", verbose=False)

        self.assertTrue(intent["is_reverse_task"])

    def test_memory_document_omits_full_output(self) -> None:
        store = object.__new__(MemoryStore)
        doc = store._build_document(
            task="task",
            plan="plan",
            output="\n".join(f"line {i}" for i in range(30)),
            review={
                "score": 90,
                "strengths": ["s"],
                "weaknesses": ["w"],
                "next_action": "n",
            },
            improvement_note="note",
        )
        self.assertIn("[Output Summary]", doc)
        self.assertNotIn("[Output]\n", doc)
        self.assertIn("lines omitted", doc)

    def test_memory_filter_removes_legacy_full_outputs(self) -> None:
        store = object.__new__(MemoryStore)
        memories = [
            "[Task]\nnew\n\n[Output Summary]\nshort",
            "[Task]\nold\n\n[Output]\nfull",
            "task:\nold\n\noutput:\nfull",
        ]
        self.assertEqual(store._filter_usable_memories(memories), ["[Task]\nnew\n\n[Output Summary]\nshort"])

    def test_simple_code_task_skips_memory(self) -> None:
        self.assertFalse(should_use_memory("PythonでFizzBuzzを書いて"))
        self.assertTrue(should_use_memory("わざとバグを含むPythonコードを書いて", is_reverse=True))

    def test_hacker_news_task_detection_and_time_parse(self) -> None:
        self.assertTrue(is_hacker_news_task("午後8時54分になったら、Hacker Newsの最初のニュースを日本語にして表示して"))
        parsed = parse_target_time(
            "午後8時54分になったら実行",
            now=datetime(2026, 4, 20, 10, 0),
        )
        self.assertEqual(parsed, datetime(2026, 4, 20, 20, 54))

    def test_memory_metadata_contains_operational_fields(self) -> None:
        store = object.__new__(MemoryStore)
        metadata = store._build_metadata(
            task="task",
            review={"score": 90, "next_action": "n"},
            improvement_note="note",
            is_reverse_task=True,
            model="local-model",
        )
        self.assertTrue(metadata["is_reverse_task"])
        self.assertEqual(metadata["model"], "local-model")
        self.assertIn("created_at", metadata)
        self.assertEqual(len(metadata["task_hash"]), 32)

    def test_run_pipeline_smoke_with_mocked_agents(self) -> None:
        class FakeStore:
            def __init__(self) -> None:
                self.saved = []

            def format_memories_for_prompt(self, query: str, include_failures: bool = True) -> str:
                return ""

            def add_memory(self, **kwargs):
                self.saved.append(kwargs)
                return "memory-id"

        with (
            patch("main.MemoryStore", FakeStore),
            patch("main.run_intent_checker", return_value='{"intent":"write","is_reverse_task":false,"success_criteria":"done"}'),
            patch("main.run_planner", return_value="plan"),
            patch("main.run_worker", return_value="十分な長さの通常回答です。依頼に沿って具体的に説明します。"),
            patch("main.run_intent_critic", return_value='{"score":90,"strengths":["ok"],"weaknesses":[],"tags":["instruction_following"],"next_action":"done"}'),
            patch("main.run_quality_critic", return_value='{"score":90,"strengths":["ok"],"weaknesses":[],"tags":["practical"],"next_action":"done"}'),
            patch("main.record_run_log"),
        ):
            result = run_pipeline("説明を書いて", verbose=False)

        self.assertIn("通常回答", result)

    def test_execute_task_quiet_suppresses_pipeline_stdout(self) -> None:
        def noisy_pipeline(task: str, verbose: bool = True) -> str:
            print("NOISY INTERNAL LOG")
            return "FINAL RESULT"

        with patch("main.run_pipeline", noisy_pipeline):
            result = execute_task("説明を書いて", mode="standard", quiet=True)

        self.assertEqual(result, "FINAL RESULT")

    def test_cli_quiet_prints_only_final_result(self) -> None:
        def noisy_pipeline(task: str, verbose: bool = True) -> str:
            print("NOISY STDOUT")
            print("NOISY STDERR", file=sys.stderr)
            return "FINAL ONLY"

        stdout = io.StringIO()
        stderr = io.StringIO()
        with (
            patch("sys.argv", ["main.py", "--quiet", "--task", "説明を書いて"]),
            patch("main.run_pipeline", noisy_pipeline),
            redirect_stdout(stdout),
            redirect_stderr(stderr),
        ):
            exit_code = cli_main()

        self.assertEqual(exit_code, 0)
        self.assertEqual(stdout.getvalue(), "FINAL ONLY\n")
        self.assertEqual(stderr.getvalue(), "")

    def test_agi_json_parsers_fallback_safely(self) -> None:
        control = safe_parse_agi_control("not json")
        reflection = safe_parse_agi_reflection("not json")
        self.assertIn("worker_instruction", control)
        self.assertFalse(reflection["done"])
        self.assertEqual(reflection["score"], 50)

    def test_run_agi_pipeline_smoke_with_mocked_agents(self) -> None:
        class FakeStore:
            def format_memories_for_prompt(self, query: str, include_failures: bool = True) -> str:
                return ""

            def add_memory(self, **kwargs):
                return "memory-id"

        with (
            patch("main.MemoryStore", FakeStore),
            patch("main.run_intent_checker", return_value='{"intent":"write","is_reverse_task":false,"success_criteria":"done"}'),
            patch(
                "main.run_agi_controller",
                return_value='{"objective":"write","assumptions":[],"subgoals":["answer"],"risks":[],"success_criteria":["done"],"worker_instruction":"直接答える"}',
            ),
            patch("main.run_worker", return_value="AGI mode smoke output"),
            patch("main.run_intent_critic", return_value='{"score":90,"strengths":["ok"],"weaknesses":[],"tags":["instruction_following"],"next_action":"done"}'),
            patch("main.run_quality_critic", return_value='{"score":90,"strengths":["ok"],"weaknesses":[],"tags":["practical"],"next_action":"done"}'),
            patch("main.run_agi_reflector", return_value='{"score":90,"done":true,"strengths":["ok"],"weaknesses":[],"next_instruction":""}'),
            patch("main.record_run_log"),
        ):
            result = run_agi_pipeline("説明を書いて", verbose=False)

        self.assertEqual(result, "AGI mode smoke output")

    def test_hacker_news_task_runs_directly_and_includes_body(self) -> None:
        story = {
            "id": 1,
            "title": "Original title",
            "url": "https://example.com/story",
            "hn_url": "https://news.ycombinator.com/item?id=1",
            "body": "Original body",
            "body_error": "",
        }
        translated = {
            "title_jp": "日本語タイトル",
            "body_jp": "日本語の本文です。",
        }

        with (
            patch("tools.hacker_news.fetch_hacker_news_top_story", return_value=story),
            patch("tools.hacker_news.translate_hn_story_to_japanese", return_value=translated),
            patch("main.run_worker", side_effect=AssertionError("worker should not run")),
            patch("main.record_run_log") as record_run_log,
        ):
            result = run_agi_pipeline("Hacker Newsの最初のニュースを日本語にして表示して", verbose=False)

        self.assertIn("日本語タイトル", result)
        self.assertIn("本文:", result)
        self.assertIn("日本語の本文です。", result)
        self.assertIn("https://example.com/story", result)
        record_run_log.assert_called_once()
        self.assertEqual(record_run_log.call_args.kwargs["tool_name"], "hacker_news_top_japanese")

    def test_tool_router_registers_and_runs_matching_tool(self) -> None:
        router = ToolRouter()
        router.register(
            ToolSpec(
                name="echo",
                description="test tool",
                can_handle=lambda task: "echo" in task,
                run=lambda task, verbose=True: f"ran: {task}",
            )
        )

        result = router.run("echo this", verbose=False)

        self.assertIsNotNone(result)
        self.assertEqual(result.tool_name, "echo")
        self.assertEqual(result.output, "ran: echo this")
        self.assertIsNone(router.run("ignore this", verbose=False))

    def test_tool_router_rejects_duplicate_names(self) -> None:
        router = ToolRouter(
            [
                ToolSpec(
                    name="same",
                    description="first",
                    can_handle=lambda task: False,
                    run=lambda task, verbose=True: "",
                )
            ]
        )

        with self.assertRaises(ValueError):
            router.register(
                ToolSpec(
                    name="same",
                    description="second",
                    can_handle=lambda task: False,
                    run=lambda task, verbose=True: "",
                )
            )

    def test_default_router_contains_hacker_news_tool(self) -> None:
        router = build_default_router()
        tool_names = [tool.name for tool in router.tools]

        self.assertIn("hacker_news_top_japanese", tool_names)
        self.assertIsNotNone(router.match("Hacker Newsの最初のニュースを日本語にして"))

    def test_save_run_log_writes_json_file(self) -> None:
        with TemporaryDirectory() as tmpdir:
            path = save_run_log(
                task="説明を書いて",
                mode="standard",
                status="completed",
                output="結果",
                plan="plan",
                review={"score": 90},
                intent={"intent": "説明"},
                tool_name=None,
                metadata={"attempt": 1},
                runs_path=tmpdir,
            )

            data = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(data["task"], "説明を書いて")
        self.assertEqual(data["mode"], "standard")
        self.assertEqual(data["status"], "completed")
        self.assertEqual(data["output"], "結果")
        self.assertEqual(data["review"]["score"], 90)
        self.assertEqual(data["metadata"]["attempt"], 1)
        self.assertTrue(data["run_id"])


if __name__ == "__main__":
    unittest.main()
