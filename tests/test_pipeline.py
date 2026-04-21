from __future__ import annotations

from contextlib import redirect_stderr, redirect_stdout
import io
import sys
import unittest
from unittest.mock import patch

from evaluation.python_verifier import verify_python_output
from concurrent.futures import Future

from main import (
    DisabledMemoryStore,
    _collect_critic,
    build_current_task,
    build_memory_store,
    execute_task,
    main as cli_main,
    parse_intent,
    run_reflective_pipeline,
    run_pipeline,
    safe_parse_reflective_control,
    safe_parse_reflective_reflection,
    should_use_memory,
)


class MemoryUsageTests(unittest.TestCase):
    def test_simple_code_task_skips_memory(self) -> None:
        self.assertFalse(should_use_memory("PythonでFizzBuzzを書いて"))
        self.assertTrue(should_use_memory("わざとバグを含むPythonコードを書いて", is_reverse=True))


class TaskBuildingTests(unittest.TestCase):
    def test_build_current_task_does_not_accumulate_old_notes(self) -> None:
        base = "FizzBuzzを書いて"
        first = build_current_task(base, "改善A")
        second = build_current_task(base, "改善B")
        self.assertIn("改善A", first)
        self.assertNotIn("改善A", second)
        self.assertIn("改善B", second)


class IntentParserTests(unittest.TestCase):
    def test_parse_intent_falls_back_when_llm_fails(self) -> None:
        def boom(task_text: str) -> str:
            raise RuntimeError("llm down")

        with patch("main.run_intent_checker", boom):
            intent = parse_intent("わざとバグを入れて", verbose=False)

        self.assertTrue(intent["is_reverse_task"])


class ReflectiveParserTests(unittest.TestCase):
    def test_reflective_json_parsers_fallback_safely(self) -> None:
        control = safe_parse_reflective_control("not json")
        reflection = safe_parse_reflective_reflection("not json")
        self.assertIn("worker_instruction", control)
        self.assertFalse(reflection["done"])
        self.assertEqual(reflection["score"], 50)

    def test_reflective_reflection_parser_coerces_string_false(self) -> None:
        reflection = safe_parse_reflective_reflection(
            '{"score": 80, "done": "false", "strengths": [], "weaknesses": [], "next_instruction": "continue"}'
        )

        self.assertFalse(reflection["done"])


class PipelineTests(unittest.TestCase):
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

    def test_run_pipeline_returns_reverse_fallback_when_worker_never_outputs_code(self) -> None:
        class FakeStore:
            def format_memories_for_prompt(self, query: str, include_failures: bool = True) -> str:
                return ""

            def add_memory(self, **kwargs):
                return "memory-id"

        review_json = '{"score":30,"strengths":[],"weaknesses":["コードがない"],"tags":["missing_code"],"next_action":"コードを返す"}'

        with (
            patch("main.MemoryStore", FakeStore),
            patch("main.run_intent_checker", return_value='{"intent":"reverse","is_reverse_task":true,"success_criteria":"buggy code"}'),
            patch("main.run_worker", return_value="【出力の準備中】Pythonコードを生成します。"),
            patch("main.run_intent_critic", return_value=review_json),
            patch("main.run_quality_critic", return_value=review_json),
            patch("main.run_reverse_critic", return_value=review_json),
            patch("main.run_improver", return_value='{"priority_fixes":["コード"],"keep":[],"improved_prompt":"Pythonコードのみを返す"}'),
            patch("main.record_run_log") as record_run_log,
        ):
            result = run_pipeline("わざとバグだらけのPythonコードを書いてください", verbose=False)

        self.assertIn("class DataProcessor", result)
        self.assertTrue(verify_python_output(result)["ok"])
        self.assertEqual(record_run_log.call_args.kwargs["status"], "reverse_local_fallback_completed")

    def test_unsupported_external_task_does_not_call_worker(self) -> None:
        with (
            patch("main.run_worker", side_effect=AssertionError("worker should not run")),
            patch("main.record_run_log") as record_run_log,
        ):
            result = run_pipeline("最新のPythonリリース情報を検索して", verbose=False)

        self.assertIn("ステータス: 失敗", result)
        record_run_log.assert_called_once()
        self.assertEqual(record_run_log.call_args.kwargs["tool_name"], "unsupported_external_execution")


class ExecuteTaskTests(unittest.TestCase):
    def test_execute_task_quiet_suppresses_pipeline_stdout(self) -> None:
        def noisy_pipeline(task: str, verbose: bool = True) -> str:
            print("NOISY INTERNAL LOG")
            return "FINAL RESULT"

        with patch("main.run_pipeline", noisy_pipeline):
            result = execute_task("説明を書いて", mode="standard", quiet=True)

        self.assertEqual(result, "FINAL RESULT")

    def test_execute_task_quiet_preserves_captured_output_on_error(self) -> None:
        def noisy_failure(task: str, verbose: bool = True) -> str:
            print("NOISY BEFORE FAIL")
            raise RuntimeError("boom")

        with patch("main.run_pipeline", noisy_failure):
            with self.assertRaisesRegex(RuntimeError, "NOISY BEFORE FAIL"):
                execute_task("説明を書いて", mode="standard", quiet=True)

    def test_execute_task_reflective_mode_aliases_reflective_pipeline(self) -> None:
        with patch("main.run_reflective_pipeline", return_value="REFLECTED") as mock_pipeline:
            result = execute_task("説明を書いて", mode="reflective", quiet=False)

        self.assertEqual(result, "REFLECTED")
        mock_pipeline.assert_called_once()


class CliTests(unittest.TestCase):
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
        # --quiet は stdout の詳細出力のみ抑制する。stderr（エラー・警告）は通過させる
        self.assertIn("NOISY STDERR", stderr.getvalue())


class ReflectivePipelineTests(unittest.TestCase):
    def test_run_reflective_pipeline_smoke_with_mocked_agents(self) -> None:
        class FakeStore:
            def format_memories_for_prompt(self, query: str, include_failures: bool = True) -> str:
                return ""

            def add_memory(self, **kwargs):
                return "memory-id"

        with (
            patch("main.MemoryStore", FakeStore),
            patch("main.run_intent_checker", return_value='{"intent":"write","is_reverse_task":false,"success_criteria":"done"}'),
            patch(
                "main.run_reflective_controller",
                return_value='{"objective":"write","assumptions":[],"subgoals":["answer"],"risks":[],"success_criteria":["done"],"worker_instruction":"直接答える"}',
            ),
            patch("main.run_worker", return_value="AGI mode smoke output"),
            patch("main.run_intent_critic", return_value='{"score":90,"strengths":["ok"],"weaknesses":[],"tags":["instruction_following"],"next_action":"done"}'),
            patch("main.run_quality_critic", return_value='{"score":90,"strengths":["ok"],"weaknesses":[],"tags":["practical"],"next_action":"done"}'),
            patch("main.run_reflective_reflector", return_value='{"score":90,"done":true,"strengths":["ok"],"weaknesses":[],"next_instruction":""}'),
            patch("main.record_run_log"),
        ):
            result = run_reflective_pipeline("説明を書いて", verbose=False)

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
            result = run_reflective_pipeline("Hacker Newsの最初のニュースを日本語にして表示して", verbose=False)

        self.assertIn("日本語タイトル", result)
        self.assertIn("本文:", result)
        self.assertIn("日本語の本文です。", result)
        self.assertIn("https://example.com/story", result)
        record_run_log.assert_called_once()
        self.assertEqual(record_run_log.call_args.kwargs["tool_name"], "hacker_news_top_japanese")


class CollectCriticTests(unittest.TestCase):
    def _make_future(self, result=None, exc=None) -> Future:
        fut: Future = Future()
        if exc is not None:
            fut.set_exception(exc)
        else:
            fut.set_result(result)
        return fut

    def test_collect_critic_parses_valid_json(self) -> None:
        fut = self._make_future('{"score":85,"strengths":["ok"],"weaknesses":[],"tags":[],"next_action":"done"}')
        review = _collect_critic(fut, "test_critic")
        self.assertEqual(review["score"], 85)

    def test_collect_critic_falls_back_on_exception(self) -> None:
        fut = self._make_future(exc=RuntimeError("llm crashed"))
        review = _collect_critic(fut, "test_critic")
        self.assertIn("score", review)
        self.assertLessEqual(review["score"], 50)
        self.assertIn("parse_error", review.get("tags", []))

    def test_collect_critic_falls_back_on_invalid_json(self) -> None:
        fut = self._make_future("not valid json at all")
        review = _collect_critic(fut, "bad_critic")
        self.assertIn("score", review)


class MemoryStoreFallbackTests(unittest.TestCase):
    def test_memory_store_falls_back_when_initialization_fails(self) -> None:
        with patch("main.MemoryStore", side_effect=RuntimeError("chroma down")):
            store = build_memory_store(verbose=False)

        self.assertIsInstance(store, DisabledMemoryStore)
        self.assertEqual(store.format_memories_for_prompt("task"), "")


if __name__ == "__main__":
    unittest.main()
