from __future__ import annotations

from datetime import datetime, timedelta
import json
from pathlib import Path
import unittest
from unittest.mock import patch
from tempfile import TemporaryDirectory

from config import int_env
from memory.chroma_store import MemoryStore
from tools.hacker_news import (
    is_hacker_news_task,
    parse_target_time,
    wait_until_target_time,
)
from tools.router import ToolRouter, ToolSpec, build_default_router
from utils.run_logger import build_run_log, prune_run_logs, save_run_log
from utils.safe_http import is_safe_article_url, safe_fetch_json


class ConfigTests(unittest.TestCase):
    def test_int_env_falls_back_for_invalid_values(self) -> None:
        with patch.dict("os.environ", {"BAD_INT": "nope", "NEG_INT": "-3"}):
            self.assertEqual(int_env("BAD_INT", 7), 7)
            self.assertEqual(int_env("NEG_INT", 7, minimum=1), 1)


class HackerNewsTests(unittest.TestCase):
    def test_hacker_news_task_detection_and_time_parse(self) -> None:
        self.assertTrue(is_hacker_news_task("午後8時54分になったら、Hacker Newsの最初のニュースを日本語にして表示して"))
        parsed = parse_target_time(
            "午後8時54分になったら実行",
            now=datetime(2026, 4, 20, 10, 0),
        )
        self.assertEqual(parsed, datetime(2026, 4, 20, 20, 54))

    def test_hacker_news_time_parse_uses_next_occurrence_when_past(self) -> None:
        parsed = parse_target_time(
            "午前9時になったら実行",
            now=datetime(2026, 4, 20, 10, 0),
        )

        self.assertEqual(parsed, datetime(2026, 4, 21, 9, 0))

    def test_hacker_news_articleis_safe_article_urlty_filter(self) -> None:
        self.assertTrue(is_safe_article_url("https://example.org/story"))
        self.assertFalse(is_safe_article_url("file:///etc/passwd"))
        self.assertFalse(is_safe_article_url("http://localhost:8080"))
        self.assertFalse(is_safe_article_url("http://127.0.0.1:8080"))

    def test_hacker_news_wait_rejects_long_blocking_sleep(self) -> None:
        target = datetime.now() + timedelta(minutes=10)

        with self.assertRaises(TimeoutError):
            wait_until_target_time(
                target,
                verbose=False,
                sleep_fn=lambda seconds: self.fail("sleep should not be called"),
                max_wait_seconds=1,
            )


class UrlSafetyTests(unittest.TestCase):
    """SSRF protection in utils/safe_http.is_safe_article_url."""

    def test_public_https_url_is_allowed(self) -> None:
        self.assertTrue(is_safe_article_url("https://news.ycombinator.com/item?id=1"))
        self.assertTrue(is_safe_article_url("http://example.org/path"))

    def test_public_ip_literal_is_allowed(self) -> None:
        self.assertTrue(is_safe_article_url("https://8.8.8.8/path"))

    def test_file_scheme_is_blocked(self) -> None:
        self.assertFalse(is_safe_article_url("file:///etc/passwd"))

    def test_ftp_scheme_is_blocked(self) -> None:
        self.assertFalse(is_safe_article_url("ftp://example.com/file"))

    def test_localhost_literal_is_blocked(self) -> None:
        self.assertFalse(is_safe_article_url("http://localhost/admin"))
        self.assertFalse(is_safe_article_url("http://localhost:8080"))

    def test_loopback_ipv4_is_blocked(self) -> None:
        self.assertFalse(is_safe_article_url("http://127.0.0.1/"))
        self.assertFalse(is_safe_article_url("http://127.0.0.1:8080"))

    def test_private_ipv4_ranges_are_blocked(self) -> None:
        self.assertFalse(is_safe_article_url("http://192.168.1.1/"))
        self.assertFalse(is_safe_article_url("http://10.0.0.1/"))
        self.assertFalse(is_safe_article_url("http://172.16.0.1/"))

    def test_link_local_aws_metadata_is_blocked(self) -> None:
        # 169.254.169.254 is the AWS EC2 metadata endpoint
        self.assertFalse(is_safe_article_url("http://169.254.169.254/latest/meta-data/"))

    def test_ipv6_loopback_is_blocked(self) -> None:
        self.assertFalse(is_safe_article_url("http://[::1]/"))

    def test_empty_url_is_blocked(self) -> None:
        self.assertFalse(is_safe_article_url(""))
        self.assertFalse(is_safe_article_url("http://"))


class SafeFetchJsonTests(unittest.TestCase):
    """safe_fetch_json のSSRF保護とJSON解析を検証する。"""

    def test_safe_fetch_json_rejects_internal_url(self) -> None:
        with self.assertRaises(ValueError):
            safe_fetch_json("http://localhost/api.json")

    def test_safe_fetch_json_rejects_private_ip(self) -> None:
        with self.assertRaises(ValueError):
            safe_fetch_json("http://192.168.1.1/data.json")

    def test_safe_fetch_json_parses_response(self) -> None:
        payload = json.dumps({"key": "value"}).encode("utf-8")

        class FakeResponse:
            def read(self, max_bytes: int) -> bytes:
                return payload
            def __enter__(self):
                return self
            def __exit__(self, *_):
                pass

        with patch("utils.safe_http.safe_urlopen", return_value=FakeResponse()):
            result = safe_fetch_json("https://example.org/data.json")

        self.assertEqual(result, {"key": "value"})


class ToolRouterTests(unittest.TestCase):
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
        self.assertEqual(result.status, "SUCCESS")
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

    def test_default_router_rejects_unsupported_external_task(self) -> None:
        router = build_default_router()

        result = router.run("今日の東京の天気を検索して日本語で表示して", verbose=False)

        self.assertIsNotNone(result)
        self.assertEqual(result.tool_name, "unsupported_external_execution")
        self.assertEqual(result.status, "FAILURE")
        self.assertIn("ステータス: 失敗", result.output)
        self.assertIn("専用ツールが登録されていません", result.output)

    def test_default_router_does_not_reject_web_app_generation(self) -> None:
        router = build_default_router()

        self.assertIsNone(router.run("Webアプリのサンプルコードを書いて", verbose=False))


class MemoryStoreTests(unittest.TestCase):
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

    def test_memory_document_redacts_obvious_secrets(self) -> None:
        store = object.__new__(MemoryStore)
        doc = store._build_document(
            task="token: abcdefghijklmnop",
            plan="secret=plan-secret",
            output="password=hunter2",
            review={
                "score": 90,
                "strengths": ["api_key=review-secret"],
                "weaknesses": [],
                "next_action": "done",
            },
            improvement_note="note",
        )

        self.assertNotIn("abcdefghijklmnop", doc)
        self.assertNotIn("plan-secret", doc)
        self.assertNotIn("hunter2", doc)
        self.assertNotIn("review-secret", doc)
        self.assertIn("[REDACTED]", doc)

    def test_memory_filter_removes_legacy_full_outputs(self) -> None:
        store = object.__new__(MemoryStore)
        memories = [
            "[Task]\nnew\n\n[Output Summary]\nshort",
            "[Task]\nold\n\n[Output]\nfull",
            "task:\nold\n\noutput:\nfull",
        ]
        self.assertEqual(store._filter_usable_memories(memories), ["[Task]\nnew\n\n[Output Summary]\nshort"])

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


class RunLoggerTests(unittest.TestCase):
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

    def test_prune_run_logs_keeps_newest_files(self) -> None:
        with TemporaryDirectory() as tmpdir:
            for idx in range(3):
                path = save_run_log(
                    task=f"task {idx}",
                    mode="standard",
                    status="completed",
                    output="result",
                    runs_path=tmpdir,
                )
                path.touch()

            removed = prune_run_logs(keep=1, runs_path=tmpdir)
            remaining = list(Path(tmpdir).glob("*.json"))

        self.assertEqual(len(removed), 2)
        self.assertEqual(len(remaining), 1)
        self.assertTrue(all(name.endswith(".json") for name in removed))

    def test_run_log_redacts_obvious_secrets(self) -> None:
        log = build_run_log(
            task="API_KEY=super-secret-value を使って説明して",
            mode="standard",
            status="completed",
            output="token: abcdefghijklmnop",
        )

        serialized = json.dumps(log, ensure_ascii=False)
        self.assertNotIn("super-secret-value", serialized)
        self.assertNotIn("abcdefghijklmnop", serialized)
        self.assertIn("[REDACTED]", serialized)


if __name__ == "__main__":
    unittest.main()
