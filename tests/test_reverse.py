from __future__ import annotations

import unittest

from evaluation.python_verifier import verify_python_output
from evaluation.reverse_fallback import FALLBACK_TEMPLATES, build_reverse_fallback_output
from evaluation.reverse_normalize import normalize_reverse_output
from main import format_max_attempt_message, format_retry_message, has_reverse_hint, max_attempt_status


class ReverseHintTests(unittest.TestCase):
    def test_reverse_hint_requires_reverse_and_negative_words(self) -> None:
        self.assertTrue(has_reverse_hint("わざとバグを入れて"))
        self.assertFalse(has_reverse_hint("このバグを直して"))


class RetryMessageTests(unittest.TestCase):
    def test_reverse_retry_messages_are_distinct_by_context(self) -> None:
        msg_reverse_parse = format_retry_message(True, ["parse_error"])
        msg_reverse_plain = format_retry_message(True, [])
        msg_normal = format_retry_message(False, [])

        # parse_error gets a distinct message compared to plain reverse retry
        self.assertNotEqual(msg_reverse_parse, msg_reverse_plain)
        # reverse retry messages differ from normal retry
        self.assertNotEqual(msg_reverse_parse, msg_normal)
        self.assertNotEqual(msg_reverse_plain, msg_normal)

    def test_max_attempt_status_returns_correct_codes(self) -> None:
        self.assertEqual(
            max_attempt_status(True, "code", ["parse_error"]),
            "reverse_max_attempts_no_code",
        )
        self.assertEqual(
            max_attempt_status(True, "def broken():\n    return 1", ["parse_error"]),
            "reverse_best_effort_completed",
        )

    def test_reverse_max_attempts_uses_no_code_status_for_meta_output(self) -> None:
        status = max_attempt_status(
            True,
            "【出力の準備中】Pythonコードを生成します。",
            ["missing_code", "reverse_structure_fail"],
        )

        self.assertEqual(status, "reverse_max_attempts_no_code")

    def test_max_attempt_messages_are_distinct_by_outcome(self) -> None:
        msg_best_effort = format_max_attempt_message(True, "reverse_best_effort_completed")
        msg_max_reached = format_max_attempt_message(False, "max_attempts_reached")
        msg_reverse_fail = format_max_attempt_message(True, "reverse_max_attempts_no_code")

        # success path differs from failure path
        self.assertNotEqual(msg_best_effort, msg_max_reached)
        # best effort differs from no-code failure
        self.assertNotEqual(msg_best_effort, msg_reverse_fail)


class ReverseFallbackTests(unittest.TestCase):
    def test_all_fallback_templates_are_valid_python_with_dataprocessor(self) -> None:
        for idx, template in enumerate(FALLBACK_TEMPLATES):
            with self.subTest(idx=idx):
                result = verify_python_output(template)
                self.assertTrue(result["checked"], f"template {idx} not recognized as Python")
                self.assertTrue(result["ok"], f"template {idx} syntax error: {result.get('error')}")
                self.assertIn("class DataProcessor", template)
                self.assertNotIn("意図的", template)

    def test_reverse_fallback_output_selects_deterministically_by_task(self) -> None:
        out_a = build_reverse_fallback_output("task A")
        out_b = build_reverse_fallback_output("task A")
        self.assertEqual(out_a, out_b)
        self.assertIn(out_a, FALLBACK_TEMPLATES)


class ReverseNormalizeTests(unittest.TestCase):
    def test_reverse_output_normalization_extracts_code_only(self) -> None:
        output = """```python
def broken():
    return 1
```

### 解説
これは説明です。
"""

        self.assertEqual(normalize_reverse_output(output), "def broken():\n    return 1")

    def test_reverse_output_normalization_removes_unused_token(self) -> None:
        output = "import os\nprint(os.getcwd())\n<unused56>"

        self.assertEqual(normalize_reverse_output(output), "import os\nprint(os.getcwd())")

    def test_reverse_output_normalization_removes_noisy_comments_and_docstrings(self) -> None:
        output = '''class Example:
    """意図的に複数の脆弱性を含んでいる。"""
    def run(self):
        # 意図的なバグ: 情報漏えい
        return "secret"
'''

        self.assertEqual(normalize_reverse_output(output), 'class Example:\n    def run(self):\n        return "secret"')

    def test_reverse_output_normalization_removes_main_side_effect_demo(self) -> None:
        output = '''def write_bad_file():
    open("bad.txt", "w").write("x")

if __name__ == "__main__":
    write_bad_file()
'''

        self.assertEqual(normalize_reverse_output(output), 'def write_bad_file():\n    open("bad.txt", "w").write("x")')

    def test_reverse_output_normalization_removes_separator_comments(self) -> None:
        output = '''# =============================================================================
def run():
    return 1
'''

        self.assertEqual(normalize_reverse_output(output), 'def run():\n    return 1')

    def test_reverse_output_normalization_removes_answer_comments(self) -> None:
        output = '''class DataProcessor:
    def __init__(self):
        # ユーザーIDが常にグローバルに影響を与える設計ミス
        self.user_id = "u1"
        # 文字列変換のエラーハンドリングが甘い
        self.count = 0
        # ユーザーからの入力をエスケープ処理していない
        self.raw = "x"
'''

        self.assertEqual(
            normalize_reverse_output(output),
            'class DataProcessor:\n    def __init__(self):\n        self.user_id = "u1"\n        self.count = 0\n        self.raw = "x"',
        )

    def test_reverse_output_normalization_removes_noisy_inline_comments(self) -> None:
        output = '''class DataProcessor:
    def __init__(self):
        self.processing_count = 0 # ステートの保持による意図的な情報漏洩ポイント
        self.name = "ok" # ordinary note
'''

        self.assertEqual(
            normalize_reverse_output(output),
            'class DataProcessor:\n    def __init__(self):\n        self.processing_count = 0\n        self.name = "ok"',
        )

    def test_reverse_output_normalization_removes_all_comments_and_docstrings(self) -> None:
        output = '''class DataProcessor:
    """データを処理するメインのクラス。"""
    def run(self, value):
        # 競合状態の核心
        text = "keep # inside string"
        return text, value # trailing explanation
'''

        self.assertEqual(
            normalize_reverse_output(output),
            'class DataProcessor:\n    def run(self, value):\n        text = "keep # inside string"\n        return text, value',
        )

    def test_reverse_output_normalization_trims_trailing_comment_block(self) -> None:
        output = '''def run():
    return 1

# メイン実行部
'''

        self.assertEqual(normalize_reverse_output(output), 'def run():\n    return 1')

    def test_reverse_output_normalization_removes_top_level_demo_without_main_guard(self) -> None:
        output = '''import hashlib

class DataProcessor:
    def run(self):
        return hashlib.sha1(b"x").hexdigest()

api_key = "supersecret"
processor = DataProcessor()
print(processor.run())
'''

        self.assertEqual(
            normalize_reverse_output(output),
            'import hashlib\n\nclass DataProcessor:\n    def run(self):\n        return hashlib.sha1(b"x").hexdigest()',
        )

    def test_reverse_output_normalization_removes_exploit_function(self) -> None:
        output = '''class SensitiveSystem:
    def run(self):
        return "ok"

def exploit_system(system):
    print("EXPLOITATION PHASE START")
    return system.run()
'''

        self.assertEqual(
            normalize_reverse_output(output),
            'class SensitiveSystem:\n    def run(self):\n        return "ok"',
        )

    def test_reverse_output_normalization_removes_noisy_print_lines(self) -> None:
        output = '''class SensitiveSystem:
    def run(self):
        print("[CRITICAL VULN] Potential SQL Injection attempt detected and executed!")
        print(f"[Thread x] SECURITY BYPASS DETECTED.")
        print("normal progress")
        return "ok"
'''

        self.assertEqual(
            normalize_reverse_output(output),
            'class SensitiveSystem:\n    def run(self):\n        print("normal progress")\n        return "ok"',
        )


if __name__ == "__main__":
    unittest.main()
