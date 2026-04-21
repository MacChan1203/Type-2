from __future__ import annotations

import unittest

from evaluation.code_scorer import score_code
from evaluation.execution_claims import (
    apply_execution_claim_review,
    detect_unsupported_execution_claim,
)
from evaluation.merge_reviews import merge_reviews
from evaluation.python_verifier import verify_python_output
from evaluation.reverse_metrics import (
    detect_comment_code_mismatch,
    detect_overexplained_bug_comments,
    score_reverse_output,
)
from evaluation.scorer import safe_parse_review


class ReviewParserTests(unittest.TestCase):
    def test_review_parser_accepts_fenced_json(self) -> None:
        parsed = safe_parse_review(
            '```json\n{"score": "82", "strengths": ["ok"], "weaknesses": [], '
            '"tags": ["x"], "next_action": "go"}\n```'
        )
        # normalize_review で 5 刻みに丸めるため 82 → 80
        self.assertEqual(parsed["score"], 80)
        self.assertEqual(parsed["strengths"], ["ok"])

    def test_safe_review_parser_rejects_natural_language_scores(self) -> None:
        parsed = safe_parse_review("Score: 95\nStrengths: great")

        self.assertEqual(parsed["score"], 50)
        self.assertIn("parse_error", parsed["tags"])


class CodeScorerTests(unittest.TestCase):
    def test_code_score_does_not_penalize_non_code_normal_output(self) -> None:
        self.assertGreater(score_code("これは設計方針の説明です。十分な長さの通常回答です。", is_reverse=False), 60)
        self.assertEqual(score_code("これは設計方針の説明です。", is_reverse=True), 0)

    def test_code_score_penalizes_invalid_plain_python(self) -> None:
        self.assertLess(score_code("def broken(:\n    return 1", is_reverse=False), 70)


class PythonVerifierTests(unittest.TestCase):
    def test_python_verifier_detects_invalid_fenced_code(self) -> None:
        result = verify_python_output("```python\ndef broken(:\n    return 1\n```")

        self.assertTrue(result["checked"])
        self.assertFalse(result["ok"])
        self.assertIn("構文エラー", result["error"])


class ReverseMetricsTests(unittest.TestCase):
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

    def test_reverse_score_does_not_count_normal_string_cleanup_as_data_bug(self) -> None:
        code = """
def normalize(value):
    return value.strip().lower()
"""
        result = score_reverse_output(code)

        self.assertNotIn("data_bug", result["tags"])

    def test_reverse_score_penalizes_overexplained_bug_comments(self) -> None:
        code = """
def process(items):
    # 意図的なバグ: 実行時エラーを起こす
    # 欠陥: 状態管理が壊れている
    # セキュリティリスク: 情報漏えいする
    return items[99]
"""
        result = score_reverse_output(code)

        self.assertTrue(detect_overexplained_bug_comments(code))
        self.assertIn("overexplained_bug_comments", result["tags"])

    def test_reverse_score_detects_comment_code_mismatch(self) -> None:
        code = """
def collect(values):
    result = []
    # バグの埋め込み: list.appendではなく、インデックス指定で誤って書き込む
    result.append(values)
    return result
"""
        result = score_reverse_output(code)

        self.assertTrue(detect_comment_code_mismatch(code))
        self.assertIn("comment_code_mismatch", result["tags"])


class MergeReviewsTests(unittest.TestCase):
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

    def test_merge_reviews_caps_parse_error_scores(self) -> None:
        # 部分的な parse_error は信頼度を下げるが、他Criticが高評価なら合格圏に残す
        partial = merge_reviews(
            intent_review={"score": 95, "strengths": [], "weaknesses": [], "tags": ["parse_error"]},
            quality_review={"score": 95, "strengths": [], "weaknesses": [], "tags": []},
            reverse_review=None,
            intent={"is_reverse_task": False},
            code_score=95,
        )
        self.assertLessEqual(partial["score"], 75)
        self.assertGreater(partial["score"], 60)

        # 全Critic parse失敗時は合格扱いにしない
        total = merge_reviews(
            intent_review={"score": 50, "strengths": [], "weaknesses": [], "tags": ["parse_error"]},
            quality_review={"score": 50, "strengths": [], "weaknesses": [], "tags": ["parse_error"]},
            reverse_review=None,
            intent={"is_reverse_task": False},
            code_score=50,
        )
        self.assertLessEqual(total["score"], 50)


class ExecutionClaimTests(unittest.TestCase):
    def test_execution_claim_detector_rejects_fake_external_success(self) -> None:
        task = "Hacker Newsの最初のニュースを日本語にして表示して"
        output = "取得しました。翻訳しました。URL: https://example.com/hn-story"

        self.assertTrue(detect_unsupported_execution_claim(task, output))

    def test_execution_claim_detector_accepts_real_evidence_or_failure(self) -> None:
        task = "Hacker Newsの最初のニュースを日本語にして表示して"
        success_output = (
            "取得しました。\n"
            "実行時刻: 2026-04-21T12:00:00\n"
            "URL:\nhttps://news.ycombinator.com/item?id=123456"
        )
        failure_output = "ステータス: 失敗\n理由:\nAPI timeout"

        self.assertFalse(detect_unsupported_execution_claim(task, success_output))
        self.assertFalse(detect_unsupported_execution_claim(task, failure_output))

    def test_execution_claim_detector_rejects_url_without_runtime_evidence(self) -> None:
        task = "最新ニュースを取得して"
        output = "取得しました。URL: https://news.ycombinator.com/item?id=123456"

        self.assertTrue(detect_unsupported_execution_claim(task, output))

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


if __name__ == "__main__":
    unittest.main()
