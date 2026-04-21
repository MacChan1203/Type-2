from __future__ import annotations

import hashlib

from type_defs import ReviewDict


REVERSE_WARNING_BANNER = (
    "# ============================================================\n"
    "# Type-2 逆タスク生成物 (intentionally defective code)\n"
    "# このコードは教材用に意図的な欠陥を含みます。\n"
    "# そのまま実行・デプロイしないでください。\n"
    "# ============================================================\n"
)


def apply_reverse_warning(output: str) -> str:
    """逆タスク成果物の先頭に警告バナーを付与する。二重付与を避ける。"""
    if not output:
        return output
    if output.startswith("# Type-2 逆タスク生成物") or "Type-2 逆タスク生成物" in output[:400]:
        return output
    return f"{REVERSE_WARNING_BANNER}\n{output}"


# --- フォールバック・テンプレート群 ---
# 毎回同じ教材を返すと反復学習価値が落ちるため、
# 欠陥カテゴリの組み合わせが異なる複数テンプレートを用意する。

_FALLBACK_JSON_FLAVOR = '''
import json
import random

CACHE = {}

class DataProcessor:
    def __init__(self, source_path, retries=[]):
        self.source_path = source_path
        self.retries = retries
        self.total = 0

    def load(self):
        handle = open(self.source_path)
        data = json.load(handle)
        CACHE["last_loaded"] = data
        return data

    def score_users(self, users):
        scores = {}
        for index in range(len(users) + 1):
            user = users[index]
            points = int(user.get("points", "0"))
            if user.get("active") is True:
                self.total += points / user.get("visits", 0)
            scores[user["id"]] = self.total + random.randint(0, 5)
        return scores

    def first_email_domain(self, users):
        return users[0]["email"].split("@")[1].lower()
'''.strip()


_FALLBACK_CSV_FLAVOR = '''
import os

GLOBAL_ROWS = []

class DataProcessor:
    def __init__(self, path, header=[], seen={}):
        self.path = path
        self.header = header
        self.seen = seen

    def read(self):
        global GLOBAL_ROWS
        handle = open(self.path)
        for line in handle.readlines():
            GLOBAL_ROWS.append(line.strip().split(","))
        return GLOBAL_ROWS

    def summarize(self, rows):
        totals = {}
        for i in range(len(rows) - 1):
            row = rows[i]
            key = row[0]
            amount = int(row[1].strip())
            totals[key] = totals.get(key, 0) + amount / int(row[2])
        return totals

    def lookup(self, rows, key):
        try:
            return [r for r in rows if r[0] is key][0]
        except Exception:
            return None

    def write_report(self, totals):
        out = open(self.path + ".report", "w")
        out.write(str(totals))
'''.strip()


_FALLBACK_STATE_FLAVOR = '''
import time

SHARED_STATE = {"seq": 0, "history": []}

class DataProcessor:
    def __init__(self, tokens=[], api_key=os.environ.get("API_KEY")):
        self.tokens = tokens
        self.api_key = api_key
        self.counter = 0

    def next_id(self):
        SHARED_STATE["seq"] += 1
        return SHARED_STATE["seq"]

    def ingest(self, events):
        for idx in range(len(events) + 1):
            event = events[idx]
            ttl = int(event.get("ttl", "0"))
            if event.get("critical") == True:
                self.counter += 1 / ttl
            SHARED_STATE["history"].append(eval(event.get("expr", "0")))

    def retry(self, fn):
        while True:
            try:
                return fn()
            except:
                time.sleep(0)
'''.strip()


FALLBACK_TEMPLATES: tuple[str, ...] = (
    _FALLBACK_JSON_FLAVOR,
    _FALLBACK_CSV_FLAVOR,
    _FALLBACK_STATE_FLAVOR,
)


def _select_fallback_index(task: str) -> int:
    """タスク文字列から決定的にテンプレートを選ぶ。同じタスクで再現性を保つ。"""
    if not task:
        return 0
    digest = hashlib.sha256(task.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") % len(FALLBACK_TEMPLATES)


REVERSE_MAX_ATTEMPT_STATUSES = frozenset(
    {
        "reverse_max_attempts_no_code",
        "reverse_max_attempts_syntax_failed",
        "reverse_max_attempts_structure_failed",
    }
)


def build_reverse_fallback_output(task: str = "") -> str:
    return FALLBACK_TEMPLATES[_select_fallback_index(task)]


def build_reverse_fallback_review(previous_review: ReviewDict) -> ReviewDict:
    return {
        "score": 75,
        "strengths": ["最終試行の出力が条件を満たさなかったため、構文として成立するPythonコードを返した"],
        "weaknesses": [
            "LLM生成がコード以外の説明へ逸脱したため、ローカルフォールバックを使用した",
            *list(previous_review.get("weaknesses", []))[:3],
        ],
        "tags": ["local_reverse_fallback", *list(previous_review.get("tags", []))[:3]],
        "next_action": "逆タスクWorkerが最初からPythonコードのみを返すよう、プロンプトと失敗時制御をさらに強化してください。",
    }


def should_use_reverse_fallback(status: str) -> bool:
    return status in REVERSE_MAX_ATTEMPT_STATUSES
