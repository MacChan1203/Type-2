from __future__ import annotations

from datetime import datetime
import re


# 実ネットワーク/時刻同期が必要な依頼にマッチさせる。
# 単に「時刻」「取得」「API」が含まれるだけでは発火しないよう、前後文脈を要求する。
EXTERNAL_EXECUTION_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"https?://", re.IGNORECASE),
    re.compile(r"(Hacker\s*News|hackernews|HN)", re.IGNORECASE),
    re.compile(r"(スクレイピング|クロール|Webスクレイピング|Webクロール)"),
    re.compile(r"(ウェブ|Web|インターネット|サイト|外部API|外部サイト)(から|を)(取得|ダウンロード|フェッチ|叩|呼)"),
    re.compile(r"(最新|今日|本日|現在|直近)の[^。、\n]{0,25}(ニュース|情報|記事|天気|株価|為替|レート|トピック|リリース)"),
    re.compile(r"最新の[^。、\n]{0,25}を(取得|検索|調べ|教え)"),
    re.compile(r"\d{1,2}\s*(時|:)\s*\d{0,2}\s*分?\s*(以降|になったら|に実行|に開始)"),
    re.compile(r"(指定時刻|現在時刻)(に|で).{0,10}(実行|開始|取得)"),
    re.compile(r"(現在|今|いま).{0,5}(時刻|時間)を(表示|知らせ|教え)"),
)


def is_external_execution_task(task: str) -> bool:
    return any(pattern.search(task) for pattern in EXTERNAL_EXECUTION_PATTERNS)


def run_unsupported_external_task(task: str, verbose: bool = True) -> str:
    return f"""
【外部実行タスク】
ステータス: 失敗
実行時刻: {datetime.now().isoformat(timespec="seconds")}

本文:
この依頼は外部取得・検索・時刻実行などの実処理が必要ですが、現在のToolRouterには対応する専用ツールが登録されていません。

理由:
未対応の外部実行タスクをLLM生成だけで成功扱いにしないため、実行を中止しました。

次の対応:
tools/router.py に専用 ToolSpec を追加し、実URL・取得ID・失敗理由などの検証可能な証拠を返す実装を用意してください。
""".strip()
