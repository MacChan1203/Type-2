from __future__ import annotations

from datetime import datetime
from html.parser import HTMLParser
import json
import re
import time
from typing import Any
from urllib.error import URLError
from urllib.request import Request, urlopen

from config import settings
from evaluation.scorer import extract_json_block


HN_TOP_STORIES_URL = "https://hacker-news.firebaseio.com/v0/topstories.json"
HN_ITEM_URL = "https://hacker-news.firebaseio.com/v0/item/{item_id}.json"
HN_ITEM_PAGE_URL = "https://news.ycombinator.com/item?id={item_id}"


class VisibleTextParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self._skip_depth = 0
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag.lower() in {"script", "style", "noscript"}:
            self._skip_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag.lower() in {"script", "style", "noscript"} and self._skip_depth:
            self._skip_depth -= 1

    def handle_data(self, data: str) -> None:
        if self._skip_depth:
            return
        text = re.sub(r"\s+", " ", data).strip()
        if len(text) >= 30:
            self.parts.append(text)

    def get_text(self, max_chars: int = 5000) -> str:
        text = "\n".join(self.parts)
        return text[:max_chars].strip()


def fetch_json_url(url: str, timeout: int = 20) -> Any:
    req = Request(url, headers={"User-Agent": "Type-2/1.0"})
    with urlopen(req, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def fetch_text_url(url: str, timeout: int = 20, max_bytes: int = 1_000_000) -> str:
    req = Request(url, headers={"User-Agent": "Type-2/1.0"})
    with urlopen(req, timeout=timeout) as response:
        raw = response.read(max_bytes)
        encoding = response.headers.get_content_charset() or "utf-8"
    return raw.decode(encoding, errors="replace")


def html_to_visible_text(html: str, max_chars: int = 5000) -> str:
    parser = VisibleTextParser()
    parser.feed(html)
    return parser.get_text(max_chars=max_chars)


def is_hacker_news_task(task: str) -> bool:
    normalized = task.lower()
    has_source = (
        "hacker news" in normalized
        or "hackernews" in normalized
        or "hacker-news" in normalized
        or "hn" in normalized
    )
    has_top_request = any(word in task for word in ("最初", "トップ", "一番上", "先頭"))
    wants_japanese = any(word in task for word in ("日本語", "翻訳", "訳"))
    return has_source and has_top_request and wants_japanese


def parse_target_time(task: str, now: datetime | None = None) -> datetime | None:
    now = now or datetime.now()
    jp_match = re.search(r"(午前|午後)?\s*(\d{1,2})\s*時\s*(\d{1,2})?\s*分?", task)
    colon_match = re.search(r"\b(\d{1,2}):(\d{2})\b", task)

    if jp_match:
        period, hour_text, minute_text = jp_match.groups()
        hour = int(hour_text)
        minute = int(minute_text or "0")
        if period == "午後" and hour < 12:
            hour += 12
        elif period == "午前" and hour == 12:
            hour = 0
    elif colon_match:
        hour = int(colon_match.group(1))
        minute = int(colon_match.group(2))
    else:
        return None

    if not 0 <= hour <= 23 or not 0 <= minute <= 59:
        return None

    return now.replace(hour=hour, minute=minute, second=0, microsecond=0)


def wait_until_target_time(
    target_time: datetime | None,
    verbose: bool = True,
    sleep_fn=time.sleep,
) -> str:
    if target_time is None:
        return ""

    now = datetime.now()
    if target_time <= now:
        return "指定時刻はすでに過ぎていたため、即時実行しました。"

    seconds = (target_time - now).total_seconds()
    if verbose:
        print(f"指定時刻 {target_time.strftime('%H:%M')} まで待機します。")

    sleep_fn(seconds)
    return f"指定時刻 {target_time.strftime('%H:%M')} に実行しました。"


def fetch_hacker_news_top_story() -> dict[str, Any]:
    story_ids = fetch_json_url(HN_TOP_STORIES_URL)
    if not isinstance(story_ids, list) or not story_ids:
        raise RuntimeError("Hacker News API returned no top stories")

    item_id = story_ids[0]
    item = fetch_json_url(HN_ITEM_URL.format(item_id=item_id))
    if not isinstance(item, dict):
        raise RuntimeError(f"Hacker News item {item_id} is not an object")

    title = str(item.get("title") or "").strip()
    if not title:
        raise RuntimeError(f"Hacker News item {item_id} has no title")

    item_url = HN_ITEM_PAGE_URL.format(item_id=item_id)
    article_url = str(item.get("url") or item_url).strip()
    body_text = ""
    body_error = ""

    if item.get("text"):
        body_text = re.sub(r"<[^>]+>", " ", str(item["text"]))
        body_text = re.sub(r"\s+", " ", body_text).strip()
    elif article_url:
        try:
            body_text = html_to_visible_text(fetch_text_url(article_url))
        except (OSError, TimeoutError, URLError, UnicodeError) as e:
            body_error = f"記事本文の取得に失敗しました: {e}"

    return {
        "id": item_id,
        "title": title,
        "url": article_url,
        "hn_url": item_url,
        "body": body_text,
        "body_error": body_error,
    }


def ollama_generate(prompt: str, timeout: int = 120) -> str:
    payload = json.dumps(
        {
            "model": settings.ollama_model,
            "prompt": prompt,
            "stream": False,
        }
    ).encode("utf-8")
    req = Request(
        "http://localhost:11434/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(req, timeout=timeout) as response:
        data = json.loads(response.read().decode("utf-8"))
    return str(data.get("response", "")).strip()


def translate_hn_story_to_japanese(story: dict[str, Any]) -> dict[str, str]:
    body = str(story.get("body") or "").strip()
    body_for_prompt = body[:3500] if body else "本文を取得できませんでした。タイトルのみ翻訳してください。"
    prompt = f"""
以下はHacker Newsのトップ記事です。日本語にしてください。
本文が長い場合は、本文全体の内容が伝わるように自然な日本語で要約してください。

必ず次のJSONだけを返してください:
{{"title_jp":"...", "body_jp":"..."}}

原文タイトル:
{story.get("title", "")}

原文本文:
{body_for_prompt}
""".strip()

    raw = ollama_generate(prompt)
    try:
        parsed = extract_json_block(raw)
        if isinstance(parsed, dict):
            return {
                "title_jp": str(parsed.get("title_jp") or "").strip(),
                "body_jp": str(parsed.get("body_jp") or "").strip(),
            }
    except Exception:
        pass

    return {
        "title_jp": str(story.get("title") or "").strip(),
        "body_jp": raw or "翻訳に失敗しました。",
    }


def format_hn_story_output(
    story: dict[str, Any],
    translated: dict[str, str],
    schedule_note: str,
) -> str:
    body_jp = translated.get("body_jp", "").strip()
    if not body_jp:
        body_jp = "本文を取得または翻訳できませんでした。"
    if story.get("body_error"):
        body_jp = f"{body_jp}\n\n取得メモ: {story['body_error']}"

    return f"""
【Hacker News トップニュース】
実行時刻: {datetime.now().isoformat(timespec="seconds")}
{schedule_note}

タイトル:
{translated.get("title_jp") or story.get("title")}

本文:
{body_jp}

原文タイトル:
{story.get("title")}

URL:
{story.get("url")}

HN:
{story.get("hn_url")}
""".strip()


def run_hacker_news_task(task: str, verbose: bool = True) -> str:
    target_time = parse_target_time(task)
    schedule_note = wait_until_target_time(target_time, verbose=verbose)
    try:
        story = fetch_hacker_news_top_story()
    except Exception as e:
        return f"""
【Hacker News トップニュース】
ステータス: FAILURE
実行時刻: {datetime.now().isoformat(timespec="seconds")}
{schedule_note}

本文:
Hacker Newsのトップニュース取得に失敗しました。

理由:
{e}
""".strip()

    try:
        translated = translate_hn_story_to_japanese(story)
    except Exception as e:
        translated = {
            "title_jp": story["title"],
            "body_jp": f"翻訳に失敗しました。原文本文を表示します。\n\n{story.get('body') or '本文を取得できませんでした。'}\n\n理由: {e}",
        }

    return format_hn_story_output(story, translated, schedule_note)
