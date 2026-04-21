from __future__ import annotations

import uuid
import hashlib
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any, Protocol, runtime_checkable

import chromadb
from chromadb.utils import embedding_functions

from config import settings
from evaluation.thresholds import SCORE_SUCCESS_THRESHOLD
from utils.run_logger import redact_sensitive_text

_MAX_MEMORY_REVIEW_ITEMS = 3
_MAX_MEMORY_ITEM_CHARS = 120
_MAX_MEMORY_DOC_CHARS = 800


@runtime_checkable
class MemoryStoreProtocol(Protocol):
    def search_success_memories(self, query: str, n_results: int | None = None) -> list[str]: ...

    def search_failure_memories(self, query: str, n_results: int | None = None) -> list[str]: ...

    def format_memories_for_prompt(self, query: str, include_failures: bool = True) -> str: ...

    def add_memory(self, **kwargs: Any) -> str: ...

    def clear_all_memories(self) -> int: ...


class MemoryStore:
    def __init__(self) -> None:
        self.client = chromadb.PersistentClient(path=settings.chroma_path)
        if settings.chroma_embedding != "default":
            raise ValueError("現在対応している CHROMA_EMBEDDING は default のみです")
        self.embedding_function = embedding_functions.DefaultEmbeddingFunction()

        self.success_collection = self.client.get_or_create_collection(
            name=settings.success_collection,
            embedding_function=self.embedding_function,
        )
        self.failure_collection = self.client.get_or_create_collection(
            name=settings.failure_collection,
            embedding_function=self.embedding_function,
        )

    def add_memory(
        self,
        task: str,
        plan: str,
        output: str,
        review: dict[str, Any],
        improvement_note: str,
        is_reverse_task: bool = False,
        model: str | None = None,
    ) -> str:
        memory_id = str(uuid.uuid4())

        document = self._build_document(
            task=task,
            plan=plan,
            output=output,
            review=review,
            improvement_note=improvement_note,
        )
        metadata = self._build_metadata(
            task=task,
            review=review,
            improvement_note=improvement_note,
            is_reverse_task=is_reverse_task,
            model=model,
        )

        score = int(review.get("score", 0))

        target_collection = (
            self.success_collection if score >= SCORE_SUCCESS_THRESHOLD else self.failure_collection
        )

        target_collection.add(
            ids=[memory_id],
            documents=[document],
            metadatas=[metadata],
        )

        return memory_id

    def search_success_memories(self, query: str, n_results: int | None = None) -> list[str]:
        return self._search_collection(self.success_collection, query, n_results)

    def search_failure_memories(self, query: str, n_results: int | None = None) -> list[str]:
        return self._search_collection(self.failure_collection, query, n_results)

    def _search_collection(
        self,
        collection: Any,
        query: str,
        n_results: int | None,
    ) -> list[str]:
        try:
            requested = n_results or settings.top_k_memories
            if collection.count() == 0:
                return []
            result = collection.query(
                query_texts=[query],
                n_results=requested * 3,
            )
            documents = result.get("documents", [])
            if not documents or not documents[0]:
                return []
            return self._filter_usable_memories([str(doc) for doc in documents[0]])[:requested]
        except Exception as e:
            print(f"メモリ検索に失敗しました: {repr(e)}")
            return []

    def format_memories_for_prompt(
        self,
        query: str,
        include_failures: bool = True,
    ) -> str:
        try:
            # 2つのコレクションへのクエリはI/Oバウンドのため並列化できる
            with ThreadPoolExecutor(max_workers=2) as executor:
                fut_success = executor.submit(self.search_success_memories, query)
                fut_failure = (
                    executor.submit(self.search_failure_memories, query)
                    if include_failures
                    else None
                )
                success_memories = fut_success.result()
                failure_memories = fut_failure.result() if fut_failure else []

            chunks = []

            if success_memories:
                for idx, memory in enumerate(success_memories, start=1):
                    chunks.append(f"[成功記憶 {idx}]\n{memory[:_MAX_MEMORY_DOC_CHARS]}")

            if failure_memories:
                for idx, memory in enumerate(failure_memories, start=1):
                    chunks.append(f"[失敗記憶 {idx}]\n{memory[:_MAX_MEMORY_DOC_CHARS]}")

            return "\n\n".join(chunks) if chunks else ""

        except Exception as e:
            print(f"⚠️ メモリ整形に失敗しました: {repr(e)}")
            return ""

    def clear_all_memories(self) -> int:
        removed = 0
        for collection in (self.success_collection, self.failure_collection):
            result = collection.get()
            ids = list(result.get("ids", []) or [])
            if ids:
                collection.delete(ids=ids)
                removed += len(ids)
        return removed

    def _build_document(
        self,
        task: str,
        plan: str,
        output: str,
        review: dict[str, Any],
        improvement_note: str,
    ) -> str:
        strengths = review.get("strengths", [])
        weaknesses = review.get("weaknesses", [])
        next_action = review.get("next_action", "")

        strengths_text = (
            "\n".join(
                f"- {redact_sensitive_text(str(x))[:_MAX_MEMORY_ITEM_CHARS]}"
                for x in strengths[:_MAX_MEMORY_REVIEW_ITEMS]
            )
            if strengths else "- なし"
        )
        weaknesses_text = (
            "\n".join(
                f"- {redact_sensitive_text(str(x))[:_MAX_MEMORY_ITEM_CHARS]}"
                for x in weaknesses[:_MAX_MEMORY_REVIEW_ITEMS]
            )
            if weaknesses else "- なし"
        )
        output_summary = self._summarize_output(output)

        return f"""
[Task]
{redact_sensitive_text(task)}

[Plan]
{redact_sensitive_text(plan)}

[Output Summary]
{output_summary}

[Review Score]
{review.get("score", 0)}

[Strengths]
{strengths_text}

[Weaknesses]
{weaknesses_text}

[Next Action]
{redact_sensitive_text(str(next_action))}

[Improvement Note]
{redact_sensitive_text(improvement_note)}
""".strip()

    def _summarize_output(self, output: str) -> str:
        text = (output or "").strip()
        if not text:
            return "出力なし"

        lines = [line.rstrip() for line in text.splitlines() if line.strip()]
        head = "\n".join(lines[:12])

        if len(lines) > 12:
            head += f"\n... ({len(lines) - 12} lines omitted)"

        return redact_sensitive_text(head[:1200])

    def _filter_usable_memories(self, memories: list[str]) -> list[str]:
        usable = []
        for memory in memories:
            # Legacy memories stored the full output and can pollute new prompts.
            lowered = memory.lower()
            if "[output]\n" in lowered or "\noutput:\n" in lowered:
                continue
            if "[output summary]" not in lowered:
                continue
            usable.append(memory)
        return usable


    def _build_metadata(
        self,
        task: str,
        review: dict[str, Any],
        improvement_note: str,
        is_reverse_task: bool = False,
        model: str | None = None,
    ) -> dict[str, Any]:
        return {
            "task": redact_sensitive_text(task[:500]),
            "score": int(review.get("score", 0)),
            "next_action": redact_sensitive_text(str(review.get("next_action", ""))[:500]),
            "improvement_note": redact_sensitive_text(improvement_note[:500]),
            "is_reverse_task": bool(is_reverse_task),
            "model": str(model or settings.ollama_model)[:100],
            "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "task_hash": hashlib.sha256(task.encode("utf-8")).hexdigest()[:32],
        }
