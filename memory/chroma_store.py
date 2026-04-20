from __future__ import annotations

import uuid
import hashlib
from datetime import datetime
from typing import Any

import chromadb
from chromadb.utils import embedding_functions

from config import settings


class MemoryStore:
    def __init__(self) -> None:
        self.client = chromadb.PersistentClient(path=settings.chroma_path)
        if settings.chroma_embedding != "default":
            raise ValueError("Only CHROMA_EMBEDDING=default is currently supported")
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
            self.success_collection if score >= 85 else self.failure_collection
        )

        target_collection.add(
            ids=[memory_id],
            documents=[document],
            metadatas=[metadata],
        )

        return memory_id

    def search_success_memories(self, query: str, n_results: int | None = None) -> list[str]:
        try:
            requested = n_results or settings.top_k_memories
            n = max(requested * 3, requested)

            if self.success_collection.count() == 0:
                return []

            result = self.success_collection.query(
                query_texts=[query],
                n_results=n,
            )

            documents = result.get("documents", [])
            if not documents or not documents[0]:
                return []

            return self._filter_usable_memories([str(doc) for doc in documents[0]])[:requested]

        except Exception as e:
            print(f"Memory search failed: {repr(e)}")
            return []

    def search_failure_memories(self, query: str, n_results: int | None = None) -> list[str]:
        try:
            requested = n_results or settings.top_k_memories
            n = max(requested * 3, requested)

            if self.failure_collection.count() == 0:
                return []

            result = self.failure_collection.query(
                query_texts=[query],
                n_results=n,
            )

            documents = result.get("documents", [])
            if not documents or not documents[0]:
                return []

            return self._filter_usable_memories([str(doc) for doc in documents[0]])[:requested]

        except Exception as e:
            print(f"Memory search failed: {repr(e)}")
            return []

    def format_memories_for_prompt(
        self,
        query: str,
        include_failures: bool = True,
    ) -> str:
        try:
            success_memories = self.search_success_memories(query=query)
            failure_memories = (
                self.search_failure_memories(query=query) if include_failures else []
            )

            chunks = []

            if success_memories:
                for idx, memory in enumerate(success_memories, start=1):
                    chunks.append(f"[成功記憶 {idx}]\n{memory}")

            if failure_memories:
                for idx, memory in enumerate(failure_memories, start=1):
                    chunks.append(f"[失敗記憶 {idx}]\n{memory}")

            return "\n\n".join(chunks) if chunks else ""

        except Exception as e:
            print(f"⚠️ Memory formatting failed: {repr(e)}")
            return ""

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

        strengths_text = "\n".join(f"- {x}" for x in strengths) if strengths else "- なし"
        weaknesses_text = "\n".join(f"- {x}" for x in weaknesses) if weaknesses else "- なし"
        output_summary = self._summarize_output(output)

        return f"""
[Task]
{task}

[Plan]
{plan}

[Output Summary]
{output_summary}

[Review Score]
{review.get("score", 0)}

[Strengths]
{strengths_text}

[Weaknesses]
{weaknesses_text}

[Next Action]
{next_action}

[Improvement Note]
{improvement_note}
""".strip()

    def _summarize_output(self, output: str) -> str:
        text = (output or "").strip()
        if not text:
            return "出力なし"

        lines = [line.rstrip() for line in text.splitlines() if line.strip()]
        head = "\n".join(lines[:12])

        if len(lines) > 12:
            head += f"\n... ({len(lines) - 12} lines omitted)"

        return head[:1200]

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
            "task": task[:500],
            "score": int(review.get("score", 0)),
            "next_action": str(review.get("next_action", ""))[:500],
            "improvement_note": improvement_note[:500],
            "is_reverse_task": bool(is_reverse_task),
            "model": str(model or settings.ollama_model)[:100],
            "created_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
            "task_hash": hashlib.sha256(task.encode("utf-8")).hexdigest()[:32],
        }
