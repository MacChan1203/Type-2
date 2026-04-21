from __future__ import annotations

import os
from dataclasses import dataclass
from dotenv import load_dotenv


load_dotenv()


def bool_env(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in ("1", "true", "yes", "y")


def int_env(name: str, default: int, minimum: int = 0) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        return default
    return max(minimum, value)


@dataclass(frozen=True)
class Settings:
    ollama_model: str = os.getenv("OLLAMA_MODEL", "qwen3")
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    chroma_path: str = os.getenv("CHROMA_PATH", "./data/chroma")
    success_collection: str = os.getenv("SUCCESS_COLLECTION", "success_memory")
    failure_collection: str = os.getenv("FAILURE_COLLECTION", "failure_memory")
    chroma_embedding: str = os.getenv("CHROMA_EMBEDDING", "default")
    top_k_memories: int = int_env("TOP_K_MEMORIES", 3, minimum=1)
    max_pipeline_attempts: int = int_env("MAX_PIPELINE_ATTEMPTS", 3, minimum=1)
    max_reflective_cycles: int = int_env("MAX_REFLECTIVE_CYCLES", 3, minimum=1)
    max_tool_wait_seconds: int = int_env("MAX_TOOL_WAIT_SECONDS", 300)
    max_log_text_chars: int = int_env("MAX_LOG_TEXT_CHARS", 12000)
    runs_path: str = os.getenv("RUNS_PATH", "./runs")
    # False にすると LLM Improver 呼び出しをスキップして決定論的な改善ノートを使う
    use_llm_improver: bool = bool_env("USE_LLM_IMPROVER", True)


settings = Settings()
