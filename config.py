from __future__ import annotations

import os
from dataclasses import dataclass
from dotenv import load_dotenv


load_dotenv()


@dataclass(frozen=True)
class Settings:
    ollama_model: str = os.getenv("OLLAMA_MODEL", "gemma4:e4b")
    chroma_path: str = os.getenv("CHROMA_PATH", "./data/chroma")
    success_collection: str = os.getenv("SUCCESS_COLLECTION", "success_memory")
    failure_collection: str = os.getenv("FAILURE_COLLECTION", "failure_memory")
    chroma_embedding: str = os.getenv("CHROMA_EMBEDDING", "default")
    top_k_memories: int = int(os.getenv("TOP_K_MEMORIES", "3"))
    max_improvement_loops: int = int(os.getenv("MAX_IMPROVEMENT_LOOPS", "2"))
    max_agi_cycles: int = int(os.getenv("MAX_AGI_CYCLES", "3"))
    runs_path: str = os.getenv("RUNS_PATH", "./runs")


settings = Settings()
