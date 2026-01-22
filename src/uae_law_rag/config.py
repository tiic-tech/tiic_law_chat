# src/uae_law_rag/config.py
from __future__ import annotations

import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv


def _find_repo_root(start: Path) -> Path:
    """
    Best-effort repository root discovery.
    - Prefer the closest ancestor containing `pyproject.toml`.
    - Fallback to filesystem root if not found.
    """
    cur = start.resolve()
    for _ in range(20):
        if (cur / "pyproject.toml").exists():
            return cur
        if cur.parent == cur:
            break
        cur = cur.parent
    return start.resolve()


PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = _find_repo_root(PACKAGE_ROOT)

# Load .env into process environment early so downstream SDKs can read it.
load_dotenv(str(REPO_ROOT / ".env"), override=False)

# Keep package-local workspace as defaults (dev-friendly),
# but allow full override via environment variables.
DATA_ROOT = REPO_ROOT / ".data"


class Settings(BaseSettings):
    LOCAL_MODELS: bool = True
    DEBUG: bool = True

    # Prefer repo root by default; override via .env if needed.
    PROJECT_ROOT: str = str(REPO_ROOT)

    UAE_LAW_RAG_DATA_RAW_PATH: str = str(DATA_ROOT / "raw")
    UAE_LAW_RAG_DATA_PARSED_PATH: str = str(DATA_ROOT / "parsed")

    UAE_LAW_RAG_DATABASE_URL: str = (
        "sqlite+aiosqlite:////Volumes/Workspace/Projects/RAG/uae_law_rag/.Local/uae_law_rag.db"
    )
    UAE_LAW_RAG_SAMPLE_PDF: str = "/Volumes/Workspace/Projects/RAG/uae_law_rag/.data/raw/demo.pdf"

    OPENAI_API_KEY: str | None = None
    OPENAI_API_BASE: str | None = "https://api.openai.com/v1"

    DASHSCOPE_API_KEY: str | None = None
    DASHSCOPE_BASE_URL: str | None = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    QWEN_CHAT_MODEL: str = "qwen3-max"
    QWEN_MULTI_MODEL: str = "qwen3-vl-plus"
    QWEN_EMBED_MODEL: str = "text-embedding-v4"

    DEEPSEEK_API_KEY: str | None = None
    DEEPSEEK_BASE_URL: str | None = "https://api.deepseek.com/v1"
    DEEPSEEK_CHAT_MODEL: str = "deepseek-chat"
    DEEPSEEK_REASONER_MODEL: str = "deepseek-reasoner"

    OLLAMA_CHAT_MODEL: str = "qwen2.5:1.5b"
    OLLAMA_EMBED_MODEL: str = "qwen3-embedding:4b"
    OLLAMA_REQUEST_TIMEOUT_S: int = int(120)

    DEVICE: str = "auto"

    RERANKER_MODEL_PATH: str = "/Volumes/Workspace/Projects/RAG/tiic/models/bge-reranker-v2-m3"
    RERANKER_DEVICE: str = "mps"
    RERANKER_TOP_N: int = int(10)

    @property
    def project_root(self) -> Path:
        if not self.PROJECT_ROOT:
            raise RuntimeError("PROJECT_ROOT is not set. Please set PROJECT_ROOT in your .env file.")
        return Path(self.PROJECT_ROOT).resolve()

    model_config = SettingsConfigDict(
        env_file=str(REPO_ROOT / ".env"),
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )


settings = Settings()


def _set_env_if_missing(key: str, value: str | None) -> None:
    """
    Keep provider SDKs working with .env-based Settings by exporting to os.environ.
    Do not override explicitly provided environment variables.
    """
    if value is None:
        return
    raw = str(value).strip()
    if not raw:
        return
    if os.getenv(key):
        return
    os.environ[key] = raw


def _bootstrap_provider_env(s: Settings) -> None:
    """
    Export provider-related settings into os.environ for downstream SDKs.
    """
    _set_env_if_missing("OPENAI_API_KEY", s.OPENAI_API_KEY)
    _set_env_if_missing("OPENAI_API_BASE", s.OPENAI_API_BASE)
    _set_env_if_missing("DASHSCOPE_API_KEY", s.DASHSCOPE_API_KEY)
    _set_env_if_missing("DASHSCOPE_BASE_URL", s.DASHSCOPE_BASE_URL)
    _set_env_if_missing("DEEPSEEK_API_KEY", s.DEEPSEEK_API_KEY)
    _set_env_if_missing("DEEPSEEK_BASE_URL", s.DEEPSEEK_BASE_URL)


_bootstrap_provider_env(settings)
