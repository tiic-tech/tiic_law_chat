# src/uae_law_rag/config.py
from __future__ import annotations

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


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

# Keep package-local workspace as defaults (dev-friendly),
# but allow full override via environment variables.
DATA_ROOT = REPO_ROOT / ".data"


class Settings(BaseSettings):
    LOCAL_MODELS: bool = True
    DEBUG: bool = True

    # Prefer repo root by default; override via .env if needed.
    PROJECT_ROOT: str = str(REPO_ROOT)

    DATA_RAW_PATH: str = str(DATA_ROOT / "raw")
    DATA_PARSED_PATH: str = str(DATA_ROOT / "parsed")

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

    DEVICE: str = "auto"

    @property
    def project_root(self) -> Path:
        if not self.PROJECT_ROOT:
            raise RuntimeError("PROJECT_ROOT is not set. Please set PROJECT_ROOT in your .env file.")
        return Path(self.PROJECT_ROOT).resolve()

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", case_sensitive=True, extra="ignore")


settings = Settings()
