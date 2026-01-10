from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    LOCAL_MODELS: bool = True
    DEBUG: bool = True
    PROJECT_ROOT: str = "/Volumes/Workspace/Projects/RAG/uae_law_rag"

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

    OLLAMA_CHAT_MODEL: str = "qwen3:4b"
    OLLAMA_EMBED_MODEL: str = "qwen3-embedding:4b"

    DEVICE: str = "auto"

    @property
    def project_root(self) -> Path:
        if not self.PROJECT_ROOT:
            raise RuntimeError("PROJECT_ROOT is not set. Please set PROJECT_ROOT in your .env file.")
        return Path(self.PROJECT_ROOT).resolve()

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", case_sensitive=True, extra="ignore")


settings = Settings()
