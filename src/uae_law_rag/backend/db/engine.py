# src/uae_law_rag/backend/db/engine.py

"""
[职责] 数据库引擎与会话工厂：创建 AsyncEngine / AsyncSession，并提供 FastAPI 可注入的 get_session。
[边界] 不包含 ORM Model 定义；不包含业务事务编排（由 service/repo 负责）；不负责 Alembic。
[上游关系] config.py / 环境变量提供数据库连接配置；应用启动时可调用 init_db。
[下游关系] api/deps.py、repo 层依赖 AsyncSession；tests 可复用 sessionmaker。
"""

from __future__ import annotations

import os
from pathlib import Path
from contextlib import asynccontextmanager
from typing import AsyncIterator

from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine

from .base import Base


def _find_repo_root(start: Path) -> Path:
    """
    Best-effort repository root discovery (avoid cwd drift).
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


def _settings_db_url() -> str | None:
    """
    Try reading DB URL from pydantic Settings (.env supported).
    Keep this optional to avoid hard binding engine.py to settings at import time.
    """
    try:
        from uae_law_rag.config import settings as _settings  # type: ignore

        v = str(getattr(_settings, "UAE_LAW_RAG_DATABASE_URL", "") or "").strip()
        return v or None
    except Exception:
        return None


def _default_db_url() -> str:
    """
    Resolve database URL.

    Priority:
        1) settings: UAE_LAW_RAG_DATABASE_URL (loads .env)
        2) env: UAE_LAW_RAG_DATABASE_URL
        3) env: DATABASE_URL
        4) fallback: local sqlite file (repo-root/.Local/uae_law_rag.db)
    """  # docstring: 最小可用配置，不强绑任何 settings 框架
    here = Path(__file__).resolve()
    repo_root = _find_repo_root(here)  # stable repo root, not cwd
    db_path = repo_root / ".Local" / "uae_law_rag.db"
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite+aiosqlite:///{db_path.as_posix()}"


def resolve_db_url(override: str | None = None) -> str:
    if override:
        return override
    s_url = _settings_db_url()
    if s_url:
        return s_url
    env_url = os.getenv("UAE_LAW_RAG_DATABASE_URL", "").strip()
    if env_url:
        return env_url
    env_url2 = os.getenv("DATABASE_URL", "").strip()
    if env_url2:
        return env_url2
    return _default_db_url()


def create_engine(*, url: str | None = None, echo: bool | None = None) -> AsyncEngine:
    """
    Create AsyncEngine.

    NOTE:
      - For SQLite we rely on aiosqlite driver.
      - Foreign keys for SQLite should be enabled at connection time (see fts.py / app startup hook if needed).
    """  # docstring: 生产/测试都可复用；测试可传入临时 sqlite 文件路径
    db_url = resolve_db_url(url)  # docstring: 数据库连接串
    db_echo = echo if echo is not None else (os.getenv("SQL_ECHO", "0") == "1")  # docstring: SQL 打印开关

    return create_async_engine(
        db_url,
        echo=db_echo,
        future=True,
        pool_pre_ping=True,
    )


def create_sessionmaker(engine: AsyncEngine) -> async_sessionmaker[AsyncSession]:
    """Create async sessionmaker."""  # docstring: 统一 expire_on_commit 行为，避免 service 层踩坑
    return async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)


# --- global singletons (app runtime) ---
ENGINE: AsyncEngine = create_engine()  # docstring: 默认全局引擎（生产运行时使用）
SessionLocal: async_sessionmaker[AsyncSession] = create_sessionmaker(ENGINE)  # docstring: 默认会话工厂


@asynccontextmanager
async def session_scope() -> AsyncIterator[AsyncSession]:
    """
    Context manager for DB session.

    Usage:
      async with session_scope() as s:
          ...
    """  # docstring: 脚本/后台任务更方便使用；事务由调用方控制 commit/rollback
    async with SessionLocal() as session:
        yield session


async def init_db(*, engine: AsyncEngine | None = None) -> None:
    """
    Initialize database schema (create_all).

    IMPORTANT:
      - This is for MVP/local use. Production should use Alembic migrations.
      - Must import models to register tables in Base.metadata.
    """  # docstring: 供 main.py startup / scripts 使用
    from . import models  # noqa: F401  # docstring: 强制注册 ORM 表

    eng = engine or ENGINE  # docstring: 允许传入测试 engine
    async with eng.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def drop_db(*, engine: AsyncEngine | None = None) -> None:
    """
    Drop all tables (dangerous).

    Only for local/dev/tests.
    """  # docstring: 方便调试；生产禁止使用
    from . import models  # noqa: F401  # docstring: 确保 metadata 完整

    eng = engine or ENGINE
    async with eng.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


async def get_session() -> AsyncIterator[AsyncSession]:
    """
    FastAPI dependency.

    Example:
      async def endpoint(session: AsyncSession = Depends(get_session)): ...
    """  # docstring: 标准 async generator 依赖注入
    async with SessionLocal() as session:
        yield session
