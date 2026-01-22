# src/uae_law_rag/backend/db/repo/run_config_repo.py

from __future__ import annotations

from typing import Optional

from sqlalchemy.ext.asyncio import AsyncSession

from ..models.run_config import RunConfigModel


class RunConfigRepo:
    """RunConfig repository (single-row config)."""

    def __init__(self, session: AsyncSession):
        self._session = session  # docstring: DB 会话（由 deps 注入）

    async def get_default(self) -> Optional[RunConfigModel]:
        """Fetch default run_config row."""  # docstring: 读取默认配置
        return await self._session.get(RunConfigModel, "default")

    async def get_default_config(self) -> dict:
        """Fetch default run_config JSON, fallback to empty dict."""  # docstring: 读取配置 JSON
        row = await self.get_default()
        if row is None:
            return {}
        try:
            return dict(row.config or {})
        except Exception:
            return {}

    async def upsert_default(self, *, config: dict) -> RunConfigModel:
        """Upsert default run_config row."""  # docstring: 写入默认配置
        row = await self.get_default()
        if row is None:
            row = RunConfigModel(name="default", config=dict(config or {}))
            self._session.add(row)
            await self._session.flush()
            return row
        row.config = dict(config or {})
        await self._session.flush()
        return row
