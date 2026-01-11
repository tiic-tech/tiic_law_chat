# src/uae_law_rag/backend/db/repo/user_repo.py

"""
[职责] UserRepo：用户表的最小数据访问层（CRUD + 查询）。
[边界] 不实现鉴权/密码校验；不实现业务流程（由 service 层负责）。
[上游关系] admin API / 初始化脚本会调用创建用户；chat/ingest 会校验 user 存在性。
[下游关系] conversation/kb 的归属依赖 user；删除用户将级联清理其会话与 KB（由 DB 外键与模型 cascade 决定）。
"""

from __future__ import annotations

from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..models.user import UserModel


class UserRepo:
    """User repository (async SQLAlchemy)."""

    def __init__(self, session: AsyncSession):
        self._session = session  # docstring: DB 会话（由 deps 注入）

    async def get_by_id(self, user_id: str) -> Optional[UserModel]:
        """Fetch user by id."""  # docstring: 用于请求校验与归属检查
        return await self._session.get(UserModel, user_id)

    async def get_by_username(self, username: str) -> Optional[UserModel]:
        """Fetch user by username."""  # docstring: admin/login 之类的查找入口
        stmt = select(UserModel).where(UserModel.username == username)
        return await self._session.scalar(stmt)

    async def create(
        self,
        *,
        username: str,
        password_hash: str | None = None,
        is_active: bool = True,
    ) -> UserModel:
        """Create a user."""  # docstring: 最小用户创建（MVP 可不启用密码）
        user = UserModel(
            username=username,  # docstring: 用户名（唯一）
            password_hash=password_hash,  # docstring: 密码哈希（可空）
            is_active=is_active,  # docstring: 启用状态
        )
        self._session.add(user)
        await self._session.flush()  # docstring: 获取 user.id（UUID default）
        return user

    async def set_active(self, user_id: str, *, is_active: bool) -> bool:
        """Enable/disable user."""  # docstring: 软禁用用户，不删除历史
        user = await self.get_by_id(user_id)
        if not user:
            return False
        user.is_active = is_active  # docstring: 更新启用状态
        await self._session.flush()
        return True
