# src/uae_law_rag/backend/db/repo/conversation_repo.py

"""
[职责] ConversationRepo：会话表的最小数据访问层（创建/查询/列举）。
[边界] 不加载完整消息内容（由 MessageRepo 负责）；不做对话业务编排。
[上游关系] chat API 创建/选择会话；admin API 列举会话。
[下游关系] message/retrieval/generation 记录归属到会话；删除会话会级联删除消息与记录（依赖外键与 cascade）。
"""

from __future__ import annotations

from typing import List, Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..models.conversation import ConversationModel


class ConversationRepo:
    """Conversation repository (async SQLAlchemy)."""

    def __init__(self, session: AsyncSession):
        self._session = session  # docstring: DB 会话（由 deps 注入）

    async def get_by_id(self, conversation_id: str) -> Optional[ConversationModel]:
        """Fetch conversation by id."""  # docstring: chat 请求校验 conversation 是否存在
        return await self._session.get(ConversationModel, conversation_id)

    async def list_by_user(self, user_id: str, *, limit: int = 50, offset: int = 0) -> List[ConversationModel]:
        """List conversations for a user."""  # docstring: UI 会话列表
        stmt = (
            select(ConversationModel)
            .where(ConversationModel.user_id == user_id)
            .order_by(ConversationModel.created_at.desc())
            .limit(limit)
            .offset(offset)
        )
        res = await self._session.scalars(stmt)
        return list(res.all())

    async def create(
        self,
        *,
        user_id: str,
        name: str | None = None,
        chat_type: str = "chat",
        default_kb_id: str | None = None,
        settings: dict | None = None,
    ) -> ConversationModel:
        """Create a conversation."""  # docstring: 最小会话创建
        conv = ConversationModel(
            user_id=user_id,  # docstring: 归属用户
            name=name,  # docstring: 展示名称
            chat_type=chat_type,  # docstring: chat/agent_chat 等
            default_kb_id=default_kb_id,  # docstring: 默认 KB（可空）
            settings=settings or {},  # docstring: 会话级默认参数快照
        )
        self._session.add(conv)
        await self._session.flush()  # docstring: 获取 conv.id（UUID default）
        return conv

    async def update_settings(self, conversation_id: str, *, settings: dict) -> bool:
        """Replace conversation settings snapshot."""  # docstring: 更新会话级默认策略
        conv = await self.get_by_id(conversation_id)
        if not conv:
            return False
        conv.settings = settings  # docstring: 覆盖 settings（可回放快照）
        await self._session.flush()
        return True

    async def rename(self, conversation_id: str, *, name: str) -> bool:
        """Rename a conversation."""  # docstring: UI 改名
        conv = await self.get_by_id(conversation_id)
        if not conv:
            return False
        conv.name = name  # docstring: 更新会话展示名
        await self._session.flush()
        return True
