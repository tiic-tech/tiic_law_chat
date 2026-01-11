# src/uae_law_rag/backend/db/repo/message_repo.py

"""
[职责] MessageRepo：消息表的最小数据访问层（创建消息、加载历史、回写 response/status/feedback）。
[边界] 不执行检索/生成（由 pipelines/services 负责）；仅维护消息持久化与历史窗口。
[上游关系] chat service 创建 message(query)；生成结束后回调写回 response/status。
[下游关系] retrieval/generation record 通过 message_id 归属；UI 展示消息历史与反馈。
"""

from __future__ import annotations

from typing import List, Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..models.message import MessageModel


class MessageRepo:
    """Message repository (async SQLAlchemy)."""

    def __init__(self, session: AsyncSession):
        self._session = session  # docstring: DB 会话（由 deps 注入）

    async def get_by_id(self, message_id: str) -> Optional[MessageModel]:
        """Fetch message by id."""  # docstring: 回放/调试定位单条消息
        return await self._session.get(MessageModel, message_id)

    async def create_user_message(
        self,
        *,
        conversation_id: str,
        chat_type: str,
        query: str,
        request_id: str | None = None,
        meta_data: dict | None = None,
    ) -> MessageModel:
        """Create a pending message with user query."""  # docstring: 进入检索/生成前先落库
        msg = MessageModel(
            conversation_id=conversation_id,  # docstring: 归属会话
            chat_type=chat_type,  # docstring: 对话类型
            query=query,  # docstring: 用户问题
            response=None,  # docstring: 生成后回写
            request_id=request_id,  # docstring: 链路 trace id（可空）
            meta_data=meta_data or {},  # docstring: 轻量扩展字段
            status="pending",  # docstring: 初始状态
        )
        self._session.add(msg)
        await self._session.flush()  # docstring: 获取 msg.id
        return msg

    async def list_history(
        self,
        *,
        conversation_id: str,
        limit: int,
        include_pending: bool = False,
    ) -> List[MessageModel]:
        """
        Load message history for a conversation (latest first in DB, caller can reverse).
        """  # docstring: 用于 history_len 窗口加载
        stmt = select(MessageModel).where(MessageModel.conversation_id == conversation_id)
        if not include_pending:
            stmt = stmt.where(MessageModel.status != "pending")
        stmt = stmt.order_by(MessageModel.created_at.desc()).limit(limit)
        res = await self._session.scalars(stmt)
        return list(res.all())

    async def set_response(
        self,
        message_id: str,
        *,
        response: str,
        status: str = "success",
        error_message: str | None = None,
    ) -> bool:
        """Write back LLM response and status."""  # docstring: generation 完成后的回写入口
        msg = await self.get_by_id(message_id)
        if not msg:
            return False
        msg.response = response  # docstring: 写回回答全文
        msg.status = status  # docstring: success/failed/partial
        msg.error_message = error_message  # docstring: 失败原因（可空）
        await self._session.flush()
        return True

    async def set_feedback(
        self,
        message_id: str,
        *,
        score: int,
        reason: str = "",
    ) -> bool:
        """Set user feedback for a message."""  # docstring: UI 评分回写
        msg = await self.get_by_id(message_id)
        if not msg:
            return False
        msg.feedback_score = score  # docstring: 评分（0-100）
        msg.feedback_reason = reason  # docstring: 评分理由
        await self._session.flush()
        return True
