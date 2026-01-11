# src/uae_law_rag/backend/db/repo/generation_repo.py

"""
[职责] GenerationRepo：生成记录（GenerationRecord）的写入与查询。
[边界] 不调用 LLM；不解析结构化输出（由 generation pipeline 负责）；repo 只保存回放所需快照。
[上游关系] generation pipeline 产出 output/citations 后调用 create_* 写入；message_id 是唯一归属。
[下游关系] message service 读取 generation_record 用于白箱展示；evaluator 可读取 output_structured/citations。
"""

from __future__ import annotations

from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..models.generation import GenerationRecordModel


class GenerationRepo:
    """Generation repository (async SQLAlchemy)."""

    def __init__(self, session: AsyncSession):
        self._session = session  # docstring: DB 会话（由 deps 注入）

    async def get_record(self, generation_record_id: str) -> Optional[GenerationRecordModel]:
        """Fetch generation record by id."""  # docstring: 回放与调试
        return await self._session.get(GenerationRecordModel, generation_record_id)

    async def get_record_by_message(self, message_id: str) -> Optional[GenerationRecordModel]:
        """Fetch generation record for a message (1-1)."""  # docstring: 一致化策略下的唯一生成记录
        stmt = select(GenerationRecordModel).where(GenerationRecordModel.message_id == message_id)
        return await self._session.scalar(stmt)

    async def create_record(
        self,
        *,
        message_id: str,
        retrieval_record_id: str,
        prompt_name: str,
        model_provider: str,
        model_name: str,
        output_raw: str,
        messages_snapshot: dict | None = None,
        citations: dict | None = None,
        output_structured: dict | None = None,
        prompt_version: str | None = None,
        status: str = "success",
        error_message: str | None = None,
    ) -> GenerationRecordModel:
        """Create generation record."""  # docstring: 保存 prompt/messages/output/citations 的回放快照
        rec = GenerationRecordModel(
            message_id=message_id,  # docstring: 归属消息（唯一）
            retrieval_record_id=retrieval_record_id,  # docstring: 归属检索证据集合
            prompt_name=prompt_name,  # docstring: prompt 名称
            prompt_version=prompt_version,  # docstring: prompt 版本（可选）
            model_provider=model_provider,  # docstring: provider
            model_name=model_name,  # docstring: 模型名
            messages_snapshot=messages_snapshot or {},  # docstring: 输入 messages 快照
            output_raw=output_raw,  # docstring: 原始输出
            output_structured=output_structured,  # docstring: 结构化输出（可空）
            citations=citations or {},  # docstring: 引用信息
            status=status,  # docstring: success/failed/partial
            error_message=error_message,  # docstring: 失败原因（可空）
        )
        self._session.add(rec)
        await self._session.flush()  # docstring: 获取 rec.id
        return rec
