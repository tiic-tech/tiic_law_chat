# src/uae_law_rag/backend/db/repo/retrieval_repo.py

"""
[职责] RetrievalRepo：检索记录（RetrievalRecord/Hit）的写入与查询。
[边界] 不执行检索算法；仅持久化 pipeline 参数快照与命中证据列表，保证可回放与可审计。
[上游关系] retrieval pipeline 产出 record + hits 后调用 create_* 写入；message_id 是唯一归属。
[下游关系] generation pipeline 通过 retrieval_record_id 获取证据；evaluator 可读取命中用于计算指标。
"""

from __future__ import annotations

from typing import List, Optional, Sequence

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..models.retrieval import RetrievalHitModel, RetrievalRecordModel


class RetrievalRepo:
    """Retrieval repository (async SQLAlchemy)."""

    def __init__(self, session: AsyncSession):
        self._session = session  # docstring: DB 会话（由 deps 注入）

    async def get_record(self, retrieval_record_id: str) -> Optional[RetrievalRecordModel]:
        """Fetch retrieval record by id."""  # docstring: 回放与调试
        return await self._session.get(RetrievalRecordModel, retrieval_record_id)

    async def get_record_by_message(self, message_id: str) -> Optional[RetrievalRecordModel]:
        """Fetch retrieval record for a message (1-1)."""  # docstring: 一致化策略下的唯一检索记录
        stmt = select(RetrievalRecordModel).where(RetrievalRecordModel.message_id == message_id)
        return await self._session.scalar(stmt)

    async def create_record(
        self,
        *,
        message_id: str,
        kb_id: str,
        query_text: str,
        keyword_top_k: int,
        vector_top_k: int,
        fusion_top_k: int,
        rerank_top_k: int,
        fusion_strategy: str,
        rerank_strategy: str,
        provider_snapshot: dict | None = None,
        timing_ms: dict | None = None,
    ) -> RetrievalRecordModel:
        """Create retrieval record (without hits)."""  # docstring: 先落 record，再批量写 hits
        rec = RetrievalRecordModel(
            message_id=message_id,  # docstring: 归属消息（唯一）
            kb_id=kb_id,  # docstring: 归属 KB
            query_text=query_text,  # docstring: 检索查询文本
            keyword_top_k=keyword_top_k,  # docstring: keyword top_k
            vector_top_k=vector_top_k,  # docstring: vector top_k
            fusion_top_k=fusion_top_k,  # docstring: fusion top_k
            rerank_top_k=rerank_top_k,  # docstring: rerank top_k
            fusion_strategy=fusion_strategy,  # docstring: 融合策略
            rerank_strategy=rerank_strategy,  # docstring: rerank 策略
            provider_snapshot=provider_snapshot or {},  # docstring: provider/model 快照
            timing_ms=timing_ms or {},  # docstring: 耗时快照
        )
        self._session.add(rec)
        await self._session.flush()  # docstring: 获取 rec.id
        return rec

    async def bulk_create_hits(
        self,
        *,
        retrieval_record_id: str,
        hits: Sequence[dict],
    ) -> List[RetrievalHitModel]:
        """
        Bulk insert hits.
        hits: list of dict with keys: node_id, source, rank, score, score_details, excerpt(optional),
              page(optional), start_offset(optional), end_offset(optional)
        """  # docstring: 记录每个证据命中（白箱证据链）
        objs: List[RetrievalHitModel] = []
        for h in hits:
            obj = RetrievalHitModel(
                retrieval_record_id=retrieval_record_id,  # docstring: 归属检索记录
                node_id=str(h["node_id"]),  # docstring: 证据节点ID
                source=str(h.get("source", "fused")),  # docstring: hit 来源
                rank=int(h["rank"]),  # docstring: 排名
                score=float(h.get("score", 0.0)),  # docstring: 综合分
                score_details=h.get("score_details") or {},  # docstring: 分数细节
                excerpt=h.get("excerpt"),  # docstring: 命中摘要（可选）
                page=h.get("page"),  # docstring: 页码快照（可选）
                start_offset=h.get("start_offset"),  # docstring: 起始偏移快照（可选）
                end_offset=h.get("end_offset"),  # docstring: 结束偏移快照（可选）
            )
            objs.append(obj)
        self._session.add_all(objs)
        await self._session.flush()
        return objs

    async def list_hits(self, retrieval_record_id: str) -> List[RetrievalHitModel]:
        """List hits for a retrieval record ordered by rank."""  # docstring: 生成与评估读取证据列表
        stmt = (
            select(RetrievalHitModel)
            .where(RetrievalHitModel.retrieval_record_id == retrieval_record_id)
            .order_by(RetrievalHitModel.rank.asc())
        )
        res = await self._session.scalars(stmt)
        return list(res.all())
