# src/uae_law_rag/backend/db/repo/retrieval_repo.py

"""
[职责] RetrievalRepo：检索记录（RetrievalRecord/Hit）的写入与查询。
[边界] 不执行检索算法；仅持久化 pipeline 参数快照与命中证据列表，保证可回放与可审计。
[上游关系] retrieval pipeline 产出 record + hits 后调用 create_* 写入；message_id 是唯一归属。
[下游关系] generation pipeline 通过 retrieval_record_id 获取证据；evaluator 可读取命中用于计算指标。
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Dict

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..models.retrieval import RetrievalHitModel, RetrievalRecordModel
from ..models.doc import NodeModel


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

    async def resolve_document_id_for_node_ids(self, node_ids: Sequence[str]) -> Optional[str]:
        """
        Resolve document_id by node_ids (first match in input order).
        [边界] 只查 NodeModel.id/document_id，不触发 relationship。
        """
        ids = [str(x).strip() for x in (node_ids or []) if str(x).strip()]
        if not ids:
            return None

        stmt = select(NodeModel.id, NodeModel.document_id).where(NodeModel.id.in_(list(dict.fromkeys(ids))))
        res = await self._session.execute(stmt)
        rows = list(res.all())
        m = {str(r[0]): str(r[1]) for r in rows if r and r[0] and r[1]}
        for nid in ids:
            doc_id = str(m.get(nid) or "").strip()
            if doc_id:
                return doc_id
        return None

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
        # --- batch fetch nodes for fallback snapshots (excerpt/page/offset) ---
        node_ids: List[str] = []
        for h in hits:
            nid = str(h.get("node_id") or "").strip()
            if nid:
                node_ids.append(nid)
        node_map: Dict[str, NodeModel] = {}
        if node_ids:
            q = select(NodeModel).where(NodeModel.id.in_(list(dict.fromkeys(node_ids))))
            res = await self._session.execute(q)
            nodes = list(res.scalars().all())
            node_map = {str(n.id): n for n in nodes}

        def _truncate(s: Optional[str], *, max_chars: int = 800) -> Optional[str]:
            if not s:
                return None
            s2 = str(s).strip()
            if not s2:
                return None
            return s2[:max_chars]

        objs: List[RetrievalHitModel] = []
        for h in hits:
            node_id = str(h["node_id"])
            n = node_map.get(node_id)

            # excerpt fallback: prefer upstream excerpt; else use node text/content
            excerpt = h.get("excerpt")
            if not excerpt and n is not None:
                # try common node text fields; pick the one your NodeModel actually has
                node_text = getattr(n, "text", None) or getattr(n, "content", None)
                excerpt = _truncate(node_text, max_chars=800)

            page = h.get("page")
            if page in (0, "0"):
                page = None
            if page is None and n is not None:
                page = getattr(n, "page", None)
                if page == 0:
                    page = None

            start_offset = h.get("start_offset")
            if n is not None:
                n_doc_s = getattr(n, "start_offset", None)
                n_page_s = getattr(n, "page_start_offset", None)
                # docstring: contract: hit stores page-local offsets
                if start_offset is None:
                    start_offset = n_page_s if n_page_s is not None else n_doc_s
                else:
                    # docstring: if upstream passed doc-global offsets, convert when we can
                    try:
                        if n_page_s is not None and n_doc_s is not None and int(start_offset) == int(n_doc_s):
                            start_offset = n_page_s
                    except Exception:
                        pass

            end_offset = h.get("end_offset")
            if n is not None:
                n_doc_e = getattr(n, "end_offset", None)
                n_page_e = getattr(n, "page_end_offset", None)
                if end_offset is None:
                    end_offset = n_page_e if n_page_e is not None else n_doc_e
                else:
                    try:
                        if n_page_e is not None and n_doc_e is not None and int(end_offset) == int(n_doc_e):
                            end_offset = n_page_e
                    except Exception:
                        pass

            article_id = h.get("article_id")
            if article_id is None and n is not None:
                article_id = getattr(n, "article_id", None)

            section_path = h.get("section_path")
            if section_path is None and n is not None:
                section_path = getattr(n, "section_path", None)

            obj = RetrievalHitModel(
                retrieval_record_id=retrieval_record_id,
                node_id=node_id,
                source=str(h.get("source", "fused")),
                rank=int(h["rank"]),
                score=float(h.get("score", 0.0)),
                score_details=h.get("score_details") or {},
                excerpt=excerpt,
                page=page,
                start_offset=start_offset,
                end_offset=end_offset,
                article_id=article_id,
                section_path=section_path,
            )
            objs.append(obj)

        self._session.add_all(objs)
        await self._session.flush()
        return objs

    async def list_hits(self, *, retrieval_record_id: str) -> List[RetrievalHitModel]:
        """List hits for a retrieval record ordered by rank."""  # docstring: 生成与评估读取证据列表
        stmt = (
            select(RetrievalHitModel)
            .where(RetrievalHitModel.retrieval_record_id == retrieval_record_id)
            .order_by(RetrievalHitModel.rank.asc())
        )
        res = await self._session.scalars(stmt)
        return list(res.all())
