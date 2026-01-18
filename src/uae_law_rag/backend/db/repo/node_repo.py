# src/uae_law_rag/backend/db/repo/node_repo.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from uae_law_rag.backend.db.models.doc import NodeModel, NodeVectorMapModel  # type: ignore


@dataclass(frozen=True)
class NodeWithKb:
    node: NodeModel
    kb_id: Optional[str]


class NodeRepo:
    """
    [职责] NodeRepo：只读查询 node（支持按 kb_id 校验）。
    [边界] 不做全文检索；不做写入；只做回放读取。
    """

    def __init__(self, session: AsyncSession) -> None:
        self._session = session

    async def get_node(self, node_id: str) -> Optional[NodeModel]:
        q = select(NodeModel).where(NodeModel.id == str(node_id))
        res = await self._session.execute(q)
        return res.scalar_one_or_none()

    async def get_node_with_kb(self, node_id: str, kb_id: str) -> Optional[NodeWithKb]:
        """
        [职责] 查询 node 并校验其属于 kb_id（通过 node_vector_map）。
        [边界] 仅校验存在性：要求 map.is_active=True。
        """
        kb_id = str(kb_id or "").strip()
        if not kb_id:
            return None

        q = (
            select(NodeModel, NodeVectorMapModel.kb_id)
            .join(NodeVectorMapModel, NodeVectorMapModel.node_id == NodeModel.id)
            .where(NodeModel.id == str(node_id))
            .where(NodeVectorMapModel.kb_id == kb_id)
            .where(NodeVectorMapModel.is_active.is_(True))
            .limit(1)
        )
        res = await self._session.execute(q)
        row = res.first()
        if row is None:
            return None
        node, kb = row[0], row[1]
        return NodeWithKb(node=node, kb_id=str(kb) if kb else None)
