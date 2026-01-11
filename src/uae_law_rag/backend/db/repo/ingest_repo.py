# src/uae_law_rag/backend/db/repo/ingest_repo.py

"""
[职责] IngestRepo：导入相关表（KB/文件/文档/节点/映射）的最小写入与查询。
[边界] 不负责 PDF 解析/切分/embedding（由 ingest pipeline 负责）；仅持久化产物。
[上游关系] ingest pipeline 产出 file/document/node/vector_id 后调用本 repo 入库。
[下游关系] retrieval pipeline 依赖 NodeModel（keyword 全量召回）与 NodeVectorMapModel（向量回查/一致性）。
"""

from __future__ import annotations

from typing import List, Optional, Sequence

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from ..models.doc import (
    DocumentModel,
    KnowledgeBaseModel,
    KnowledgeFileModel,
    NodeModel,
    NodeVectorMapModel,
)


class IngestRepo:
    """Ingest repository (async SQLAlchemy)."""

    def __init__(self, session: AsyncSession):
        self._session = session  # docstring: DB 会话（由 deps 注入）

    # --- KB ---
    async def get_kb(self, kb_id: str) -> Optional[KnowledgeBaseModel]:
        """Fetch KB by id."""  # docstring: ingest/chat 需要 KB 配置
        return await self._session.get(KnowledgeBaseModel, kb_id)

    async def create_kb(
        self,
        *,
        user_id: str,
        kb_name: str,
        milvus_collection: str,
        embed_model: str,
        embed_dim: int,
        kb_info: str | None = None,
        embed_provider: str = "ollama",
        rerank_provider: str | None = None,
        rerank_model: str | None = None,
        chunking_config: dict | None = None,
    ) -> KnowledgeBaseModel:
        """Create a KB."""  # docstring: KB 初始化（MVP 可由脚本/接口创建）
        kb = KnowledgeBaseModel(
            user_id=user_id,  # docstring: 归属用户
            kb_name=kb_name,  # docstring: KB 名称
            kb_info=kb_info,  # docstring: 简介
            milvus_collection=milvus_collection,  # docstring: Milvus collection
            embed_provider=embed_provider,  # docstring: embedding provider
            embed_model=embed_model,  # docstring: embedding 模型名
            embed_dim=embed_dim,  # docstring: embedding 维度
            rerank_provider=rerank_provider,  # docstring: rerank provider
            rerank_model=rerank_model,  # docstring: rerank 模型
            chunking_config=chunking_config or {},  # docstring: 切分配置快照
        )
        self._session.add(kb)
        await self._session.flush()
        return kb

    # --- File ---
    async def get_file(self, file_id: str) -> Optional[KnowledgeFileModel]:
        """Fetch file by id."""  # docstring: ingest_gate/服务层常用读接口
        return await self._session.get(KnowledgeFileModel, file_id)

    async def get_file_by_sha256(self, kb_id: str, sha256: str) -> Optional[KnowledgeFileModel]:
        """Find file in KB by content hash."""  # docstring: 幂等导入判定
        stmt = select(KnowledgeFileModel).where(
            KnowledgeFileModel.kb_id == kb_id,
            KnowledgeFileModel.sha256 == sha256,
        )
        return await self._session.scalar(stmt)

    async def create_file(
        self,
        *,
        kb_id: str,
        file_name: str,
        sha256: str,
        source_uri: str | None = None,
        file_ext: str | None = None,
        file_version: int = 1,
        file_mtime: float = 0.0,
        file_size: int = 0,
        pages: int | None = None,
        ingest_profile: dict | None = None,
    ) -> KnowledgeFileModel:
        """Create file record."""  # docstring: 导入开始时创建文件记录
        f = KnowledgeFileModel(
            kb_id=kb_id,  # docstring: 归属 KB
            file_name=file_name,  # docstring: 文件名
            file_ext=file_ext,  # docstring: 扩展名
            source_uri=source_uri,  # docstring: 源 URI
            sha256=sha256,  # docstring: 指纹
            file_version=file_version,  # docstring: 版本号
            file_mtime=file_mtime,  # docstring: mtime
            file_size=file_size,  # docstring: size
            pages=pages,  # docstring: 页数
            ingest_profile=ingest_profile or {},  # docstring: 导入配置快照
            ingest_status="pending",  # docstring: 初始导入状态
        )
        self._session.add(f)
        await self._session.flush()
        return f

    async def mark_file_ingested(
        self,
        file_id: str,
        *,
        status: str,
        node_count: int,
        last_ingested_at,  # datetime
    ) -> bool:
        """Update file ingest status/stats."""  # docstring: ingest 完成后更新统计
        f = await self._session.get(KnowledgeFileModel, file_id)
        if not f:
            return False
        f.ingest_status = status  # docstring: success/failed
        f.node_count = node_count  # docstring: 节点数量统计
        f.last_ingested_at = last_ingested_at  # docstring: 完成时间
        await self._session.flush()
        return True

    # --- Document & Nodes ---
    async def get_document(self, document_id: str) -> Optional[DocumentModel]:
        """Fetch document by id."""  # docstring: ingest_gate/服务层常用读接口
        return await self._session.get(DocumentModel, document_id)

    async def create_document(
        self,
        *,
        kb_id: str,
        file_id: str,
        title: str | None = None,
        source_name: str | None = None,
        meta_data: dict | None = None,
    ) -> DocumentModel:
        """Create document record."""  # docstring: 一般每个 file 对应一个 document
        doc = DocumentModel(
            kb_id=kb_id,  # docstring: 归属 KB
            file_id=file_id,  # docstring: 来源文件
            title=title,  # docstring: 标题
            source_name=source_name,  # docstring: 展示名
            meta_data=meta_data or {},  # docstring: 文档元数据
        )
        self._session.add(doc)
        await self._session.flush()
        return doc

    async def list_nodes_by_document(self, document_id: str) -> List[NodeModel]:
        """List nodes for a document ordered by node_index."""  # docstring: 回放/检索/调试需要稳定顺序
        stmt = select(NodeModel).where(NodeModel.document_id == document_id).order_by(NodeModel.node_index.asc())
        rows = (await self._session.execute(stmt)).scalars().all()
        return list(rows)

    async def list_node_vector_maps_by_file(self, file_id: str) -> List[NodeVectorMapModel]:
        """List node-vector maps for a file."""  # docstring: ingest 完成后核验映射数量与一致性
        stmt = select(NodeVectorMapModel).where(NodeVectorMapModel.file_id == file_id)
        rows = (await self._session.execute(stmt)).scalars().all()
        return list(rows)

    async def bulk_create_nodes(self, *, document_id: str, nodes: Sequence[dict]) -> List[NodeModel]:
        """
        Bulk insert nodes for a document.
        nodes: list of dict with keys: node_index,text,page,start_offset,end_offset,article_id,section_path,meta_data
        """  # docstring: ingest 切分结果批量入库（keyword 全量召回基础）
        objs: List[NodeModel] = []
        for n in nodes:
            obj = NodeModel(
                document_id=document_id,  # docstring: 归属文档
                node_index=int(n["node_index"]),  # docstring: 文档内序号
                text=str(n["text"]),  # docstring: 节点原文
                page=n.get("page"),  # docstring: 页码
                start_offset=n.get("start_offset"),  # docstring: 起始偏移
                end_offset=n.get("end_offset"),  # docstring: 结束偏移
                article_id=n.get("article_id"),  # docstring: 法条标识
                section_path=n.get("section_path"),  # docstring: 结构路径
                meta_data=n.get("meta_data") or {},  # docstring: 扩展元数据
            )
            objs.append(obj)
        self._session.add_all(objs)
        await self._session.flush()  # docstring: 获取每个 node.id（用于向量映射）
        return objs

    async def bulk_create_node_vector_maps(
        self,
        *,
        kb_id: str,
        file_id: str,
        maps: Sequence[dict],
    ) -> List[NodeVectorMapModel]:
        """
        Bulk insert node-vector mappings.
        maps: list of dict with keys: node_id, vector_id, meta_data(optional)
        """  # docstring: 写入 Milvus 后，批量落库映射
        objs: List[NodeVectorMapModel] = []
        for m in maps:
            obj = NodeVectorMapModel(
                kb_id=kb_id,  # docstring: 归属 KB
                file_id=file_id,  # docstring: 归属文件（便于批量删除/重建）
                node_id=str(m["node_id"]),  # docstring: 节点ID
                vector_id=str(m["vector_id"]),  # docstring: Milvus 主键
                meta_data=m.get("meta_data") or {},  # docstring: 映射元数据（collection/partition等）
                is_active=True,  # docstring: 映射有效
            )
            objs.append(obj)
        self._session.add_all(objs)
        await self._session.flush()
        return objs
