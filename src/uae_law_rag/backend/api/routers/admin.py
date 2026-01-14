# src/uae_law_rag/backend/api/routers/admin.py

"""
[职责] Admin Router：提供 KB/File/Document 的最小审计查询接口。
[边界] 不执行业务编排；不触发 pipeline；仅做列表查询与映射。
[上游关系] 运营/审计系统调用 admin 列表接口。
[下游关系] DB models 查询并映射为 schemas_http/admin 输出结构。
"""

from __future__ import annotations

from typing import Any, List, Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from uae_law_rag.backend.api.deps import get_session, get_trace_context
from uae_law_rag.backend.api.errors import to_json_response
from uae_law_rag.backend.api.schemas_http._common import (
    DocumentId,
    KnowledgeBaseId,
    KnowledgeFileId,
    UUIDStr,
)
from uae_law_rag.backend.api.schemas_http.admin import DocumentView, FileView, KBView
from uae_law_rag.backend.api.schemas_http.ingest import IngestStatus
from uae_law_rag.backend.db.models.doc import DocumentModel, KnowledgeBaseModel, KnowledgeFileModel, NodeModel
from uae_law_rag.backend.schemas.audit import TraceContext


router = APIRouter(prefix="/admin", tags=["admin"])  # docstring: admin 路由前缀


def _normalize_iso(dt_value: Any) -> Optional[str]:
    """
    [职责] 归一化 datetime 为 ISO 字符串。
    [边界] 非 datetime 返回 None；不补时区。
    [上游关系] admin router 映射时调用。
    [下游关系] FileView.last_ingested_at 输出。
    """
    if hasattr(dt_value, "isoformat"):
        return dt_value.isoformat()  # docstring: datetime -> ISO
    return None  # docstring: 非 datetime 返回 None


def _coerce_ingest_status(value: Any) -> IngestStatus:
    """
    [职责] 将输入值规范为 IngestStatus。
    [边界] 非法值回退为 failed。
    [上游关系] admin router 映射 file 状态时调用。
    [下游关系] FileView.ingest_status 使用规范值。
    """
    raw = str(value or "").strip().lower()
    if raw in {"pending", "success", "failed"}:
        return raw  # type: ignore[return-value]  # docstring: 合法状态直接返回
    return "failed"  # type: ignore[return-value]  # docstring: 非法值回退 failed


@router.get("/kbs", response_model=List[KBView])
async def list_kbs(
    user_id: Optional[str] = Query(default=None),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    session: AsyncSession = Depends(get_session),
    trace_context: TraceContext = Depends(get_trace_context),
) -> List[KBView]:
    """
    [职责] 列出知识库（可按 user_id 过滤）。
    [边界] 仅做只读查询；不做权限裁决。
    [上游关系] 运营/审计页面调用。
    [下游关系] 返回 KBView 列表。
    """
    try:
        stmt = select(KnowledgeBaseModel)  # docstring: 基础查询
        if user_id:
            stmt = stmt.where(KnowledgeBaseModel.user_id == str(user_id))  # docstring: user_id 过滤
        stmt = stmt.order_by(KnowledgeBaseModel.created_at.desc())  # docstring: 最新优先
        stmt = stmt.limit(int(limit)).offset(int(offset))  # docstring: 分页窗口
        rows = (await session.execute(stmt)).scalars().all()
    except Exception as exc:
        return to_json_response(  # type: ignore[return-value]
            exc,
            trace_id=str(trace_context.trace_id),
            request_id=str(trace_context.request_id),
        )  # docstring: 异常映射为 ErrorResponse

    views: List[KBView] = []
    for kb in rows:
        views.append(
            KBView(
                kb_id=KnowledgeBaseId(str(kb.id)),
                user_id=UUIDStr(str(kb.user_id)) if kb.user_id else None,
                kb_name=str(kb.kb_name),
                kb_info=str(kb.kb_info) if kb.kb_info is not None else None,
                vs_type=str(kb.vs_type) if kb.vs_type is not None else None,
                milvus_collection=str(kb.milvus_collection) if kb.milvus_collection is not None else None,
                milvus_partition=str(kb.milvus_partition) if kb.milvus_partition else None,
                embed_provider=str(kb.embed_provider) if kb.embed_provider is not None else None,
                embed_model=str(kb.embed_model) if kb.embed_model is not None else None,
                embed_dim=int(kb.embed_dim) if kb.embed_dim is not None else None,
                rerank_provider=str(kb.rerank_provider) if kb.rerank_provider else None,
                rerank_model=str(kb.rerank_model) if kb.rerank_model else None,
                chunking_config=dict(kb.chunking_config or {}),
                file_count=int(getattr(kb, "file_count", 0) or 0),
            )
        )  # docstring: 映射 KBView
    return views  # docstring: 返回 KB 列表


@router.get("/files", response_model=List[FileView])
async def list_files(
    kb_id: Optional[str] = Query(default=None),
    user_id: Optional[str] = Query(default=None),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    session: AsyncSession = Depends(get_session),
    trace_context: TraceContext = Depends(get_trace_context),
) -> List[FileView]:
    """
    [职责] 列出知识文件（可按 kb_id/user_id 过滤）。
    [边界] 仅做只读查询；不触发导入流程。
    [上游关系] 运营/审计页面调用。
    [下游关系] 返回 FileView 列表。
    """
    try:
        stmt = select(KnowledgeFileModel)  # docstring: 基础查询
        if user_id:
            stmt = stmt.join(
                KnowledgeBaseModel,
                KnowledgeBaseModel.id == KnowledgeFileModel.kb_id,
            ).where(KnowledgeBaseModel.user_id == str(user_id))  # docstring: user_id 过滤
        if kb_id:
            stmt = stmt.where(KnowledgeFileModel.kb_id == str(kb_id))  # docstring: kb_id 过滤
        stmt = stmt.order_by(KnowledgeFileModel.created_at.desc())  # docstring: 最新优先
        stmt = stmt.limit(int(limit)).offset(int(offset))  # docstring: 分页窗口
        rows = (await session.execute(stmt)).scalars().all()
    except Exception as exc:
        return to_json_response(  # type: ignore[return-value]
            exc,
            trace_id=str(trace_context.trace_id),
            request_id=str(trace_context.request_id),
        )  # docstring: 异常映射为 ErrorResponse

    views: List[FileView] = []
    for f in rows:
        views.append(
            FileView(
                file_id=KnowledgeFileId(str(f.id)),
                kb_id=KnowledgeBaseId(str(f.kb_id)),
                file_name=str(f.file_name),
                file_ext=str(f.file_ext) if f.file_ext else None,
                source_uri=str(f.source_uri) if f.source_uri else None,
                sha256=str(f.sha256),
                file_version=int(f.file_version),
                file_mtime=float(f.file_mtime or 0.0),
                file_size=int(f.file_size or 0),
                pages=int(f.pages) if f.pages is not None else None,
                ingest_profile=dict(f.ingest_profile or {}),
                node_count=int(f.node_count or 0),
                ingest_status=_coerce_ingest_status(f.ingest_status),
                last_ingested_at=_normalize_iso(f.last_ingested_at),
            )
        )  # docstring: 映射 FileView
    return views  # docstring: 返回文件列表


@router.get("/documents", response_model=List[DocumentView])
async def list_documents(
    kb_id: Optional[str] = Query(default=None),
    file_id: Optional[str] = Query(default=None),
    user_id: Optional[str] = Query(default=None),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    session: AsyncSession = Depends(get_session),
    trace_context: TraceContext = Depends(get_trace_context),
) -> List[DocumentView]:
    """
    [职责] 列出文档（可按 kb_id/file_id/user_id 过滤）。
    [边界] 仅做只读查询；不返回节点正文。
    [上游关系] 运营/审计页面调用。
    [下游关系] 返回 DocumentView 列表。
    """
    try:
        stmt = select(DocumentModel, func.count(NodeModel.id).label("node_count")).outerjoin(
            NodeModel, NodeModel.document_id == DocumentModel.id
        )  # docstring: 文档 + 节点统计
        if user_id:
            stmt = stmt.join(
                KnowledgeBaseModel,
                KnowledgeBaseModel.id == DocumentModel.kb_id,
            ).where(KnowledgeBaseModel.user_id == str(user_id))  # docstring: user_id 过滤
        if kb_id:
            stmt = stmt.where(DocumentModel.kb_id == str(kb_id))  # docstring: kb_id 过滤
        if file_id:
            stmt = stmt.where(DocumentModel.file_id == str(file_id))  # docstring: file_id 过滤
        stmt = stmt.group_by(
            DocumentModel.id,
            DocumentModel.kb_id,
            DocumentModel.file_id,
            DocumentModel.title,
            DocumentModel.source_name,
            DocumentModel.meta_data,
            DocumentModel.created_at,
        )  # docstring: group by 全字段（兼容 Postgres 等严格 SQL）
        stmt = stmt.order_by(DocumentModel.created_at.desc())  # docstring: 最新优先
        stmt = stmt.limit(int(limit)).offset(int(offset))  # docstring: 分页窗口
        rows = (await session.execute(stmt)).all()
    except Exception as exc:
        return to_json_response(  # type: ignore[return-value]
            exc,
            trace_id=str(trace_context.trace_id),
            request_id=str(trace_context.request_id),
        )  # docstring: 异常映射为 ErrorResponse

    views: List[DocumentView] = []
    for doc, node_count in rows:
        views.append(
            DocumentView(
                document_id=DocumentId(str(doc.id)),
                kb_id=KnowledgeBaseId(str(doc.kb_id)),
                file_id=KnowledgeFileId(str(doc.file_id)),
                title=str(doc.title) if doc.title else None,
                source_name=str(doc.source_name) if doc.source_name else None,
                meta_data=dict(doc.meta_data or {}),
                node_count=int(node_count or 0),
            )
        )  # docstring: 映射 DocumentView
    return views  # docstring: 返回文档列表
