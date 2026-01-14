# src/uae_law_rag/backend/api/routers/ingest.py

"""
[职责] Ingest Router：暴露导入接口（/ingest），负责 HTTP 入参校验与服务调用。
[边界] 不直接调用 pipeline；不做事务控制；仅进行输入/输出映射。
[上游关系] 前端/外部调用发起 ingest 请求。
[下游关系] ingest_service 执行导入流程并返回结果。
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy.ext.asyncio import AsyncSession

from uae_law_rag.backend.api.deps import (
    get_milvus_repo,
    get_session,
    get_trace_context,
)
from uae_law_rag.backend.api.errors import to_json_response
from uae_law_rag.backend.api.schemas_http._common import (
    DebugEnvelope,
    DocumentId,
    KnowledgeBaseId,
    KnowledgeFileId,
)
from uae_law_rag.backend.api.schemas_http.ingest import (
    IngestRequest,
    IngestResponse,
    IngestStatus,
    IngestTimingMs,
)
from uae_law_rag.backend.db.repo import IngestRepo
from uae_law_rag.backend.kb.repo import MilvusRepo
from uae_law_rag.backend.schemas.audit import TraceContext
from uae_law_rag.backend.utils.constants import (
    DEBUG_KEY,
    REQUEST_ID_KEY,
    TIMING_MS_KEY,
    TRACE_ID_KEY,
)
from uae_law_rag.backend.utils.errors import NotFoundError
from uae_law_rag.backend.services.ingest_service import ingest_file


router = APIRouter(prefix="/ingest", tags=["ingest"])  # docstring: ingest 路由前缀


def _build_debug_envelope(
    *,
    trace_id: str,
    request_id: str,
    timing_ms: Dict[str, Any],
    debug_payload: Dict[str, Any],
) -> DebugEnvelope:
    """
    [职责] 组装 DebugEnvelope（兼容 ingest debug payload）。
    [边界] 仅做字段合并；不校验 payload 语义。
    [上游关系] ingest router 调用。
    [下游关系] IngestResponse.debug 输出结构。
    """
    document_id = debug_payload.get("document_id")  # docstring: 读取 document_id
    records: Dict[str, Any] = {}
    if document_id:
        records["document_id"] = document_id  # docstring: 对齐 debug.records
    envelope: Dict[str, Any] = {
        "trace_id": trace_id,
        "request_id": request_id,
        "records": records,
        "timing_ms": dict(timing_ms),
    }  # docstring: DebugEnvelope 基础字段
    for key, value in debug_payload.items():
        if key in {"trace_id", "request_id", "records", "timing_ms"}:
            continue  # docstring: 避免覆盖基础字段
        envelope[key] = value  # docstring: 合并额外 debug 字段
    return DebugEnvelope.model_validate(envelope)  # docstring: 输出 DebugEnvelope


def _coerce_ingest_status(value: Any) -> IngestStatus:
    """
    [职责] 将输入值规范为 IngestStatus。
    [边界] 非法值回退为 failed。
    [上游关系] ingest router 映射 service 输出时调用。
    [下游关系] IngestResponse.status 使用规范值。
    """
    raw = str(value or "").strip().lower()
    if raw in {"pending", "success", "failed"}:
        return raw  # type: ignore[return-value]  # docstring: 合法状态直接返回
    return "failed"  # type: ignore[return-value]  # docstring: 非法值回退 failed


@router.post("", response_model=IngestResponse)
async def ingest(
    request: IngestRequest,
    debug: bool = Query(False),
    session: AsyncSession = Depends(get_session),
    milvus_repo: MilvusRepo = Depends(get_milvus_repo),
    trace_context: TraceContext = Depends(get_trace_context),
) -> IngestResponse:
    """
    [职责] 触发文件导入并返回导入结果。
    [边界] 不直接调用 pipelines；异常统一转为 ErrorResponse。
    [上游关系] 前端/外部调用请求。
    [下游关系] ingest_service 执行导入并返回 JSON-safe 结果。
    """
    profile = (
        request.ingest_profile.model_dump() if request.ingest_profile is not None else None
    )  # docstring: 归一化 ingest_profile

    try:
        result = await ingest_file(
            session=session,
            kb_id=str(request.kb_id),
            source_uri=str(request.source_uri),
            file_name=str(request.file_name),
            ingest_profile=profile,
            milvus_repo=milvus_repo,
            trace_context=trace_context,
            dry_run=bool(request.dry_run),
            debug=bool(debug),
        )  # docstring: 调用 ingest_service
    except Exception as exc:
        return to_json_response(  # type: ignore[return-value]
            exc,
            trace_id=str(trace_context.trace_id),
            request_id=str(trace_context.request_id),
        )  # docstring: 异常映射为 ErrorResponse

    timing_ms = dict(result.get(TIMING_MS_KEY) or {})  # docstring: 读取 timing_ms
    debug_payload = result.get(DEBUG_KEY) if debug else None  # docstring: 读取 debug payload

    debug_envelope: Optional[DebugEnvelope] = None
    if isinstance(debug_payload, dict):
        trace_id = str(result.get(TRACE_ID_KEY) or trace_context.trace_id)
        request_id = str(result.get(REQUEST_ID_KEY) or trace_context.request_id)
        debug_envelope = _build_debug_envelope(
            trace_id=trace_id,
            request_id=request_id,
            timing_ms=timing_ms,
            debug_payload=debug_payload,
        )  # docstring: 组装 DebugEnvelope

    document_id = None
    if isinstance(debug_payload, dict):
        document_id = debug_payload.get("document_id")  # docstring: 提取 document_id

    file_id_raw = str(result.get("file_id") or "").strip()
    if not file_id_raw:
        raise RuntimeError("ingest_service returned empty file_id")  # docstring: 锁定对外合同，避免 silent bad response

    return IngestResponse(
        kb_id=KnowledgeBaseId(str(result.get("kb_id") or request.kb_id)),
        file_id=KnowledgeFileId(file_id_raw),
        file_name=str(request.file_name),
        document_id=DocumentId(str(document_id)) if document_id else None,
        status=_coerce_ingest_status(result.get("status")),  # docstring: 规范 status
        node_count=int(result.get("node_count") or 0),
        timing_ms=IngestTimingMs.model_validate(timing_ms),  # docstring: 规范 timing_ms
        debug=debug_envelope,
    )  # docstring: 输出 IngestResponse


@router.get("/{file_id}", response_model=IngestResponse)
async def get_ingest_status(
    file_id: str,
    session: AsyncSession = Depends(get_session),
    trace_context: TraceContext = Depends(get_trace_context),
) -> IngestResponse:
    """
    [职责] 查询导入状态（file_id → ingest_status/node_count）。
    [边界] 不触发导入流程；仅查询 DB 记录。
    [上游关系] 前端审计/运营查询。
    [下游关系] 返回 IngestResponse（不含 debug）。
    """
    try:
        ingest_repo = IngestRepo(session)  # docstring: 装配 ingest repo
        file_row = await ingest_repo.get_file(file_id)  # docstring: 查询文件记录
        if file_row is None:
            raise NotFoundError(message="file not found")  # docstring: 文件不存在
    except Exception as exc:
        return to_json_response(  # type: ignore[return-value]
            exc,
            trace_id=str(trace_context.trace_id),
            request_id=str(trace_context.request_id),
        )  # docstring: 异常映射为 ErrorResponse

    return IngestResponse(
        kb_id=KnowledgeBaseId(str(file_row.kb_id)),
        file_id=KnowledgeFileId(str(file_row.id)),
        file_name=str(file_row.file_name),
        document_id=None,
        status=_coerce_ingest_status(file_row.ingest_status),  # docstring: 规范 status
        node_count=int(file_row.node_count or 0),
        timing_ms=IngestTimingMs(),  # docstring: 空 timing_ms
        debug=None,
    )  # docstring: 返回导入状态
