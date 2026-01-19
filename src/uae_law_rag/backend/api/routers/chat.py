# src/uae_law_rag/backend/api/routers/chat.py

"""
[职责] Chat Router：暴露聊天接口（/chat），负责 HTTP 入参校验与服务调用。
[边界] 不直接调用 pipeline；不控制事务；仅进行输入/输出映射。
[上游关系] 前端/外部调用发起 chat 请求。
[下游关系] chat_service 执行编排并返回结果。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, cast

from fastapi import APIRouter, Depends, Query, Request
from sqlalchemy.ext.asyncio import AsyncSession

from uae_law_rag.backend.api.deps import (
    get_milvus_repo,
    get_session,
    get_trace_context,
)
from uae_law_rag.backend.api.errors import to_json_response
from uae_law_rag.backend.api.schemas_http._common import (
    ConversationId,
    KnowledgeBaseId,
    MessageId,
    RequestId,
    TraceId,
)
from uae_law_rag.backend.api.schemas_http.chat import (
    ChatDebugEnvelope,
    ChatGateSummary,
    ChatRequest,
    ChatResponse,
    ChatStatus,
    ChatTimingMs,
    CitationView,
    EvaluatorSummary,
)
from uae_law_rag.backend.db.repo import MessageRepo
from uae_law_rag.backend.kb.repo import MilvusRepo
from uae_law_rag.backend.schemas.audit import TraceContext
from uae_law_rag.backend.services.chat_service import chat
from uae_law_rag.backend.utils.constants import (
    DEBUG_KEY,
    EVALUATION_RECORD_ID_KEY,
    GENERATION_RECORD_ID_KEY,
    REQUEST_ID_KEY,
    RETRIEVAL_RECORD_ID_KEY,
    TIMING_MS_KEY,
    TRACE_ID_KEY,
)
from uae_law_rag.backend.pipelines.evaluator.query_plan import build_query_plan  # type: ignore
from uae_law_rag.backend.pipelines.evaluator.keyword_recall import evaluate_keyword_recall  # type: ignore


router = APIRouter(prefix="/chat", tags=["chat"])  # docstring: chat 路由前缀
_USER_HEADER = "x-user-id"  # docstring: user_id header 约定


def _resolve_user_id(request: Request, trace_context: TraceContext) -> Optional[str]:
    """
    [职责] 从 header/trace tags 解析 user_id（可选）。
    [边界] 仅做字符串清理；无值则返回 None。
    [上游关系] chat router 调用。
    [下游关系] chat_service 创建会话时使用。
    """
    header_user_id = str(request.headers.get(_USER_HEADER, "") or "").strip()
    if header_user_id:
        return header_user_id  # docstring: header 优先
    tags = getattr(trace_context, "tags", None)
    if isinstance(tags, dict):
        tag_user_id = str(tags.get("user_id", "") or "").strip()
        if tag_user_id:
            return tag_user_id  # docstring: trace_context.tags 兜底
    return None  # docstring: 无 user_id 时返回 None


def _coerce_chat_status(value: Any) -> ChatStatus:
    """
    [职责] 将输入值规范为 ChatStatus。
    [边界] 非法值回退为 failed。
    [上游关系] chat router 映射 service 输出时调用。
    [下游关系] ChatResponse.status 使用规范值。
    """
    raw = str(value or "").strip().lower()
    if raw in {"blocked", "success", "partial", "failed"}:
        return raw  # type: ignore[return-value]  # docstring: 合法状态直接返回
    return "failed"  # type: ignore[return-value]  # docstring: 非法值回退 failed


def _build_citations(raw: Any) -> List[CitationView]:
    """
    [职责] 将 citations 规范化为 CitationView 列表。
    [边界] 非列表输入回退为空；不进行业务校验。
    [上游关系] chat router 映射 service 输出时调用。
    [下游关系] ChatResponse.citations 输出。
    """
    if not isinstance(raw, list):
        return []  # docstring: 非列表回退空列表
    items: List[CitationView] = []
    for item in raw:
        if isinstance(item, CitationView):
            items.append(item)  # docstring: 兼容已是 CitationView 的输入
            continue
        if isinstance(item, dict):
            v = CitationView.model_validate(item)  # docstring: dict 走模型校验
            loc = v.locator if isinstance(v.locator, dict) else {}
            # docstring: flatten locator -> top-level fields for frontend grouping
            if v.page is None and loc.get("page") is not None:
                try:
                    v.page = int(cast(int, loc.get("page")))
                except Exception:
                    pass
            if v.article_id is None and loc.get("article_id"):
                v.article_id = str(loc.get("article_id"))
            if v.section_path is None and loc.get("section_path"):
                v.section_path = str(loc.get("section_path"))
            items.append(v)
    return items  # docstring: 返回 citations 列表


def _build_debug_envelope(
    *,
    trace_id: str,
    request_id: str,
    timing_ms: Dict[str, Any],
    debug_payload: Dict[str, Any],
    include_gate: bool = True,
) -> ChatDebugEnvelope:
    """
    [职责] 组装 ChatDebugEnvelope（records/gate/扩展字段）。
    [边界] 仅做字段映射；可通过 include_gate 控制 gate 输出。
    [上游关系] chat router 调用。
    [下游关系] ChatResponse.debug 输出结构。
    """
    records: Dict[str, Any] = {}
    for key in (RETRIEVAL_RECORD_ID_KEY, GENERATION_RECORD_ID_KEY, EVALUATION_RECORD_ID_KEY, "document_id"):
        value = debug_payload.get(key)
        if value:
            records[key] = value  # docstring: 收集 record_id

    envelope: Dict[str, Any] = {
        "trace_id": trace_id,
        "request_id": request_id,
        "records": records,
        "timing_ms": dict(timing_ms),
    }  # docstring: DebugEnvelope 基础字段

    if include_gate:
        gate_payload = debug_payload.get("gate")
        if isinstance(gate_payload, dict):
            envelope["gate"] = ChatGateSummary.model_validate(gate_payload)  # docstring: 规范 gate 摘要

    for key, value in debug_payload.items():
        if key in {"trace_id", "request_id", "records", "timing_ms", "gate"}:
            continue  # docstring: 避免覆盖基础字段
        if key in {RETRIEVAL_RECORD_ID_KEY, GENERATION_RECORD_ID_KEY, EVALUATION_RECORD_ID_KEY, "document_id"}:
            continue  # docstring: record_id 已收敛到 records
        envelope[key] = value  # docstring: 合并额外 debug 字段

    return ChatDebugEnvelope.model_validate(envelope)  # docstring: 输出 ChatDebugEnvelope


@router.post("", response_model=ChatResponse)
async def chat_endpoint(
    request: ChatRequest,
    http_request: Request,
    debug: bool = Query(False),
    session: AsyncSession = Depends(get_session),
    milvus_repo: MilvusRepo = Depends(get_milvus_repo),
    trace_context: TraceContext = Depends(get_trace_context),
) -> ChatResponse:
    """
    [职责] 执行 chat 主链路并返回对外响应。
    [边界] 不直接调用 pipelines；异常统一转为 ErrorResponse。
    [上游关系] 前端/外部调用请求。
    [下游关系] chat_service 执行编排并返回 JSON-safe 结果。
    """
    debug_enabled = bool(debug) or bool(request.debug)  # docstring: debug 开关（query/body 二选一）
    context_payload = request.context.model_dump() if request.context is not None else None  # docstring: 归一化 context
    return_records = bool((context_payload or {}).get("return_records"))  # docstring: return_records 开关
    user_id = _resolve_user_id(http_request, trace_context)  # docstring: user_id 解析

    try:
        result = await chat(
            session=session,
            milvus_repo=milvus_repo,
            query=str(request.query),
            conversation_id=str(request.conversation_id) if request.conversation_id else None,
            user_id=user_id,
            kb_id=str(request.kb_id) if request.kb_id else None,
            chat_type="chat",
            context=context_payload,
            trace_context=trace_context,
            debug=debug_enabled,
        )  # docstring: 调用 chat_service
    except Exception as exc:
        return to_json_response(  # type: ignore[return-value]
            exc,
            trace_id=str(trace_context.trace_id),
            request_id=str(trace_context.request_id),
        )  # docstring: 异常映射为 ErrorResponse

    trace_id = str(result.get(TRACE_ID_KEY) or trace_context.trace_id)
    request_id = str(result.get(REQUEST_ID_KEY) or trace_context.request_id)
    timing_ms = dict(result.get(TIMING_MS_KEY) or {})  # docstring: 读取 timing_ms

    debug_payload = (
        result.get(DEBUG_KEY) if (debug_enabled or return_records) else None
    )  # docstring: 读取 debug payload
    debug_envelope: Optional[ChatDebugEnvelope] = None
    if isinstance(debug_payload, dict):
        # docstring: debug=true 时注入 keyword_stats（query_plan -> keyword_recall），用于解释“关键词命中/覆盖”
        if debug_enabled:
            try:
                kb_id_for_stats = str(request.kb_id or "default").strip() or "default"
                plan = build_query_plan(raw_query=str(request.query), kb_id=kb_id_for_stats)

                stats_result = await evaluate_keyword_recall(
                    session=session,
                    kb_id=str(kb_id_for_stats),
                    raw_query=str(request.query),
                    keywords=list(plan.keywords_list),
                    keyword_top_k=None,  # docstring: 使用 evaluator 默认（与服务一致：200）
                    allow_fallback=True,
                    case_sensitive=False,
                    sample_n=0,  # docstring: debug 统计默认不返回 sample，避免 payload 膨胀
                    trace_id=str(getattr(trace_context, "trace_id", "") or ""),
                    request_id=str(getattr(trace_context, "request_id", "") or ""),
                )

                # docstring: 生成轻量统计结构（面向前端折叠面板）
                items = []
                for m in list(stats_result.get("metrics") or []):
                    # m 是 KeywordRecallMetricView（pydantic），可直接 model_dump
                    md = m.model_dump()
                    items.append(
                        {
                            "keyword": md.get("keyword"),
                            "gt_total": md.get("gt_total"),
                            "kw_total": md.get("kw_total"),
                            "overlap": md.get("overlap"),
                            "recall": md.get("recall"),
                            "precision": md.get("precision"),
                            "capped": md.get("capped"),
                        }
                    )

                debug_payload["keyword_stats"] = {
                    "raw_query": str(request.query),
                    "keywords_list": list(plan.keywords_list),
                    "items": items,
                    "timing_ms": dict(stats_result.get("timing_ms") or {}),
                    "meta": {
                        "strategy": str((plan.meta or {}).get("strategy") or "rule_v1"),
                        "kb_id": kb_id_for_stats,
                    },
                }
            except Exception as _exc:
                # docstring: keyword_stats 仅用于 debug 解释层；失败不应影响 chat 主链路
                debug_payload["keyword_stats"] = {
                    "raw_query": str(request.query),
                    "keywords_list": [],
                    "items": [],
                    "timing_ms": {},
                    "meta": {"error": str(_exc)},
                }

        debug_envelope = _build_debug_envelope(
            trace_id=trace_id,
            request_id=request_id,
            timing_ms=timing_ms,
            debug_payload=debug_payload,
            include_gate=bool(debug_enabled),
        )  # docstring: 组装 ChatDebugEnvelope

    conversation_id_raw = str(result.get("conversation_id") or "").strip()
    message_id_raw = str(result.get("message_id") or "").strip()
    kb_id_raw = str(result.get("kb_id") or "").strip() or str(request.kb_id or "").strip()
    if not conversation_id_raw:
        raise RuntimeError("chat_service returned empty conversation_id")  # docstring: 锁定对外合同
    if not message_id_raw:
        raise RuntimeError("chat_service returned empty message_id")  # docstring: 锁定对外合同
    if not kb_id_raw:
        raise RuntimeError("chat_service returned empty kb_id")  # docstring: 锁定对外合同

    citations = _build_citations(result.get("citations"))  # docstring: 规范 citations
    evaluator = EvaluatorSummary.model_validate(result.get("evaluator") or {})  # docstring: 规范 evaluator

    return ChatResponse(
        conversation_id=ConversationId(conversation_id_raw),
        message_id=MessageId(message_id_raw),
        kb_id=KnowledgeBaseId(kb_id_raw),
        status=_coerce_chat_status(result.get("status")),
        answer=str(result.get("answer") or ""),
        citations=citations,
        evaluator=evaluator,
        timing_ms=ChatTimingMs.model_validate(timing_ms),
        trace_id=TraceId(trace_id),
        request_id=RequestId(request_id),
        debug=debug_envelope,
    )  # docstring: 输出 ChatResponse


@router.get("/{conversation_id}/messages", response_model=List[Dict[str, Any]])
async def list_chat_messages(
    conversation_id: str,
    limit: int = Query(50, ge=1, le=200),
    include_pending: bool = Query(False),
    session: AsyncSession = Depends(get_session),
    trace_context: TraceContext = Depends(get_trace_context),
) -> List[Dict[str, Any]]:
    """
    [职责] 拉取会话消息列表（history）。
    [边界] 仅做 DB 读取；不做 pipeline/裁决逻辑。
    [上游关系] 前端回放/历史页面。
    [下游关系] 返回消息摘要列表。
    """
    try:
        repo = MessageRepo(session)  # docstring: 装配 message repo
        messages = await repo.list_history(
            conversation_id=str(conversation_id),
            limit=int(limit),
            include_pending=bool(include_pending),
        )  # docstring: 加载消息历史
    except Exception as exc:
        return to_json_response(  # type: ignore[return-value]
            exc,
            trace_id=str(trace_context.trace_id),
            request_id=str(trace_context.request_id),
        )  # docstring: 异常映射为 ErrorResponse

    items: List[Dict[str, Any]] = []
    for msg in reversed(messages):
        created_at = getattr(msg, "created_at", None)
        items.append(
            {
                "conversation_id": str(msg.conversation_id),
                "message_id": str(msg.id),
                "query": str(msg.query),
                "answer": str(msg.response or ""),
                "status": str(msg.status),
                "created_at": created_at.isoformat() if created_at else None,
            }
        )  # docstring: 输出消息摘要
    return items  # docstring: 返回历史列表
