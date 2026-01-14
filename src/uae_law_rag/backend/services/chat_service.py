# src/uae_law_rag/backend/services/chat_service.py

"""
[职责] chat_service：编排对话链路的服务入口（消息创建 + 检索执行 + Gate 裁决）。
[边界] Phase 1 仅覆盖 retrieval；不触发 generation/evaluator；不处理 HTTP 语义。
[上游关系] api/routers/chat.py 调用 chat(...)；依赖 TraceContext 与 session 注入。
[下游关系] retrieval pipeline 产出 RetrievalRecord/Hits；message/conversation 写回状态以供审计与回放。
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence

from sqlalchemy.ext.asyncio import AsyncSession

from uae_law_rag.backend.db.repo import ConversationRepo, IngestRepo, MessageRepo, RetrievalRepo
from uae_law_rag.backend.kb.repo import MilvusRepo
from uae_law_rag.backend.pipelines.base.context import PipelineContext
from uae_law_rag.backend.pipelines.ingest import embed as embed_mod
from uae_law_rag.backend.pipelines.retrieval.pipeline import run_retrieval_pipeline
from uae_law_rag.backend.schemas.audit import TraceContext
from uae_law_rag.backend.utils.constants import (
    DEBUG_KEY,
    REQUEST_ID_KEY,
    TIMING_MS_KEY,
    TIMING_TOTAL_MS_KEY,
    TRACE_ID_KEY,
)
from uae_law_rag.backend.utils.errors import (
    BadRequestError,
    DomainError,
    ExternalDependencyError,
    InternalError,
    NotFoundError,
    PipelineError,
)
from uae_law_rag.backend.utils.logging_ import get_logger, log_event, truncate_text


MESSAGE_STATUS_PENDING = "pending"
MESSAGE_STATUS_BLOCKED = "blocked"
MESSAGE_STATUS_READY = "ready"
MESSAGE_STATUS_FAILED = "failed"

STATE_MESSAGE_CREATED = "MESSAGE_CREATED"
STATE_RETRIEVAL_DONE = "RETRIEVAL_DONE"
STATE_MESSAGE_BLOCKED = "MESSAGE_BLOCKED"
STATE_MESSAGE_READY = "MESSAGE_READY"

STATE_FLOW = {
    STATE_MESSAGE_CREATED: {STATE_RETRIEVAL_DONE},
    STATE_RETRIEVAL_DONE: {STATE_MESSAGE_BLOCKED, STATE_MESSAGE_READY},
}


@dataclass(frozen=True)
class EmbedDecision:
    """
    [职责] EmbedDecision：记录 embedding 决策结果（provider/model/dim/权限）。
    [边界] 不执行 embedding；仅表达服务层最终决策。
    [上游关系] chat_service 在解析 context/KB 后生成。
    [下游关系] provider_snapshot 写入 RetrievalRecord。
    """

    provider: str
    model: str
    dim: Optional[int]
    source: str
    entitled: bool
    entitlement_reason: Optional[str]


def _normalize_context(context: Optional[Any]) -> Dict[str, Any]:
    """
    [职责] 归一化 context 为 dict（兼容 pydantic/model_dump）。
    [边界] 不做字段校验；仅处理数据形态。
    [上游关系] chat(...) 调用。
    [下游关系] embed/retrieval 配置解析。
    """
    if context is None:
        return {}
    if isinstance(context, Mapping):
        return dict(context)
    if hasattr(context, "model_dump"):
        return dict(context.model_dump())  # docstring: 兼容 pydantic v2
    if hasattr(context, "dict"):
        return dict(context.dict())  # docstring: 兼容 pydantic v1
    try:
        return dict(vars(context))  # docstring: 兜底对象属性
    except Exception:
        return {}


def _resolve_value(
    *,
    key: str,
    context: Mapping[str, Any],
    kb: Mapping[str, Any],
    settings: Mapping[str, Any],
    default: Any,
) -> tuple[Any, str]:
    """
    [职责] 按优先级解析配置值（context > settings > default）。
    [边界] 不校验类型；调用方负责类型转换。
    [上游关系] embed/retrieval 决策调用。
    [下游关系] 返回值与来源标签。
    """
    if key in context and context.get(key) is not None:
        return context.get(key), "context"  # docstring: request/context 覆盖
    if key in kb and kb.get(key) is not None:
        return kb.get(key), "kb"  # docstring: KB 默认
    if key in settings and settings.get(key) is not None:
        return settings.get(key), "conversation"  # docstring: conversation settings 覆盖
    return default, "default"  # docstring: 最终兜底（不得散落硬编码）


def _resolve_int_value(
    *,
    key: str,
    context: Mapping[str, Any],
    kb: Mapping[str, Any],
    settings: Mapping[str, Any],
    default: int,
) -> int:
    """
    [职责] 解析整型配置值（保留 0）。
    [边界] 仅做 int 转换；不做范围校验。
    [上游关系] chat(...) 调用。
    [下游关系] retrieval 配置与 gate 判定。
    """
    value, _source = _resolve_value(
        key=key,
        context=context,
        kb=kb,
        settings=settings,
        default=default,
    )
    if value is None:
        return int(default)  # docstring: None 回退默认值
    return int(value)  # docstring: 转为 int


def _resolve_embed_decision(
    *,
    context: Mapping[str, Any],
    kb: Mapping[str, Any],
    settings: Mapping[str, Any],
    default_embed_provider: str,
    default_embed_model: str,
    default_embed_dim: Optional[int],
) -> EmbedDecision:
    """
    [职责] 解析 embedding 决策（provider/model/dim）。
    [边界] 不执行权限判定；仅决定最终参数与来源。
    [上游关系] chat(...) 调用。
    [下游关系] _check_entitlement 与 embedding 调用。
    """
    provider_raw, provider_src = _resolve_value(
        key="embed_provider",
        context=context,
        kb=kb,
        settings=settings,
        default=default_embed_provider,
    )
    model_raw, model_src = _resolve_value(
        key="embed_model",
        context=context,
        kb=kb,
        settings=settings,
        default=default_embed_model,
    )
    dim_raw, dim_src = _resolve_value(
        key="embed_dim",
        context=context,
        kb=kb,
        settings=settings,
        default=default_embed_dim,
    )
    source = provider_src if provider_src == model_src == dim_src else "mixed"  # docstring: 来源追踪
    provider = str(provider_raw or "").strip()
    model = str(model_raw or "").strip()
    dim = int(dim_raw) if dim_raw is not None else None
    return EmbedDecision(
        provider=provider,
        model=model,
        dim=dim,
        source=source,
        entitled=True,
        entitlement_reason=None,
    )


def _check_entitlement(embed: EmbedDecision) -> EmbedDecision:
    """
    [职责] 执行最小 entitlement 检查（embedding provider allowlist）。
    [边界] 仅做 provider allowlist；不实现计费/配额。
    [上游关系] chat(...) 决策链路调用。
    [下游关系] 控制是否允许向量检索。
    """
    allowlist = {"hash", "local", "mock", "ollama", "openai"}  # docstring: MVP allowlist
    if embed.provider in allowlist:
        return embed  # docstring: 允许使用该 provider
    return EmbedDecision(
        provider=embed.provider,
        model=embed.model,
        dim=embed.dim,
        source=embed.source,
        entitled=False,
        entitlement_reason="embed_provider_not_allowed",
    )


def _build_retrieval_config(
    *,
    context: Mapping[str, Any],
    settings: Mapping[str, Any],
    milvus_collection: Optional[str],
    vector_top_k: int,
    keyword_top_k: int,
) -> Dict[str, Any]:
    """
    [职责] 组装 retrieval pipeline 配置（含 collection 与 top_k）。
    [边界] 不做语义校验；仅归一化与透传。
    [上游关系] chat(...) 调用。
    [下游关系] run_retrieval_pipeline 使用该配置落库。
    """
    cfg: Dict[str, Any] = {
        "keyword_top_k": int(keyword_top_k),  # docstring: keyword top_k
        "vector_top_k": int(vector_top_k),  # docstring: vector top_k
        "milvus_collection": milvus_collection,  # docstring: collection 绑定
        "metric_type": "COSINE",  # docstring: MVP 默认 metric
    }
    fusion_top_k, _ = _resolve_value(
        key="fusion_top_k",
        context=context,
        kb={},
        settings=settings,
        default=None,
    )
    if fusion_top_k is not None:
        cfg["fusion_top_k"] = int(fusion_top_k)  # docstring: fusion top_k 覆盖
    rerank_top_k, _ = _resolve_value(
        key="rerank_top_k",
        context=context,
        kb={},
        settings=settings,
        default=None,
    )
    if rerank_top_k is not None:
        cfg["rerank_top_k"] = int(rerank_top_k)  # docstring: rerank top_k 覆盖
    fusion_strategy, _ = _resolve_value(
        key="fusion_strategy",
        context=context,
        kb={},
        settings=settings,
        default=None,
    )
    if fusion_strategy:
        cfg["fusion_strategy"] = str(fusion_strategy)  # docstring: fusion strategy 覆盖
    rerank_strategy, _ = _resolve_value(
        key="rerank_strategy",
        context=context,
        kb={},
        settings=settings,
        default=None,
    )
    if rerank_strategy:
        cfg["rerank_strategy"] = str(rerank_strategy)  # docstring: rerank strategy 覆盖
    output_fields, _ = _resolve_value(
        key="output_fields",
        context=context,
        kb={},
        settings=settings,
        default=None,
    )
    if output_fields is not None:
        cfg["output_fields"] = list(output_fields)  # docstring: output_fields 覆盖
    metric_type, _ = _resolve_value(
        key="metric_type",
        context=context,
        kb={},
        settings=settings,
        default=None,
    )
    if metric_type:
        cfg["metric_type"] = str(metric_type)  # docstring: metric_type 覆盖
    file_id, _ = _resolve_value(
        key="file_id",
        context=context,
        kb={},
        settings=settings,
        default=None,
    )
    if file_id:
        cfg["file_id"] = str(file_id)  # docstring: file_id 过滤
    document_id, _ = _resolve_value(
        key="document_id",
        context=context,
        kb={},
        settings=settings,
        default=None,
    )
    if document_id:
        cfg["document_id"] = str(document_id)  # docstring: document_id 过滤
    return cfg


def _advance_state(current: str, next_state: str) -> str:
    """
    [职责] 显式状态机推进（校验状态转移合法性）。
    [边界] 仅维护 service 内部状态，不落库。
    [上游关系] chat(...) 编排调用。
    [下游关系] debug 输出与日志。
    """
    allowed = STATE_FLOW.get(current, set())
    if next_state not in allowed:
        raise InternalError(
            message="invalid chat state transition",
            detail={"from": current, "to": next_state},
        )  # docstring: 状态机约束违反
    return next_state


def _build_debug_payload(
    *,
    retrieval_record_id: Optional[str],
    provider_snapshot: Optional[Dict[str, Any]],
    timing_ms: Optional[Dict[str, Any]],
    hits_count: Optional[int],
) -> Dict[str, Any]:
    """
    [职责] 组装 debug 输出（record_id + provider_snapshot + timing_ms）。
    [边界] 不输出全文证据；仅输出审计摘要。
    [上游关系] chat(...) 调用。
    [下游关系] debug 输出用于排障与回放。
    """
    return {
        "retrieval_record_id": retrieval_record_id,
        "provider_snapshot": provider_snapshot or {},
        "timing_ms": timing_ms or {},
        "hits_count": hits_count,
    }  # docstring: debug 字段集合


def _classify_error(exc: Exception, *, stage: Optional[str]) -> DomainError:
    """
    [职责] 将异常映射为 DomainError（pipeline/external）。
    [边界] 仅依据 stage 做最小分类。
    [上游关系] chat(...) 捕获异常后调用。
    [下游关系] routers/errors.py 映射为 HTTP。
    """
    if isinstance(exc, DomainError):
        return exc  # docstring: 领域错误直接透传
    detail = {"stage": stage or "", "error_type": exc.__class__.__name__, "error": str(exc)}
    if stage in {"embed", "vector", "milvus"}:
        return ExternalDependencyError(message="external dependency failed", detail=detail, cause=exc)
    return PipelineError(message="chat pipeline failed", detail=detail, cause=exc)


async def _mark_message_failed(*, session: AsyncSession, message_id: str, reason: str) -> None:
    """
    [职责] 将 message 标记为 failed 并提交。
    [边界] message 不存在时忽略；异常时抛 InternalError。
    [上游关系] chat(...) 失败路径调用。
    [下游关系] message.status/error_message 写回。
    """
    if not message_id:
        return  # docstring: 无 message_id 直接跳过
    try:
        msg_repo = MessageRepo(session)  # docstring: repo 装配
        msg = await msg_repo.get_by_id(message_id)  # docstring: 加载 message
        if msg is None:
            return  # docstring: message 不存在直接返回
        msg.status = MESSAGE_STATUS_FAILED  # docstring: 标记 failed
        msg.error_message = reason  # docstring: 写入失败原因
        await session.flush()  # docstring: 刷新写入
        await session.commit()  # docstring: 提交 failed 状态
    except Exception as exc:
        await session.rollback()  # docstring: 回滚失败写入
        raise InternalError(message="failed to mark message failed", detail={"message_id": message_id}, cause=exc)


async def chat(
    *,
    session: AsyncSession,
    milvus_repo: MilvusRepo,
    query: str,
    conversation_id: Optional[str] = None,
    user_id: Optional[str] = None,
    kb_id: Optional[str] = None,
    chat_type: str = "chat",
    context: Optional[Any] = None,
    trace_context: Optional[TraceContext] = None,
    debug: bool = False,
) -> Dict[str, Any]:
    """
    [职责] chat：执行 Message 创建 + Retrieval pipeline，并返回 blocked/ready 状态。
    [边界] Phase 1 不接 generation/evaluator；不做 HTTP 映射。
    [上游关系] routers/chat.py 调用；上游提供 session/milvus_repo/trace_context。
    [下游关系] 写入 Message/RetrievalRecord/Hits；返回可映射 JSON-safe 结果。
    """
    logger = get_logger("services.chat")  # docstring: 服务日志入口
    start_ts = time.perf_counter()  # docstring: 总耗时起点

    query_text = str(query or "").strip()
    if not query_text:
        raise BadRequestError(message="query is required")  # docstring: query 必填

    ctx = PipelineContext.from_session(session, trace_context=trace_context)  # docstring: 装配 ctx
    context_dict = _normalize_context(context)  # docstring: context 归一化

    log_event(
        logger,
        logging.INFO,
        "chat.start",
        context=ctx,
        fields={"conversation_id": conversation_id, "kb_id": kb_id, "query": truncate_text(query_text)},
    )  # docstring: chat 开始日志

    conv_repo = ConversationRepo(session)  # docstring: conversation repo
    msg_repo = MessageRepo(session)  # docstring: message repo
    ingest_repo = IngestRepo(session)  # docstring: kb repo
    retrieval_repo = RetrievalRepo(session)  # docstring: retrieval repo

    if conversation_id:
        conv = await conv_repo.get_by_id(str(conversation_id))  # docstring: 加载会话
        if conv is None:
            raise NotFoundError(message="conversation not found")  # docstring: 会话必须存在
    else:
        if not user_id:
            raise BadRequestError(message="user_id is required to create conversation")  # docstring: 新会话需 user_id
        conv = await conv_repo.create(
            user_id=str(user_id),
            name=None,
            chat_type=chat_type,
            default_kb_id=str(kb_id) if kb_id else None,
            settings={},
        )  # docstring: 创建新会话
    await session.refresh(conv)  # docstring: 确保会话字段已加载，避免懒加载

    conv_id = str(conv.id)  # docstring: 会话ID快照
    conv_default_kb_id = str(getattr(conv, "default_kb_id", "") or "")  # docstring: 默认 KB 快照
    conv_settings = dict(getattr(conv, "settings", {}) or {})  # docstring: settings 快照

    kb_id_final = str(kb_id or conv_default_kb_id or "").strip()
    if not kb_id_final:
        raise BadRequestError(message="kb_id is required")  # docstring: KB 必填

    kb = await ingest_repo.get_kb(kb_id_final)  # docstring: 加载 KB 配置
    if kb is None:
        raise NotFoundError(message=f"kb not found: {kb_id_final}")  # docstring: KB 必须存在
    await session.refresh(kb)  # docstring: 确保 KB 字段已加载，避免懒加载

    kb_embed_provider = str(getattr(kb, "embed_provider", "ollama") or "ollama")  # docstring: embed provider 快照
    kb_embed_model = str(getattr(kb, "embed_model", "") or "")  # docstring: embed model 快照
    kb_embed_dim = int(getattr(kb, "embed_dim", 0) or 0) or None  # docstring: embed dim 快照
    kb_collection = str(getattr(kb, "milvus_collection", "") or "")  # docstring: collection 快照

    settings = conv_settings  # docstring: 使用会话 settings 快照
    kb_cfg = {
        "embed_provider": kb_embed_provider,
        "embed_model": kb_embed_model,
        "embed_dim": kb_embed_dim,
    }  # docstring: KB 侧配置快照（用于优先级解析）
    embed_decision = _resolve_embed_decision(
        context=context_dict,
        kb=kb_cfg,
        settings=settings,
        default_embed_provider=kb_embed_provider,
        default_embed_model=kb_embed_model,
        default_embed_dim=kb_embed_dim,
    )  # docstring: embed 决策（context > kb > conversation > default）
    embed_decision = _check_entitlement(embed_decision)  # docstring: entitlement 检查

    ctx.with_provider(
        "embed",
        {
            "provider": embed_decision.provider,
            "model": embed_decision.model,
            "dim": embed_decision.dim,
            "source": embed_decision.source,
            "entitled": embed_decision.entitled,
            "entitlement_reason": embed_decision.entitlement_reason,
        },
    )  # docstring: provider_snapshot 写入 ctx

    keyword_top_k_raw = _resolve_int_value(
        key="keyword_top_k",
        context=context_dict,
        kb={},
        settings=settings,
        default=200,
    )  # docstring: keyword top_k 原始值
    vector_top_k_raw = _resolve_int_value(
        key="vector_top_k",
        context=context_dict,
        kb={},
        settings=settings,
        default=50,
    )  # docstring: vector top_k 原始值
    if keyword_top_k_raw < 1:
        keyword_top_k_raw = 1  # docstring: keyword_top_k 最小为 1
    if vector_top_k_raw < 0:
        vector_top_k_raw = 0  # docstring: vector_top_k 负值归零

    keyword_top_k = int(max(1, keyword_top_k_raw))  # docstring: 记录用 keyword_top_k
    # NOTE: RetrievalRecord.vector_top_k schema 要求 >= 1。
    # service 允许请求层 vector_top_k=0 表示“禁用向量”，但落库/record 层必须归一化为 >=1。
    vector_top_k_requested = int(vector_top_k_raw)  # docstring: 请求层语义（可为 0）
    vector_top_k_record = int(max(1, vector_top_k_requested))  # docstring: record 层语义（>=1）

    allow_vector = vector_top_k_raw > 0 and embed_decision.entitled  # docstring: 是否启用向量检索

    retrieval_config = _build_retrieval_config(
        context=context_dict,
        settings=settings,
        milvus_collection=kb_collection,
        vector_top_k=vector_top_k_record,
        keyword_top_k=keyword_top_k,
    )  # docstring: 构造 retrieval 配置
    # 明确记录“是否启用向量”的服务层裁决，同时保持 record/schema 合同（vector_top_k>=1）。
    retrieval_config["vector_enabled"] = bool(allow_vector)  # docstring: 审计字段（pipeline 可忽略）
    retrieval_config["vector_top_k_requested"] = int(vector_top_k_requested)  # docstring: 审计字段（pipeline 可忽略）
    if allow_vector and (not embed_decision.provider or not embed_decision.model):
        raise BadRequestError(
            message="embed_provider/embed_model is required"
        )  # docstring: 向量检索必须有 provider/model

    msg = await msg_repo.create_user_message(
        conversation_id=conv_id,
        chat_type=chat_type,
        query=query_text,
        request_id=str(ctx.request_id),
        meta_data={"context": context_dict},
    )  # docstring: 创建 pending message
    message_id = str(msg.id)  # docstring: message_id 快照
    await session.commit()  # docstring: phase-1 提交 message

    state = STATE_MESSAGE_CREATED  # docstring: 初始状态
    retrieval_record_id: Optional[str] = None
    retrieval_provider_snapshot: Optional[Dict[str, Any]] = None
    retrieval_timing_ms: Optional[Dict[str, Any]] = None
    hits_count: Optional[int] = None
    current_stage: Optional[str] = None

    try:
        state = _advance_state(state, STATE_RETRIEVAL_DONE)  # docstring: 状态推进到 RETRIEVAL_DONE
        current_stage = "embed"
        query_vector: Optional[Sequence[float]] = None
        if allow_vector:
            vectors = await embed_mod.embed_texts(
                texts=[query_text],
                provider=embed_decision.provider,
                model=embed_decision.model,
                dim=embed_decision.dim,
            )  # docstring: 生成 query 向量
            query_vector = vectors[0] if vectors else None  # docstring: 提取首个向量

        current_stage = "retrieval"
        bundle = await run_retrieval_pipeline(
            session=session,
            milvus_repo=milvus_repo,
            retrieval_repo=retrieval_repo,
            message_id=message_id,
            kb_id=kb_id_final,
            query_text=query_text,
            query_vector=list(query_vector) if query_vector is not None else None,
            config=retrieval_config,
            ctx=ctx,
        )  # docstring: 执行 retrieval pipeline

        retrieval_record_id = str(bundle.record.id)  # docstring: retrieval_record_id
        retrieval_provider_snapshot = dict(bundle.record.provider_snapshot)  # docstring: provider_snapshot
        retrieval_timing_ms = dict(bundle.record.timing_ms)  # docstring: timing_ms
        hits_count = len(bundle.hits)  # docstring: 命中数量

        msg_row = await msg_repo.get_by_id(message_id)  # docstring: 回查 message 以写回状态
        if msg_row is None:
            raise InternalError(message="message not found for status update")  # docstring: message 必须存在

        if hits_count <= 0:
            msg_row.status = MESSAGE_STATUS_BLOCKED  # docstring: Gate blocked
            msg_row.error_message = "no_evidence"  # docstring: 阻断原因
            await session.flush()  # docstring: 写回 message 状态
            await session.commit()  # docstring: phase-2 提交 blocked 状态
            state = _advance_state(state, STATE_MESSAGE_BLOCKED)  # docstring: 状态推进到 BLOCKED
            status = MESSAGE_STATUS_BLOCKED
        else:
            msg_row.status = MESSAGE_STATUS_READY  # docstring: Gate ready
            msg_row.error_message = None  # docstring: 清空错误
            await session.flush()  # docstring: 写回 message 状态
            await session.commit()  # docstring: phase-2 提交 ready 状态
            state = _advance_state(state, STATE_MESSAGE_READY)  # docstring: 状态推进到 READY
            status = MESSAGE_STATUS_READY

        total_ms = (time.perf_counter() - start_ts) * 1000.0  # docstring: 总耗时
        timing_ms = {TIMING_TOTAL_MS_KEY: total_ms}  # docstring: 总耗时快照
        debug_payload = (
            _build_debug_payload(
                retrieval_record_id=retrieval_record_id,
                provider_snapshot=retrieval_provider_snapshot,
                timing_ms=retrieval_timing_ms,
                hits_count=hits_count,
            )
            if debug
            else None
        )

        log_event(
            logger,
            logging.INFO,
            "chat.retrieval_done",
            context=ctx,
            fields={"message_id": message_id, "status": status, "hits_count": hits_count},
        )  # docstring: retrieval 完成日志

        response: Dict[str, Any] = {
            "conversation_id": conv_id,
            "message_id": message_id,
            "kb_id": kb_id_final,
            "status": status,
            "answer": "",
            TIMING_MS_KEY: timing_ms,
            TRACE_ID_KEY: str(ctx.trace_id),
            REQUEST_ID_KEY: str(ctx.request_id),
        }  # docstring: 返回 JSON-safe 结果
        if debug_payload is not None:
            response[DEBUG_KEY] = debug_payload  # docstring: debug 仅在显式开启时返回
        return response
    except Exception as exc:
        await session.rollback()  # docstring: 回滚异常事务
        await _mark_message_failed(session=session, message_id=message_id, reason=str(exc))  # docstring: 标记失败
        error = _classify_error(exc, stage=current_stage)  # docstring: 异常归类
        log_event(
            logger,
            logging.ERROR,
            "chat.failed",
            context=ctx,
            fields={"message_id": message_id, "stage": current_stage},
            exc_info=exc,
        )  # docstring: chat 失败日志
        raise error
