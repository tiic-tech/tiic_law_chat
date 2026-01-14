# src/uae_law_rag/backend/services/chat_service.py

"""
[职责] chat_service：编排对话链路的服务入口（消息创建 + 检索/生成/评估编排 + Gate 裁决）。
[边界] 不处理 HTTP 语义；不直接调用底层 SDK；仅负责服务层状态机与事务边界。
[上游关系] api/routers/chat.py 调用 chat(...)；依赖 TraceContext 与 session 注入。
[下游关系] retrieval/generation/evaluator pipeline 产出审计记录；message/conversation 写回状态供回放。
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence, cast

from sqlalchemy.ext.asyncio import AsyncSession

from uae_law_rag.backend.db.repo import (
    ConversationRepo,
    EvaluatorRepo,
    GenerationRepo,
    IngestRepo,
    MessageRepo,
    RetrievalRepo,
)
from uae_law_rag.backend.kb.repo import MilvusRepo
from uae_law_rag.backend.pipelines.base.context import PipelineContext
from uae_law_rag.backend.pipelines.ingest import embed as embed_mod
from uae_law_rag.backend.pipelines.evaluator import checks as evaluator_checks
from uae_law_rag.backend.pipelines.evaluator.pipeline import run_evaluator_pipeline
from uae_law_rag.backend.pipelines.generation.pipeline import run_generation_pipeline
from uae_law_rag.backend.pipelines.retrieval.pipeline import run_retrieval_pipeline
from uae_law_rag.backend.schemas.audit import TraceContext
from uae_law_rag.backend.schemas.evaluator import EvaluatorConfig, EvaluationResult
from uae_law_rag.backend.schemas.generation import GenerationBundle
from uae_law_rag.backend.utils.constants import (
    DEFAULT_PROMPT_NAME,
    DEFAULT_PROMPT_VERSION,
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
from uae_law_rag.config import settings


MESSAGE_STATUS_PENDING = "pending"
MESSAGE_STATUS_BLOCKED = "blocked"
MESSAGE_STATUS_SUCCESS = "success"
MESSAGE_STATUS_PARTIAL = "partial"
MESSAGE_STATUS_FAILED = "failed"

STATE_MESSAGE_CREATED = "MESSAGE_CREATED"
STATE_RETRIEVAL_DONE = "RETRIEVAL_DONE"
STATE_GENERATION_DONE = "GENERATION_DONE"
STATE_EVALUATION_DONE = "EVALUATION_DONE"
STATE_MESSAGE_BLOCKED = "MESSAGE_BLOCKED"
STATE_MESSAGE_SUCCESS = "MESSAGE_SUCCESS"
STATE_MESSAGE_PARTIAL = "MESSAGE_PARTIAL"
STATE_MESSAGE_FAILED = "MESSAGE_FAILED"

STATE_FLOW = {
    STATE_MESSAGE_CREATED: {STATE_RETRIEVAL_DONE},
    STATE_RETRIEVAL_DONE: {STATE_MESSAGE_BLOCKED, STATE_GENERATION_DONE},
    STATE_GENERATION_DONE: {STATE_EVALUATION_DONE},
    STATE_EVALUATION_DONE: {STATE_MESSAGE_SUCCESS, STATE_MESSAGE_PARTIAL, STATE_MESSAGE_FAILED},
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


@dataclass(frozen=True)
class LlmDecision:
    """
    [职责] LlmDecision：记录 LLM 决策结果（provider/model/权限/来源）。
    [边界] 不执行 LLM 调用；仅表达服务层最终决议。
    [上游关系] chat_service 在解析 context/KB 后生成。
    [下游关系] generation pipeline 使用该决议构建 provider_snapshot。
    """

    provider: str
    model: str
    source: str
    entitled: bool
    entitlement_reason: Optional[str]


@dataclass(frozen=True)
class RetrievalGateDecision:
    """
    [职责] RetrievalGateDecision：封装 retrieval gate 裁决结果（是否通过 + 原因）。
    [边界] 仅表达 gate 结果，不负责 DB 写回。
    [上游关系] chat_service 在 retrieval pipeline 完成后调用。
    [下游关系] chat_service 根据裁决决定 blocked/继续。
    """

    passed: bool
    reasons: Sequence[str]


@dataclass(frozen=True)
class GenerationGateDecision:
    """
    [职责] GenerationGateDecision：封装 generation gate 状态与原因。
    [边界] 仅记录生成状态，不裁决 message.status。
    [上游关系] chat_service 在 generation pipeline 完成后调用。
    [下游关系] chat_service debug 输出与 evaluator 输入。
    """

    status: str
    reasons: Sequence[str]


@dataclass(frozen=True)
class EvaluatorGateDecision:
    """
    [职责] EvaluatorGateDecision：封装 evaluator 裁决状态与原因。
    [边界] 仅记录 evaluator 判定，不负责 message.status 映射。
    [上游关系] chat_service 在 evaluator pipeline 完成后调用。
    [下游关系] message.status 映射与 debug 输出。
    """

    status: str
    reasons: Sequence[str]


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
    [职责] 按优先级解析配置值（context > kb > settings > default）。
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


def _resolve_mapping_value(
    *,
    keys: Sequence[str],
    context: Mapping[str, Any],
    kb: Mapping[str, Any],
    settings: Mapping[str, Any],
) -> tuple[Dict[str, Any], str]:
    """
    [职责] 解析 Mapping 配置值（context > kb > settings）。
    [边界] 非 Mapping 回退空 dict；不做字段校验。
    [上游关系] generation/evaluator 配置构造调用。
    [下游关系] run_generation_pipeline/run_evaluator_pipeline 入参。
    """
    for key in keys:
        value = context.get(key) if key in context else None
        if isinstance(value, Mapping):
            return dict(value), "context"  # docstring: context 覆盖
    for key in keys:
        value = kb.get(key) if key in kb else None
        if isinstance(value, Mapping):
            return dict(value), "kb"  # docstring: KB 覆盖
    for key in keys:
        value = settings.get(key) if key in settings else None
        if isinstance(value, Mapping):
            return dict(value), "conversation"  # docstring: conversation settings 覆盖
    return {}, "default"  # docstring: 缺省回退空 dict


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


def _resolve_provider_mode(provider: str) -> str:
    """
    [职责] 根据 provider 推断运行模式（local/remote）。
    [边界] 仅基于 settings.LOCAL_MODELS 与 provider 名称判断。
    [上游关系] chat_service 生成 provider_snapshot。
    [下游关系] debug/provider_snapshot 审计输出。
    """
    if settings.LOCAL_MODELS:
        local_providers = {"ollama", "mock", "local", "hash"}  # docstring: 本地 provider 集合
        if str(provider or "").strip().lower() in local_providers:
            return "local"  # docstring: local 模式
    return "remote"  # docstring: 兜底 remote


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


def _resolve_llm_decision(
    *,
    context: Mapping[str, Any],
    kb: Mapping[str, Any],
    settings: Mapping[str, Any],
    default_provider: str,
    default_model: str,
) -> LlmDecision:
    """
    [职责] 解析 LLM 决策（provider/model）。
    [边界] 不执行权限判定；仅决定最终参数与来源。
    [上游关系] chat(...) 调用。
    [下游关系] _check_llm_entitlement 与 generation 调用。
    """
    provider_raw, provider_src = _resolve_value(
        key="model_provider",
        context=context,
        kb=kb,
        settings=settings,
        default=default_provider,
    )
    if not provider_raw:
        provider_raw, provider_src = _resolve_value(
            key="chat_provider",
            context=context,
            kb=kb,
            settings=settings,
            default=default_provider,
        )  # docstring: 兼容 chat_provider
    model_raw, model_src = _resolve_value(
        key="model_name",
        context=context,
        kb=kb,
        settings=settings,
        default=default_model,
    )
    if not model_raw:
        model_raw, model_src = _resolve_value(
            key="chat_model",
            context=context,
            kb=kb,
            settings=settings,
            default=default_model,
        )  # docstring: 兼容 chat_model
    source = provider_src if provider_src == model_src else "mixed"  # docstring: 来源追踪
    provider = str(provider_raw or "").strip()
    model = str(model_raw or "").strip()
    return LlmDecision(
        provider=provider,
        model=model,
        source=source,
        entitled=True,
        entitlement_reason=None,
    )


def _check_llm_entitlement(llm: LlmDecision) -> LlmDecision:
    """
    [职责] 执行最小 LLM entitlement 检查（provider allowlist）。
    [边界] 仅做 provider allowlist；不实现计费/配额。
    [上游关系] chat(...) 决策链路调用。
    [下游关系] 控制是否允许调用 generation pipeline。
    """
    allowlist = {
        "ollama",
        "openai",
        "dashscope",
        "qwen",
        "huggingface",
        "hf",
        "deepseek",
        "openai_like",
        "openai-like",
        "mock",
        "local",
        "hash",
    }  # docstring: MVP allowlist
    if llm.provider in allowlist:
        return llm  # docstring: 允许使用该 provider
    return LlmDecision(
        provider=llm.provider,
        model=llm.model,
        source=llm.source,
        entitled=False,
        entitlement_reason="llm_provider_not_allowed",
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


def _build_generation_config(
    *,
    context: Mapping[str, Any],
    kb: Mapping[str, Any],
    settings: Mapping[str, Any],
    llm: LlmDecision,
) -> Dict[str, Any]:
    """
    [职责] 组装 generation pipeline 配置（prompt + model + postprocess）。
    [边界] 不校验 prompt/LLM 语义；仅归一化与透传。
    [上游关系] chat(...) 调用。
    [下游关系] run_generation_pipeline 使用该配置落库。
    """
    prompt_name, _ = _resolve_value(
        key="prompt_name",
        context=context,
        kb=kb,
        settings=settings,
        default=DEFAULT_PROMPT_NAME,
    )  # docstring: prompt_name 解析
    prompt_version, _ = _resolve_value(
        key="prompt_version",
        context=context,
        kb=kb,
        settings=settings,
        default=DEFAULT_PROMPT_VERSION,
    )  # docstring: prompt_version 解析
    generation_config, _ = _resolve_mapping_value(
        keys=("generation_config", "llm_config", "generation"),
        context=context,
        kb=kb,
        settings=settings,
    )  # docstring: generation_config 解析
    temperature, _ = _resolve_value(
        key="temperature",
        context=context,
        kb=kb,
        settings=settings,
        default=None,
    )  # docstring: temperature 解析
    if temperature is not None and "temperature" not in generation_config:
        generation_config["temperature"] = temperature  # docstring: 温度覆盖
    prompt_config, _ = _resolve_mapping_value(
        keys=("prompt_config", "prompt"),
        context=context,
        kb=kb,
        settings=settings,
    )  # docstring: prompt_config 解析
    postprocess_config, _ = _resolve_mapping_value(
        keys=("postprocess_config", "postprocess"),
        context=context,
        kb=kb,
        settings=settings,
    )  # docstring: postprocess_config 解析

    cfg: Dict[str, Any] = {
        "model_provider": llm.provider,
        "model_name": llm.model,
        "prompt_name": str(prompt_name or DEFAULT_PROMPT_NAME),
        "prompt_version": str(prompt_version or DEFAULT_PROMPT_VERSION),
        "generation_config": generation_config,
    }  # docstring: generation 配置快照
    if prompt_config:
        cfg["prompt_config"] = dict(prompt_config)  # docstring: prompt_config 覆盖
    if postprocess_config:
        cfg["postprocess_config"] = dict(postprocess_config)  # docstring: postprocess_config 覆盖

    no_evidence_use_llm, _ = _resolve_value(
        key="no_evidence_use_llm",
        context=context,
        kb=kb,
        settings=settings,
        default=None,
    )
    if no_evidence_use_llm is not None:
        cfg["no_evidence_use_llm"] = bool(no_evidence_use_llm)  # docstring: no_evidence_use_llm 覆盖
    return cfg


def _build_evaluator_config(
    *,
    context: Mapping[str, Any],
    kb: Mapping[str, Any],
    settings: Mapping[str, Any],
) -> EvaluatorConfig:
    """
    [职责] 组装 evaluator pipeline 配置（EvaluatorConfig）。
    [边界] 配置异常回退默认；不做策略校验。
    [上游关系] chat(...) 调用。
    [下游关系] run_evaluator_pipeline 使用该配置落库。
    """
    raw, _ = _resolve_mapping_value(
        keys=("evaluator_config", "evaluator"),
        context=context,
        kb=kb,
        settings=settings,
    )  # docstring: evaluator_config 解析
    try:
        return EvaluatorConfig(**raw) if raw else EvaluatorConfig()  # docstring: 构造 EvaluatorConfig
    except Exception:
        return EvaluatorConfig()  # docstring: 配置异常回退默认


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


def _evaluate_retrieval_gate(*, hits_count: int) -> RetrievalGateDecision:
    """
    [职责] 执行最小 retrieval gate 裁决（必须有命中）。
    [边界] 仅基于 hits_count；不做 coverage/quality 判断。
    [上游关系] chat(...) 在 retrieval pipeline 后调用。
    [下游关系] message.status blocked/continue 决策。
    """
    if hits_count <= 0:
        return RetrievalGateDecision(passed=False, reasons=("no_evidence",))  # docstring: 无证据阻断
    return RetrievalGateDecision(passed=True, reasons=())  # docstring: 命中通过


def _evaluate_generation_gate(bundle: GenerationBundle) -> GenerationGateDecision:
    """
    [职责] 执行最小 generation gate 记录（状态 + 原因）。
    [边界] 不裁决 message.status；仅记录 generation 状态。
    [上游关系] chat(...) 在 generation pipeline 后调用。
    [下游关系] debug 输出与 evaluator 裁决解释。
    """
    status = str(getattr(bundle.record, "status", "") or "failed").strip().lower()
    reasons = []
    error_message = str(getattr(bundle.record, "error_message", "") or "").strip()
    if error_message:
        reasons.append(error_message)  # docstring: 透传 generation 错误
    if status and status != "success":
        reasons.append(f"generation_status={status}")  # docstring: 记录状态
    return GenerationGateDecision(status=status or "failed", reasons=tuple(reasons))


def _evaluate_evaluator_gate(result: EvaluationResult) -> EvaluatorGateDecision:
    """
    [职责] 解析 evaluator 结果并生成 gate 决策（status + reasons）。
    [边界] 仅基于 checks/warnings；不改变 evaluator status。
    [上游关系] chat(...) 在 evaluator pipeline 后调用。
    [下游关系] message.status 映射与 debug 输出。
    """
    status = str(result.status or "skipped")  # docstring: evaluator status
    reasons: list[str] = []  # docstring: 原因列表
    for check in list(result.checks or []):
        check_status = str(getattr(check, "status", "") or "")
        if check_status in {"fail", "warn"}:
            # NOTE: EvaluatorCheck MVP contract uses `reason` as the primary human-readable field.
            # Keep backward-compat with `message` if legacy objects exist.
            reason_text = str(
                getattr(check, "reason", "") or getattr(check, "message", "") or getattr(check, "name", "") or ""
            ).strip()
            reasons.append(reason_text or "check_failed")  # docstring: 兜底原因
    return EvaluatorGateDecision(status=status, reasons=tuple([r for r in reasons if r]))


def _map_evaluation_status(status: str) -> str:
    """
    [职责] 将 EvaluationStatus 映射为 message.status（最终裁决）。
    [边界] 仅处理 pass/partial/fail/skipped；未知回退 failed。
    [上游关系] chat(...) evaluator 完成后调用。
    [下游关系] message.status 写回与 response.status。
    """
    if status == "pass":
        return MESSAGE_STATUS_SUCCESS  # docstring: evaluator pass -> success
    if status == "partial":
        return MESSAGE_STATUS_PARTIAL  # docstring: evaluator partial -> partial
    return MESSAGE_STATUS_FAILED  # docstring: fail/skipped -> failed


def _build_debug_payload(
    *,
    retrieval_record_id: Optional[str],
    generation_record_id: Optional[str],
    evaluation_record_id: Optional[str],
    retrieval_gate: Optional[RetrievalGateDecision],
    generation_gate: Optional[GenerationGateDecision],
    evaluator_gate: Optional[EvaluatorGateDecision],
    provider_snapshot: Optional[Dict[str, Any]],
    timing_ms: Optional[Dict[str, Any]],
    hits_count: Optional[int],
) -> Dict[str, Any]:
    """
    [职责] 组装 debug 输出（record_id/gate/provider_snapshot/timing_ms）。
    [边界] 不输出全文证据；仅输出审计摘要。
    [上游关系] chat(...) 调用。
    [下游关系] debug 输出用于排障与回放。
    """
    gate_payload: Dict[str, Any] = {}  # docstring: gate 汇总
    if retrieval_gate is not None:
        gate_payload["retrieval"] = {
            "passed": bool(retrieval_gate.passed),
            "reasons": list(retrieval_gate.reasons),
        }  # docstring: retrieval gate 摘要
    if generation_gate is not None:
        gate_payload["generation"] = {
            "status": str(generation_gate.status),
            "reasons": list(generation_gate.reasons),
        }  # docstring: generation gate 摘要
    if evaluator_gate is not None:
        gate_payload["evaluator"] = {
            "status": str(evaluator_gate.status),
            "reasons": list(evaluator_gate.reasons),
        }  # docstring: evaluator gate 摘要
    return {
        "retrieval_record_id": retrieval_record_id,
        "generation_record_id": generation_record_id,
        "evaluation_record_id": evaluation_record_id,
        "gate": gate_payload,
        "provider_snapshot": provider_snapshot or {},
        "timing_ms": timing_ms or {},
        "hits_count": hits_count,
    }  # docstring: debug 字段集合


def _merge_provider_snapshot(*snapshots: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """
    [职责] 合并 provider_snapshot（后者覆盖同名 key）。
    [边界] 非 Mapping 忽略；不做深合并。
    [上游关系] chat(...) 构建 debug 输出时调用。
    [下游关系] debug.provider_snapshot。
    """
    merged: Dict[str, Any] = {}  # docstring: 合并容器
    for snapshot in snapshots:
        if isinstance(snapshot, Mapping):
            merged.update(snapshot)  # docstring: 覆盖更新
    return merged


def _build_evaluator_summary(
    *,
    evaluator: Optional[EvaluationResult],
    fallback_status: str,
    fallback_rule_version: str,
    fallback_reasons: Sequence[str],
) -> Dict[str, Any]:
    """
    [职责] 组装 evaluator 摘要（status/rule_version/warnings）。
    [边界] evaluator 为空时回退到 fallback；不输出完整 checks。
    [上游关系] chat(...) 调用。
    [下游关系] response.evaluator 供前端展示。
    """
    if evaluator is None:
        return {
            "status": fallback_status,
            "rule_version": fallback_rule_version,
            "warnings": list(fallback_reasons),
        }  # docstring: evaluator 缺失兜底
    warnings = []
    for check in list(evaluator.checks or []):
        if str(getattr(check, "status", "")) in {"warn", "fail"}:
            msg = str(
                getattr(check, "reason", "") or getattr(check, "message", "") or getattr(check, "name", "") or ""
            ).strip()
            if msg:
                warnings.append(msg)  # docstring: 收集 warn/fail 消息
    return {
        "status": str(evaluator.status),
        "rule_version": str(evaluator.config.rule_version),
        "warnings": [w for w in warnings if w],
    }  # docstring: evaluator 摘要


def _extract_generation_citations(bundle: Optional[GenerationBundle]) -> list[Dict[str, Any]]:
    """
    [职责] 从 GenerationBundle 提取 citations 列表（JSON-safe）。
    [边界] bundle 缺失或格式异常时返回空列表。
    [上游关系] chat(...) generation 结束后调用。
    [下游关系] response.citations。
    """
    if bundle is None:
        return []  # docstring: 无 bundle 直接回退
    payload = getattr(bundle.record, "citations", None)  # docstring: CitationsPayload
    if payload is None:
        return []  # docstring: 缺失 citations 回退空
    if hasattr(payload, "model_dump"):
        items = payload.model_dump().get("items")  # type: ignore[attr-defined]
    elif hasattr(payload, "dict"):
        items = payload.dict().get("items")  # type: ignore[call-arg]
    elif isinstance(payload, Mapping):
        items = payload.get("items")
    else:
        items = None
    if isinstance(items, list):
        return list(items)  # docstring: citations items
    return []  # docstring: 非列表回退空


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
    # generation / evaluator can also be external if they call LLMs or remote services
    if stage in {"embed", "vector", "milvus", "generation", "llm"}:
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
    [职责] chat：执行 Message 创建 + Retrieval/Generation/Evaluator pipeline，并返回最终裁决。
    [边界] 不做 HTTP 映射；不触碰底层 SDK；仅编排与状态机。
    [上游关系] routers/chat.py 调用；上游提供 session/milvus_repo/trace_context。
    [下游关系] 写入 Message/Retrieval/Generation/Evaluation 记录；返回可映射 JSON-safe 结果。
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
    generation_repo = GenerationRepo(session)  # docstring: generation repo
    evaluator_repo = EvaluatorRepo(session)  # docstring: evaluator repo

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
    kb_chat_provider = str(getattr(kb, "chat_provider", "") or "")  # docstring: chat provider 快照
    kb_chat_model = str(getattr(kb, "chat_model", "") or "")  # docstring: chat model 快照

    conversation_settings = conv_settings  # docstring: 使用会话 settings 快照
    kb_cfg = {
        "embed_provider": kb_embed_provider,
        "embed_model": kb_embed_model,
        "embed_dim": kb_embed_dim,
        "chat_provider": kb_chat_provider,
        "chat_model": kb_chat_model,
    }  # docstring: KB 侧配置快照（用于优先级解析）
    embed_decision = _resolve_embed_decision(
        context=context_dict,
        kb=kb_cfg,
        settings=conversation_settings,
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
            "mode": _resolve_provider_mode(embed_decision.provider),
        },
    )  # docstring: provider_snapshot 写入 ctx

    keyword_top_k_raw = _resolve_int_value(
        key="keyword_top_k",
        context=context_dict,
        kb={},
        settings=conversation_settings,
        default=200,
    )  # docstring: keyword top_k 原始值
    vector_top_k_raw = _resolve_int_value(
        key="vector_top_k",
        context=context_dict,
        kb={},
        settings=conversation_settings,
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
        settings=conversation_settings,
        milvus_collection=kb_collection,
        vector_top_k=vector_top_k_record,
        keyword_top_k=keyword_top_k,
    )  # docstring: 构造 retrieval 配置
    # IMPORTANT: Do NOT inject extra keys into retrieval_config unless the retrieval pipeline schema
    # explicitly allows them (many schema use extra="forbid").
    # Keep the "vector enabled" semantics in the service layer: allow_vector controls whether we embed
    # and whether we pass query_vector.
    if allow_vector and (not embed_decision.provider or not embed_decision.model):
        raise BadRequestError(
            message="embed_provider/embed_model is required"
        )  # docstring: 向量检索必须有 provider/model

    default_chat_provider = "ollama" if settings.LOCAL_MODELS else "dashscope"  # docstring: chat provider 默认值
    default_chat_model = (
        settings.OLLAMA_CHAT_MODEL if settings.LOCAL_MODELS else "qwen3-max"
    )  # docstring: chat model 默认值
    llm_decision = _resolve_llm_decision(
        context=context_dict,
        kb=kb_cfg,
        settings=conversation_settings,
        default_provider=str(default_chat_provider),
        default_model=str(default_chat_model),
    )  # docstring: LLM 决策
    llm_decision = _check_llm_entitlement(llm_decision)  # docstring: LLM entitlement 检查
    if not llm_decision.entitled:
        raise BadRequestError(
            message=f"model_provider not allowed: {llm_decision.provider}"
        )  # docstring: provider 不在 allowlist

    # Ensure LLM provider snapshot is written BEFORE generation, so generation pipeline can persist it.
    ctx.with_provider(
        "llm",
        {
            "provider": llm_decision.provider,
            "model": llm_decision.model,
            "source": llm_decision.source,
            "entitled": llm_decision.entitled,
            "entitlement_reason": llm_decision.entitlement_reason,
            "mode": _resolve_provider_mode(llm_decision.provider),
        },
    )  # docstring: LLM provider_snapshot 写入 ctx（generation 前）

    generation_config = _build_generation_config(
        context=context_dict,
        kb=kb_cfg,
        settings=conversation_settings,
        llm=llm_decision,
    )  # docstring: generation 配置
    evaluator_config = _build_evaluator_config(
        context=context_dict,
        kb=kb_cfg,
        settings=conversation_settings,
    )  # docstring: evaluator 配置

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
    generation_record_id: Optional[str] = None
    generation_provider_snapshot: Optional[Dict[str, Any]] = None
    generation_timing_ms: Optional[Dict[str, Any]] = None
    evaluation_record_id: Optional[str] = None
    evaluation_timing_ms: Optional[Dict[str, Any]] = None
    retrieval_gate: Optional[RetrievalGateDecision] = None
    generation_gate: Optional[GenerationGateDecision] = None
    evaluator_gate: Optional[EvaluatorGateDecision] = None
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
        await session.commit()  # docstring: 提交 retrieval 结果（可回放）

        retrieval_gate = _evaluate_retrieval_gate(hits_count=hits_count or 0)  # docstring: retrieval gate 裁决

        msg_row = await msg_repo.get_by_id(message_id)  # docstring: 回查 message 以写回状态
        if msg_row is None:
            raise InternalError(message="message not found for status update")  # docstring: message 必须存在

        if retrieval_gate is not None and not retrieval_gate.passed:
            msg_row.status = MESSAGE_STATUS_BLOCKED  # docstring: Gate blocked
            msg_row.error_message = "no_evidence"  # docstring: 阻断原因
            msg_row.response = ""  # docstring: blocked 不返回 answer
            await session.flush()  # docstring: 写回 message 状态
            await session.commit()  # docstring: 提交 blocked 状态
            state = _advance_state(state, STATE_MESSAGE_BLOCKED)  # docstring: 状态推进到 BLOCKED

            evaluator_summary = _build_evaluator_summary(
                evaluator=None,
                fallback_status="skipped",
                fallback_rule_version=str(evaluator_config.rule_version),
                fallback_reasons=list(retrieval_gate.reasons),
            )  # docstring: evaluator 摘要（blocked）
            total_ms = (time.perf_counter() - start_ts) * 1000.0  # docstring: 总耗时
            timing_ms = {TIMING_TOTAL_MS_KEY: total_ms}  # docstring: 总耗时快照
            debug_payload = (
                _build_debug_payload(
                    retrieval_record_id=retrieval_record_id,
                    generation_record_id=None,
                    evaluation_record_id=None,
                    retrieval_gate=retrieval_gate,
                    generation_gate=None,
                    evaluator_gate=None,
                    provider_snapshot=_merge_provider_snapshot(retrieval_provider_snapshot, ctx.provider_snapshot),
                    timing_ms={"retrieval": retrieval_timing_ms},
                    hits_count=hits_count,
                )
                if debug
                else None
            )

            log_event(
                logger,
                logging.INFO,
                "chat.blocked",
                context=ctx,
                fields={"message_id": message_id, "reasons": list(retrieval_gate.reasons)},
            )  # docstring: blocked 日志

            response: Dict[str, Any] = {
                "conversation_id": conv_id,
                "message_id": message_id,
                "kb_id": kb_id_final,
                "status": MESSAGE_STATUS_BLOCKED,
                "answer": "",
                "citations": [],
                "evaluator": evaluator_summary,
                TIMING_MS_KEY: timing_ms,
                TRACE_ID_KEY: str(ctx.trace_id),
                REQUEST_ID_KEY: str(ctx.request_id),
            }  # docstring: blocked 返回结果
            if debug_payload is not None:
                response[DEBUG_KEY] = debug_payload  # docstring: debug 输出
            return response

        state = _advance_state(state, STATE_GENERATION_DONE)  # docstring: 状态推进到 GENERATION_DONE
        current_stage = "generation"
        generation_bundle = await run_generation_pipeline(
            session=session,
            generation_repo=generation_repo,
            message_id=message_id,
            retrieval_bundle=bundle,
            config=generation_config,
            ctx=ctx,
        )  # docstring: 执行 generation pipeline

        generation_record_id = str(generation_bundle.record.id)  # docstring: generation_record_id
        # Prefer record-native fields for audit. Avoid reading from messages_snapshot.
        generation_timing_ms = dict(getattr(generation_bundle.record, "timing_ms", {}) or {})  # type: ignore[assignment]
        generation_provider_snapshot = dict(getattr(generation_bundle.record, "provider_snapshot", {}) or {})  # type: ignore[assignment]
        generation_gate = _evaluate_generation_gate(generation_bundle)  # docstring: generation gate 记录
        await session.commit()  # docstring: 提交 generation 结果（可回放）

        state = _advance_state(state, STATE_EVALUATION_DONE)  # docstring: 状态推进到 EVALUATION_DONE
        current_stage = "evaluator"
        evaluator_input = {
            "conversation_id": conv_id,
            "message_id": message_id,
            "retrieval_bundle": bundle,
            "generation_bundle": generation_bundle,
            "config": evaluator_config,
        }  # docstring: evaluator 输入快照
        evaluator_input_typed = cast(
            evaluator_checks.EvaluatorInput, evaluator_input
        )  # docstring: 类型收窄以匹配 EvaluatorInput
        evaluation_result = await run_evaluator_pipeline(
            session=session,
            evaluator_repo=evaluator_repo,
            input=evaluator_input_typed,
            ctx=ctx,
        )  # docstring: 执行 evaluator pipeline
        evaluation_record_id = str(evaluation_result.id)  # docstring: evaluation_record_id
        evaluation_timing_ms = dict(
            (evaluation_result.meta or {}).get(TIMING_MS_KEY, {}) or {}
        )  # docstring: evaluator timing_ms
        evaluator_gate = _evaluate_evaluator_gate(evaluation_result)  # docstring: evaluator gate 记录
        await session.commit()  # docstring: 提交 evaluator 结果（可回放）

        status = _map_evaluation_status(str(evaluation_result.status))  # docstring: 映射最终 status
        answer_text = str(generation_bundle.answer or "").strip()  # docstring: answer 输出
        citations = _extract_generation_citations(generation_bundle)  # docstring: citations 输出
        if status not in {MESSAGE_STATUS_SUCCESS, MESSAGE_STATUS_PARTIAL}:
            answer_text = ""  # docstring: failed 不返回 answer
            citations = []  # docstring: failed 不返回 citations

        msg_row = await msg_repo.get_by_id(message_id)  # docstring: 回查 message 写回状态
        if msg_row is None:
            raise InternalError(message="message not found for status update")  # docstring: message 必须存在
        msg_row.response = answer_text  # docstring: 写回回答
        msg_row.status = status  # docstring: 写回最终状态
        if status == MESSAGE_STATUS_FAILED:
            failure_reasons = "; ".join(evaluator_gate.reasons) if evaluator_gate else ""  # docstring: 失败原因汇总
            msg_row.error_message = failure_reasons or "evaluator_failed"  # docstring: 失败原因
        elif status == MESSAGE_STATUS_PARTIAL:
            msg_row.error_message = (
                "; ".join(evaluator_gate.reasons) if evaluator_gate else None
            )  # docstring: partial 提示
        else:
            msg_row.error_message = None  # docstring: success 清空错误
        await session.flush()  # docstring: flush message 更新
        await session.commit()  # docstring: 提交 message 状态

        log_event(
            logger,
            logging.INFO,
            "chat.evaluator_done",
            context=ctx,
            fields={
                "message_id": message_id,
                "status": status,
                "evaluation_status": str(evaluation_result.status),
            },
        )  # docstring: evaluator 完成日志

        if status == MESSAGE_STATUS_SUCCESS:
            state = _advance_state(state, STATE_MESSAGE_SUCCESS)  # docstring: 状态推进到 SUCCESS
        elif status == MESSAGE_STATUS_PARTIAL:
            state = _advance_state(state, STATE_MESSAGE_PARTIAL)  # docstring: 状态推进到 PARTIAL
        else:
            state = _advance_state(state, STATE_MESSAGE_FAILED)  # docstring: 状态推进到 FAILED

        total_ms = (time.perf_counter() - start_ts) * 1000.0  # docstring: 总耗时
        timing_ms = {TIMING_TOTAL_MS_KEY: total_ms}  # docstring: 总耗时快照
        debug_payload = (
            _build_debug_payload(
                retrieval_record_id=retrieval_record_id,
                generation_record_id=generation_record_id,
                evaluation_record_id=evaluation_record_id,
                retrieval_gate=retrieval_gate,
                generation_gate=generation_gate,
                evaluator_gate=evaluator_gate,
                provider_snapshot=_merge_provider_snapshot(
                    retrieval_provider_snapshot, generation_provider_snapshot, ctx.provider_snapshot
                ),
                timing_ms={
                    "retrieval": retrieval_timing_ms,
                    "generation": generation_timing_ms,
                    "evaluator": evaluation_timing_ms,
                },
                hits_count=hits_count,
            )
            if debug
            else None
        )

        evaluator_summary = _build_evaluator_summary(
            evaluator=evaluation_result,
            fallback_status="skipped",
            fallback_rule_version=str(evaluator_config.rule_version),
            fallback_reasons=(),
        )  # docstring: evaluator 摘要

        response: Dict[str, Any] = {
            "conversation_id": conv_id,
            "message_id": message_id,
            "kb_id": kb_id_final,
            "status": status,
            "answer": answer_text,
            "citations": citations,
            "evaluator": evaluator_summary,
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
