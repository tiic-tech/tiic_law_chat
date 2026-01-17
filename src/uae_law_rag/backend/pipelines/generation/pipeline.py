# src/uae_law_rag/backend/pipelines/generation/pipeline.py

"""
[职责] generation pipeline：编排 prompt → LLM → postprocess → persist 的生成全链路，产出可回放 GenerationBundle。
[边界] 不创建 Message/KB；不提交事务；不做 evaluator 判定；不处理外部网关错误重试。
[上游关系] services/chat_service 或脚本调用；依赖 RetrievalBundle 与 GenerationRepo。
[下游关系] GenerationRecord 落库供审计；evaluator/chat 使用回答与 citations。
"""

from __future__ import annotations

import re
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, cast

from sqlalchemy.ext.asyncio import AsyncSession

from uae_law_rag.backend.db.repo.generation_repo import GenerationRepo
from uae_law_rag.backend.pipelines.base.context import PipelineContext
from uae_law_rag.backend.schemas.generation import (
    Citation,
    CitationsPayload,
    GenerationBundle,
    GenerationRecord,
    GenerationStatus,
)
from uae_law_rag.backend.schemas.ids import GenerationRecordId, MessageId, RetrievalRecordId, UUIDStr
from uae_law_rag.backend.schemas.retrieval import RetrievalBundle, RetrievalHit
from uae_law_rag.backend.utils.constants import (
    MESSAGE_ID_KEY,
    PROMPT_NAME_KEY,
    PROMPT_VERSION_KEY,
    PROVIDER_SNAPSHOT_KEY,
    RETRIEVAL_RECORD_ID_KEY,
    TIMING_MS_KEY,
    TIMING_TOTAL_KEY,
    TRACE_KEY,
)

from . import generator as generator_mod
from . import persist as persist_mod
from . import postprocess as postprocess_mod
from . import prompt as prompt_mod


# NOTE:
# status order controls merge + normalization.
# "blocked" means: retrieval has evidence, but generation produced no verifiable citations.
# It is not an infrastructure failure, so MUST NOT be forced to "failed".

# Example (adjust to your actual constants):
_STATUS_ORDER = {
    "success": 0,
    "partial": 1,
    "blocked": 2,
    "failed": 3,
}


@dataclass(frozen=True)
class _GenerationConfig:
    """Normalized generation config."""  # docstring: 内部使用的配置快照

    prompt_name: str
    prompt_version: Optional[str]
    max_excerpt_chars: int
    node_snapshots: Optional[Mapping[str, Mapping[str, Any]]]
    model_provider: str
    model_name: str
    generation_config: Dict[str, Any]
    postprocess_config: Dict[str, Any]
    no_evidence_answer: str
    no_evidence_status: str
    no_evidence_use_llm: bool


def _normalize_config(config: Optional[Mapping[str, Any]]) -> _GenerationConfig:
    """
    [职责] 归一化 generation 配置（补齐默认值并转换类型）。
    [边界] 不做业务策略校验；仅处理缺省与类型。
    [上游关系] run_generation_pipeline 调用。
    [下游关系] prompt/generator/postprocess 使用。
    """
    cfg = dict(config or {})  # docstring: 复制配置
    prompt_cfg = dict(cfg.get("prompt_config") or cfg.get("prompt") or {})  # docstring: prompt 配置
    gen_cfg = dict(
        cfg.get("generation_config") or cfg.get("llm_config") or cfg.get("generation") or {}
    )  # docstring: LLM 配置
    post_cfg = dict(cfg.get("postprocess_config") or cfg.get("postprocess") or {})  # docstring: postprocess 配置

    def _as_str(key: str, default: str = "") -> str:
        val = cfg.get(key, default)
        return str(default if val is None else val).strip()  # docstring: str 兜底

    def _as_opt_str(key: str) -> Optional[str]:
        val = cfg.get(key)
        if val is None:
            return None
        s = str(val).strip()
        return s or None  # docstring: 空字符串回退 None

    def _as_int(key: str, default: int) -> int:
        val = cfg.get(key, default)
        try:
            return int(default if val is None else val)  # docstring: int 兜底
        except (TypeError, ValueError):
            return int(default)  # docstring: 异常回退

    def _as_bool(key: str, default: bool) -> bool:
        val = cfg.get(key, default)
        return bool(default if val is None else val)  # docstring: bool 兜底

    prompt_name = _as_str(
        PROMPT_NAME_KEY, str(prompt_cfg.get(PROMPT_NAME_KEY) or prompt_cfg.get("name") or "")
    )  # docstring: prompt 名称
    prompt_version = _as_opt_str(PROMPT_VERSION_KEY)  # docstring: prompt 版本
    if prompt_version is None:
        pv_raw = prompt_cfg.get(PROMPT_VERSION_KEY) or prompt_cfg.get("version")  # docstring: prompt 版本兜底值
        pv_text = str(pv_raw or "").strip()  # docstring: prompt 版本归一化
        prompt_version = pv_text or None  # docstring: 空字符串回退 None
    max_excerpt_chars = _as_int(
        "max_excerpt_chars", prompt_cfg.get("max_excerpt_chars", prompt_mod.DEFAULT_MAX_EXCERPT_CHARS)
    )  # docstring: excerpt 长度
    node_snapshots = prompt_cfg.get("node_snapshots") or cfg.get("node_snapshots")  # docstring: node 快照映射

    model_provider = _as_str(
        "model_provider",
        gen_cfg.get("model_provider") or gen_cfg.get("provider") or "mock",
    )  # docstring: provider
    model_name = _as_str(
        "model_name",
        gen_cfg.get("model_name") or gen_cfg.get("model") or "mock",
    )  # docstring: model_name

    no_evidence_answer = _as_str(
        "no_evidence_answer",
        "The provided evidence is insufficient to answer this question.",
    )  # docstring: no_evidence 回答
    no_evidence_status = _as_str("no_evidence_status", "failed").lower()  # docstring: no_evidence 状态
    if no_evidence_status not in _STATUS_ORDER:
        no_evidence_status = "failed"  # docstring: 状态兜底

    return _GenerationConfig(
        prompt_name=prompt_name,
        prompt_version=prompt_version,
        max_excerpt_chars=int(max_excerpt_chars),
        node_snapshots=node_snapshots if isinstance(node_snapshots, Mapping) else None,
        model_provider=str(model_provider or "mock"),
        model_name=str(model_name or "mock"),
        generation_config=gen_cfg,
        postprocess_config=post_cfg,
        no_evidence_answer=no_evidence_answer,
        no_evidence_status=no_evidence_status,
        no_evidence_use_llm=_as_bool("no_evidence_use_llm", False),
    )


def _merge_status(current: str, new_status: str) -> str:
    """
    [职责] 合并 status（选择更严重的状态）。
    [边界] 未知 status 视为 current。
    [上游关系] run_generation_pipeline 调用。
    [下游关系] GenerationRecord.status。
    """
    if new_status not in _STATUS_ORDER:
        return current  # docstring: 未知 status 不更新
    if current not in _STATUS_ORDER:
        return new_status  # docstring: current 异常时回退 new_status
    return new_status if _STATUS_ORDER[new_status] > _STATUS_ORDER[current] else current  # docstring: 选择更严重


def _timing_snapshot(ctx: PipelineContext) -> Dict[str, float]:
    """
    [职责] 导出 generation timing 快照（key 统一为 total）。
    [边界] 不做字段裁剪；直接导出 TimingCollector dict。
    [上游关系] run_generation_pipeline 调用。
    [下游关系] messages_snapshot.timing_ms。
    """
    return ctx.timing.to_dict(include_total=True, total_key=TIMING_TOTAL_KEY)  # docstring: total 使用一致 key


def _build_provider_snapshot(
    *,
    base_snapshot: Dict[str, Any],
    cfg: _GenerationConfig,
    prompt_name: str,
    prompt_version: Optional[str],
    usage: Optional[Mapping[str, Any]],
    errors: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """
    [职责] 构建 provider_snapshot（生成配置与错误快照）。
    [边界] 不校验 provider 语义；仅做聚合与透传。
    [上游关系] run_generation_pipeline 调用。
    [下游关系] messages_snapshot.provider_snapshot。
    """
    snapshot = dict(base_snapshot or {})  # docstring: 基础快照复制
    snapshot["prompt"] = {
        PROMPT_NAME_KEY: prompt_name,
        PROMPT_VERSION_KEY: prompt_version,
        "max_excerpt_chars": cfg.max_excerpt_chars,
    }  # docstring: prompt 快照
    snapshot["generation"] = {
        "model_provider": cfg.model_provider,
        "model_name": cfg.model_name,
        "generation_config": dict(cfg.generation_config or {}),
    }  # docstring: generation 快照
    if usage:
        snapshot["generation"]["usage"] = dict(usage)  # docstring: usage 快照
    if errors:
        snapshot["generation"]["errors"] = dict(errors)  # docstring: 错误快照
    return snapshot


def _build_no_evidence_raw(answer: str) -> str:
    """
    [职责] 构造 no_evidence 的 JSON 输出。
    [边界] 不添加 citations；保持最小结构。
    [上游关系] run_generation_pipeline 调用。
    [下游关系] postprocess 解析输出。
    """
    payload = {"answer": str(answer or ""), "citations": []}  # docstring: no_evidence payload
    return json.dumps(payload, ensure_ascii=True)  # docstring: 输出 JSON 字符串


def _build_error_raw(reason: str, detail: str) -> str:
    """
    [职责] 构造错误 JSON 输出（用于落库与审计）。
    [边界] 不包含敏感信息；仅用于失败兜底。
    [上游关系] run_generation_pipeline 调用。
    [下游关系] postprocess 解析输出。
    """
    message = f"Generation failed: {reason}"  # docstring: 失败回答
    payload = {"answer": message, "citations": [], "error": str(detail or "")}  # docstring: 错误 payload
    return json.dumps(payload, ensure_ascii=True)  # docstring: 输出 JSON 字符串


def _extract_answer_from_raw(raw_text: str) -> str:
    """
    [职责] 从 raw_text 尝试提取 answer（fallback）。
    [边界] 仅尝试 JSON 解析（支持 code fence / 包裹文本）；失败返回空。
    """
    text0 = str(raw_text or "").strip()
    if not text0:
        return ""

    # local minimal fence stripping (avoid importing private helpers)
    if text0.lstrip().startswith("```"):
        text0 = re.sub(r"^\s*```(?:json)?\s*\n?", "", text0, flags=re.IGNORECASE).strip()
        text0 = re.sub(r"\n?\s*```\s*$", "", text0).strip()

    start = text0.find("{")
    end = text0.rfind("}")
    if start < 0 or end < 0 or end <= start:
        return ""

    try:
        data = json.loads(text0[start : end + 1])
    except Exception:
        return ""
    if isinstance(data, dict) and isinstance(data.get("answer"), str):
        return str(data.get("answer") or "").strip()
    return ""


def _build_citations_payload(citations: Sequence[Citation]) -> CitationsPayload:
    """
    [职责] 构建 CitationsPayload（nodes + items）。
    [边界] 不校验 node_id 存在性；仅做结构聚合。
    [上游关系] run_generation_pipeline 调用。
    [下游关系] GenerationRecord.citations。
    """
    if not citations:
        return CitationsPayload()  # docstring: 空 citations 直接返回默认
    nodes = [c.node_id for c in citations]  # docstring: node_id 列表
    # docstring: 兼容 pydantic v2 / v1 / dataclass-like
    items: List[Dict[str, Any]] = []
    for c in citations:
        if hasattr(c, "model_dump"):
            items.append(c.model_dump())  # type: ignore[attr-defined]
        elif hasattr(c, "dict"):
            items.append(c.dict())  # type: ignore[call-arg]
        else:
            items.append(dict(getattr(c, "__dict__", {})))
    return CitationsPayload(version="v1", nodes=nodes, items=items, meta={})  # docstring: payload 组装


def _fallback_messages_snapshot(
    *,
    query_text: str,
    prompt_name: str,
    prompt_version: Optional[str],
    error: str,
) -> Dict[str, Any]:
    """
    [职责] 构造 prompt 失败时的最小 messages_snapshot。
    [边界] 不生成实际 prompt；仅记录必要上下文。
    [上游关系] run_generation_pipeline 调用。
    [下游关系] GenerationRecord.messages_snapshot。
    """
    return {
        PROMPT_NAME_KEY: prompt_name,
        PROMPT_VERSION_KEY: prompt_version,
        "query": str(query_text or ""),
        "messages": [],
        "evidence": [],
        "error": str(error or ""),
    }  # docstring: 最小快照


async def run_generation_pipeline(
    *,
    session: AsyncSession,
    generation_repo: GenerationRepo,
    message_id: str,
    retrieval_bundle: RetrievalBundle,
    config: Optional[Mapping[str, Any]] = None,
    ctx: Optional[PipelineContext] = None,
) -> GenerationBundle:
    """
    [职责] run_generation_pipeline：执行生成链路并落库，返回 GenerationBundle。
    [边界] 不提交事务；不创建 message/KB；不执行 evaluator。
    [上游关系] services/chat_service 或脚本调用；上游提供 RetrievalBundle 与模型配置。
    [下游关系] GenerationRecord 写入 DB；evaluator/chat 使用结果。
    """
    message_id = str(message_id or "").strip()  # docstring: message_id 兜底
    if not message_id:
        raise ValueError("message_id is required")  # docstring: 必填且不可为空
    if retrieval_bundle is None:
        raise ValueError("retrieval_bundle is required")  # docstring: 必填且不可为空

    retrieval_record_id = str(getattr(retrieval_bundle.record, "id", "") or "").strip()  # docstring: 检索记录ID
    if not retrieval_record_id:
        raise ValueError("retrieval_record_id is required")  # docstring: 必填且不可为空

    query_text = str(getattr(retrieval_bundle.record, "query_text", "") or "").strip()  # docstring: query 快照
    hits: List[RetrievalHit] = list(retrieval_bundle.hits or [])  # docstring: 命中快照

    ctx = ctx or PipelineContext.from_session(session)  # docstring: 统一 ctx 装配
    cfg = _normalize_config(config)  # docstring: 归一化配置

    ctx.timing.reset()  # docstring: 清理上次 timing

    messages_snapshot: Dict[str, Any] = {}  # docstring: prompt 快照
    prompt_error: Optional[str] = None  # docstring: prompt 错误

    with ctx.timing.stage("prompt"):
        try:
            messages_snapshot = prompt_mod.build_messages(
                query=query_text,
                hits=hits,
                prompt_name=cfg.prompt_name,
                prompt_version=cfg.prompt_version,
                node_snapshots=cfg.node_snapshots,
                max_excerpt_chars=cfg.max_excerpt_chars,
            )  # docstring: 构建 prompt/messages_snapshot
        except Exception as exc:
            prompt_error = f"{exc.__class__.__name__}: {exc}"  # docstring: 记录 prompt 错误
            messages_snapshot = _fallback_messages_snapshot(
                query_text=query_text,
                prompt_name=cfg.prompt_name or "",
                prompt_version=cfg.prompt_version,
                error=prompt_error,
            )  # docstring: 兜底 messages_snapshot

    prompt_name = str(messages_snapshot.get(PROMPT_NAME_KEY) or cfg.prompt_name or "")  # docstring: prompt 名称
    if not prompt_name:
        prompt_name = prompt_mod.DEFAULT_PROMPT_NAME  # docstring: prompt 名称兜底
    prompt_version = cast(
        Optional[str], messages_snapshot.get(PROMPT_VERSION_KEY) or cfg.prompt_version
    )  # docstring: prompt 版本
    messages_snapshot[PROMPT_NAME_KEY] = prompt_name  # docstring: 回写 prompt_name
    messages_snapshot[PROMPT_VERSION_KEY] = prompt_version  # docstring: 回写 prompt_version

    ctx.with_provider(
        "llm",
        {
            "provider": cfg.model_provider,
            "model": cfg.model_name,
            "generation_config": dict(cfg.generation_config or {}),
        },
    )  # docstring: 记录初始 LLM provider 快照

    gen_usage: Optional[Mapping[str, Any]] = None  # docstring: LLM usage 快照
    gen_error: Optional[str] = None  # docstring: LLM 错误
    raw_text = ""  # docstring: LLM 输出原文

    if prompt_error:
        ctx.timing.add_ms("llm", 0.0, accumulate=False)  # docstring: prompt 失败跳过 LLM
        gen_error = f"prompt_error: {prompt_error}"  # docstring: 记录错误
        raw_text = _build_error_raw("prompt_error", prompt_error)  # docstring: 生成错误 JSON
    elif not hits and not cfg.no_evidence_use_llm:
        ctx.timing.add_ms("llm", 0.0, accumulate=False)  # docstring: 无证据跳过 LLM
        gen_error = "no_evidence"  # docstring: 记录 no_evidence
        raw_text = _build_no_evidence_raw(cfg.no_evidence_answer)  # docstring: no_evidence JSON 输出
    else:
        with ctx.timing.stage("llm"):
            try:
                gen_result = await generator_mod.run_generation(
                    messages_snapshot=messages_snapshot,
                    model_provider=cfg.model_provider,
                    model_name=cfg.model_name,
                    generation_config=cfg.generation_config,
                )  # docstring: 调用 LLM
                raw_text = str(gen_result.get("raw_text") or "")  # docstring: raw_text 快照
                cfg = _GenerationConfig(
                    prompt_name=cfg.prompt_name,
                    prompt_version=cfg.prompt_version,
                    max_excerpt_chars=cfg.max_excerpt_chars,
                    node_snapshots=cfg.node_snapshots,
                    model_provider=str(gen_result.get("provider") or cfg.model_provider),
                    model_name=str(gen_result.get("model") or cfg.model_name),
                    generation_config=cfg.generation_config,
                    postprocess_config=cfg.postprocess_config,
                    no_evidence_answer=cfg.no_evidence_answer,
                    no_evidence_status=cfg.no_evidence_status,
                    no_evidence_use_llm=cfg.no_evidence_use_llm,
                )  # docstring: provider/model 以实际结果为准
                gen_usage = gen_result.get("usage")  # docstring: usage 快照
            except Exception as exc:
                gen_error = f"{exc.__class__.__name__}: {exc}"  # docstring: 记录 LLM 错误
                raw_text = _build_error_raw("generation_error", gen_error)  # docstring: 错误 JSON 输出

    ctx.with_provider(
        "llm",
        {
            "provider": cfg.model_provider,
            "model": cfg.model_name,
            "generation_config": dict(cfg.generation_config or {}),
        },
    )  # docstring: 更新 LLM provider 快照

    if not raw_text.strip():
        raw_text = _build_error_raw("empty_output", "raw_text is empty")  # docstring: 空输出兜底

    with ctx.timing.stage("postprocess"):
        post_result = postprocess_mod.postprocess_generation(
            raw_text=raw_text,
            hits=hits,
            config=cfg.postprocess_config,
        )  # docstring: 解析输出与 citations 对齐

    messages_snapshot["postprocess_snapshot"] = {
        "status": post_result.get("status"),
        "error_message": post_result.get("error_message"),
        "citations_count": len(post_result.get("citations") or []),
        "answer_head": str(post_result.get("answer") or "")[:80],
    }

    status = str(post_result.get("status") or "failed")  # docstring: postprocess 状态
    if gen_error:
        status = _merge_status(status, "failed")  # docstring: LLM 错误标记 failed
    # docstring: no_evidence 策略仅在明确“跳过 LLM”的路径中强制应用，避免覆盖 postprocess 策略
    if not hits and not cfg.no_evidence_use_llm:
        status = _merge_status(status, cfg.no_evidence_status)  # docstring: no_evidence 状态策略
    if status not in _STATUS_ORDER:
        status = "failed"  # docstring: 状态兜底

    error_messages: List[str] = []  # docstring: 错误汇总
    if post_result.get("error_message"):
        error_messages.append(str(post_result.get("error_message")))  # docstring: postprocess 错误
    if gen_error:
        error_messages.append(str(gen_error))  # docstring: LLM 错误
    # docstring: 仅在跳过 LLM 的 no_evidence 分支记录该标记，避免误导审计
    if not hits and not cfg.no_evidence_use_llm:
        error_messages.append("no_evidence")  # docstring: no_evidence 标记
    error_message = "; ".join([m for m in error_messages if m]) or None  # docstring: 错误合并

    citations: List[Citation] = list(post_result.get("citations") or [])  # docstring: 引用列表
    output_structured = post_result.get("output_structured")  # docstring: 结构化输出
    answer = str(post_result.get("answer") or "").strip()  # docstring: answer
    if not answer:
        answer = _extract_answer_from_raw(raw_text)  # docstring: fallback 提取 answer

    provider_snapshot = _build_provider_snapshot(
        base_snapshot=dict(ctx.provider_snapshot),
        cfg=cfg,
        prompt_name=prompt_name,
        prompt_version=prompt_version,
        usage=gen_usage,
        errors={"generation_error": gen_error} if gen_error else None,
    )  # docstring: provider 快照
    timing_ms = _timing_snapshot(ctx)  # docstring: timing 快照

    messages_snapshot[PROVIDER_SNAPSHOT_KEY] = provider_snapshot  # docstring: 写入 provider 快照
    messages_snapshot[TIMING_MS_KEY] = timing_ms  # docstring: 写入 timing 快照
    # docstring: trace 快照尽量写入可序列化 dict
    trace_ctx = ctx.as_trace_context()
    if hasattr(trace_ctx, "model_dump"):
        messages_snapshot[TRACE_KEY] = trace_ctx.model_dump()  # type: ignore[attr-defined]
    elif hasattr(trace_ctx, "dict"):
        messages_snapshot[TRACE_KEY] = trace_ctx.dict()  # type: ignore[call-arg]
    else:
        messages_snapshot[TRACE_KEY] = str(trace_ctx)

    record_params = {
        MESSAGE_ID_KEY: message_id,
        RETRIEVAL_RECORD_ID_KEY: retrieval_record_id,
        PROMPT_NAME_KEY: prompt_name or cfg.prompt_name,
        PROMPT_VERSION_KEY: prompt_version,
        "model_provider": cfg.model_provider,
        "model_name": cfg.model_name,
        "messages_snapshot": messages_snapshot,
        "output_raw": raw_text,
        "output_structured": output_structured,
        "citations": citations,
        "status": status,
        "error_message": error_message,
    }  # docstring: GenerationRecord 参数快照

    generation_record_id = await persist_mod.persist_generation(
        generation_repo=generation_repo,
        record_params=record_params,
    )  # docstring: 落库 generation_record

    citations_payload = _build_citations_payload(citations)  # docstring: citations payload
    record = GenerationRecord(
        id=cast(GenerationRecordId, UUIDStr(str(generation_record_id))),  # docstring: generation_record_id
        message_id=cast(MessageId, UUIDStr(str(message_id))),  # docstring: message_id
        retrieval_record_id=cast(
            RetrievalRecordId, UUIDStr(str(retrieval_record_id))
        ),  # docstring: retrieval_record_id
        prompt_name=str(prompt_name or cfg.prompt_name or ""),
        prompt_version=prompt_version,
        model_provider=str(cfg.model_provider),
        model_name=str(cfg.model_name),
        messages_snapshot=messages_snapshot,
        output_raw=str(raw_text),
        output_structured=cast(Optional[Dict[str, Any]], output_structured),
        citations=citations_payload,
        status=cast(GenerationStatus, status),  # docstring: status 已规范化
        error_message=error_message,
    )  # docstring: 构造 GenerationRecord schema

    return GenerationBundle(record=record, answer=answer)  # docstring: 返回 GenerationBundle
