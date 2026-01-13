# src/uae_law_rag/backend/pipelines/evaluator/pipeline.py

"""
[职责] evaluator pipeline：编排 checks 执行、结果汇总与 EvaluationRecord 落库，输出 EvaluationResult。
[边界] 不执行检索/生成；不提交事务；不实现 DB 访问细节（由 repo/persist 负责）。
[上游关系] services/chat_service 或脚本调用；上游提供 Retrieval/Generation 快照与 EvaluatorConfig。
[下游关系] EvaluatorRepo 写入评估记录；服务层/审计消费 EvaluationResult。
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, cast

from sqlalchemy.ext.asyncio import AsyncSession

from uae_law_rag.backend.db.repo.evaluator_repo import EvaluatorRepo
from uae_law_rag.backend.pipelines.base.context import PipelineContext
from uae_law_rag.backend.schemas.evaluator import (
    CheckStatus,
    EvaluatorConfig,
    EvaluationCheck,
    EvaluationResult,
    EvaluationScores,
    EvaluationStatus,
)
from uae_law_rag.backend.schemas.generation import GenerationRecord
from uae_law_rag.backend.schemas.ids import (
    GenerationRecordId,
    MessageId,
    RetrievalRecordId,
    UUIDStr,
)
from uae_law_rag.backend.schemas.retrieval import RetrievalHit, RetrievalRecord

from . import checks as checks_mod
from . import persist as persist_mod
from . import utils as utils_mod


__all__ = ["run_evaluator_pipeline"]


_CHECKS: Sequence[Tuple[str, Any]] = (
    ("require_citations", checks_mod.check_require_citations),
    ("citation_coverage", checks_mod.check_citation_coverage),
    ("min_answer_length", checks_mod.check_min_answer_length),
    ("no_empty_answer", checks_mod.check_no_empty_answer),
    ("min_retrieval_hits", checks_mod.check_min_retrieval_hits),
    ("require_vector_hits", checks_mod.check_require_vector_hits),
    ("require_keyword_hits", checks_mod.check_require_keyword_hits),
    ("require_structured", checks_mod.check_require_structured),
)  # docstring: 默认 checks 顺序（稳定输出）


def _coerce_str(value: Any) -> str:
    """
    [职责] 将 value 转为字符串并去空白。
    [边界] 空值返回空字符串；不做格式校验。
    [上游关系] 归一化输入字段时调用。
    [下游关系] ID/status/answer 文本规范化。
    """
    return str(value or "").strip()  # docstring: 字符串兜底


def _read_field(obj: Any, key: str) -> Any:
    """
    [职责] 从对象或 mapping 安全读取字段值。
    [边界] 不抛异常；字段不存在返回 None。
    [上游关系] bundle/record 兼容读取时调用。
    [下游关系] 规范化输入字段解析。
    """
    if obj is None:
        return None  # docstring: 空对象回退
    if isinstance(obj, Mapping):
        return obj.get(key)  # docstring: mapping 读取
    return getattr(obj, key, None)  # docstring: attribute 读取


def _as_mapping(value: Any) -> Mapping[str, Any]:
    """
    [职责] 将对象转换为 Mapping（兼容 pydantic v2/v1）。
    [边界] 转换失败返回空 dict；不做业务解释。
    [上游关系] generation_output/output_structured 归一化调用。
    [下游关系] checks 使用 generation_output mapping。
    """
    if value is None:
        return {}  # docstring: 空值回退
    if isinstance(value, Mapping):
        return value  # docstring: mapping 直接返回
    if hasattr(value, "model_dump"):
        try:
            return value.model_dump()  # type: ignore[attr-defined]
        except Exception:
            return {}  # docstring: model_dump 失败兜底
    if hasattr(value, "dict"):
        try:
            return value.dict()  # type: ignore[call-arg]
        except Exception:
            return {}  # docstring: dict 失败兜底
    return {}  # docstring: 无法转换回退空 dict


def _normalize_evaluator_config(raw: Any) -> EvaluatorConfig:
    """
    [职责] 将 raw 配置归一化为 EvaluatorConfig。
    [边界] 配置异常回退默认；不做策略校验。
    [上游关系] run_evaluator_pipeline 调用。
    [下游关系] checks 执行与 rule_version 写入。
    """
    if isinstance(raw, EvaluatorConfig):
        return raw  # docstring: 已是 EvaluatorConfig
    if isinstance(raw, Mapping):
        try:
            return EvaluatorConfig(**dict(raw))  # docstring: mapping 构造配置
        except Exception:
            return EvaluatorConfig()  # docstring: 配置异常回退默认
    return EvaluatorConfig()  # docstring: 缺省配置


def _extract_retrieval_bundle(input: Mapping[str, Any]) -> Any:
    """
    [职责] 从 input 提取 retrieval_bundle（兼容 mapping/对象）。
    [边界] 未提供返回 None。
    [上游关系] run_evaluator_pipeline 输入。
    [下游关系] retrieval_record/hits 兜底来源。
    """
    return input.get("retrieval_bundle")  # docstring: 读取 retrieval_bundle


def _extract_generation_bundle(input: Mapping[str, Any]) -> Any:
    """
    [职责] 从 input 提取 generation_bundle（兼容 mapping/对象）。
    [边界] 未提供返回 None。
    [上游关系] run_evaluator_pipeline 输入。
    [下游关系] generation_record/answer 兜底来源。
    """
    return input.get("generation_bundle")  # docstring: 读取 generation_bundle


def _extract_retrieval_record(input: Mapping[str, Any]) -> Optional[RetrievalRecord]:
    """
    [职责] 解析 retrieval_record（兼容 retrieval_bundle 兜底）。
    [边界] 不校验 record 内容完整性。
    [上游关系] run_evaluator_pipeline 输入。
    [下游关系] EvaluationResult.retrieval_record_id。
    """
    record = input.get("retrieval_record")  # docstring: 优先 retrieval_record
    if record is not None:
        return cast(Optional[RetrievalRecord], record)
    bundle = _extract_retrieval_bundle(input)  # docstring: retrieval_bundle 兜底
    return cast(Optional[RetrievalRecord], _read_field(bundle, "record"))


def _extract_retrieval_hits(input: Mapping[str, Any]) -> List[RetrievalHit]:
    """
    [职责] 解析 retrieval hits（兼容 retrieval_bundle 兜底）。
    [边界] 非序列返回空列表；不校验 hit 内容。
    [上游关系] run_evaluator_pipeline 输入。
    [下游关系] checks/scores 使用 hits。
    """
    hits = input.get("retrieval_hits")  # docstring: 优先 retrieval_hits
    if hits is None:
        bundle = _extract_retrieval_bundle(input)  # docstring: retrieval_bundle 兜底
        hits = _read_field(bundle, "hits")  # docstring: bundle.hits
    if isinstance(hits, Sequence) and not isinstance(hits, (str, bytes, bytearray)):
        return list(hits)  # docstring: 序列转 list
    return []  # docstring: 非序列回退空列表


def _extract_generation_record(input: Mapping[str, Any]) -> Optional[GenerationRecord]:
    """
    [职责] 解析 generation_record（兼容 generation_bundle 兜底）。
    [边界] 不校验 record 内容完整性。
    [上游关系] run_evaluator_pipeline 输入。
    [下游关系] EvaluationResult.generation_record_id。
    """
    record = input.get("generation_record")  # docstring: 优先 generation_record
    if record is not None:
        return cast(Optional[GenerationRecord], record)
    bundle = _extract_generation_bundle(input)  # docstring: generation_bundle 兜底
    return cast(Optional[GenerationRecord], _read_field(bundle, "record"))


def _extract_generation_answer(input: Mapping[str, Any]) -> str:
    """
    [职责] 提取 answer 文本（generation_bundle 优先，fallback 到 generation_output/record）。
    [边界] 不做语义判断；仅做空值回退。
    [上游关系] generation_bundle/record/output 提供答案。
    [下游关系] scores 与 checks 使用。
    """
    bundle = _extract_generation_bundle(input)  # docstring: generation_bundle
    answer = _read_field(bundle, "answer")  # docstring: bundle.answer
    if isinstance(answer, str) and answer.strip():
        return answer.strip()  # docstring: 优先使用 bundle.answer
    output = _as_mapping(input.get("generation_output"))  # docstring: generation_output
    output_answer = output.get("answer")  # docstring: output.answer
    if isinstance(output_answer, str) and output_answer.strip():
        return output_answer.strip()  # docstring: fallback output.answer
    record = _extract_generation_record(input)  # docstring: generation_record
    structured = _as_mapping(_read_field(record, "output_structured"))  # docstring: output_structured
    structured_answer = structured.get("answer")  # docstring: structured.answer
    if isinstance(structured_answer, str) and structured_answer.strip():
        return structured_answer.strip()  # docstring: fallback structured.answer
    return _coerce_str(_read_field(record, "output_raw"))  # docstring: fallback output_raw


def _extract_citations_from_payload(payload: Any) -> List[Any]:
    """
    [职责] 从 citations payload 中提取引用条目列表（items 优先，nodes 兜底）。
    [边界] 不校验 node_id 真实性；非序列返回空列表。
    [上游关系] generation_record.citations 提供 payload。
    [下游关系] generation_output.citations 构建。
    """
    if payload is None:
        return []  # docstring: 空 payload 回退
    if isinstance(payload, Mapping):
        items = payload.get("items")  # docstring: payload.items
        nodes = payload.get("nodes")  # docstring: payload.nodes
    else:
        items = _read_field(payload, "items")  # docstring: object.items
        nodes = _read_field(payload, "nodes")  # docstring: object.nodes
    source = items if items is not None else nodes  # docstring: items 优先 nodes 兜底
    if isinstance(source, Sequence) and not isinstance(source, (str, bytes, bytearray)):
        return list(source)  # docstring: 序列转 list
    return []  # docstring: 非序列回退空列表


def _build_generation_output(input: Mapping[str, Any]) -> Dict[str, Any]:
    """
    [职责] 构建 generation_output 快照（answer/citations/output_structured）。
    [边界] 不做 schema 校验；仅做结构化兜底与字段拼装。
    [上游关系] generation_bundle/generation_record 提供原始输出。
    [下游关系] checks 使用 generation_output。
    """
    raw = input.get("generation_output")  # docstring: 原始 generation_output
    if raw is not None:
        return dict(_as_mapping(raw))  # docstring: 显式 output 优先

    record = _extract_generation_record(input)  # docstring: generation_record
    citations = _extract_citations_from_payload(_read_field(record, "citations"))  # docstring: citations payload
    output_structured = _as_mapping(_read_field(record, "output_structured"))  # docstring: output_structured
    answer = _extract_generation_answer(input)  # docstring: 归一化 answer

    out: Dict[str, Any] = {
        "answer": answer,
        "citations": citations,
    }  # docstring: generation_output 快照
    if output_structured:
        out["output_structured"] = dict(output_structured)  # docstring: 透传结构化输出
    return out


def _aggregate_status(checks: Sequence[EvaluationCheck]) -> EvaluationStatus:
    """
    [职责] 根据 checks 状态汇总总体 status。
    [边界] 未知状态不做处理；仅依赖 pass/fail/warn/skipped。
    [上游关系] run_evaluator_pipeline 在完成 checks 后调用。
    [下游关系] EvaluationResult.status 与落库 status。
    """
    if not checks:
        return "skipped"  # docstring: 无 checks 视为 skipped
    statuses = [c.status for c in checks]  # docstring: 收集状态
    if any(s == "fail" for s in statuses):
        return "fail"  # docstring: 任一 fail -> fail
    if any(s == "warn" for s in statuses):
        return "partial"  # docstring: 存在 warn -> partial
    if all(s == "skipped" for s in statuses):
        return "skipped"  # docstring: 全 skipped -> skipped
    return "pass"  # docstring: 全部 pass


def _build_scores(
    *,
    answer: str,
    citations: Any,
    hits: Sequence[Any],
) -> EvaluationScores:
    """
    [职责] 构建 EvaluationScores（coverage 等数值指标）。
    [边界] 不引入语义模型；仅计算简单覆盖率与计数。
    [上游关系] run_evaluator_pipeline 组装输入后调用。
    [下游关系] EvaluationResult.scores 与落库 scores 快照。
    """
    coverage = utils_mod.compute_citation_coverage(citations, hits)  # docstring: 引用覆盖率
    citation_ids = utils_mod.extract_node_ids(citations)  # docstring: citations node_id
    hit_ids = utils_mod.extract_node_ids(hits)  # docstring: hits node_id
    overall = {"citation_coverage": float(coverage)}  # docstring: overall scores
    per_metric: Dict[str, Any] = {
        "citation_coverage": {
            "value": float(coverage),
            "citations": len(citation_ids),
            "hits": len(hit_ids),
        },
        "answer_length": {"value": float(len(answer or ""))},
    }  # docstring: per-metric 细节
    return EvaluationScores(overall=overall, per_metric=per_metric)  # docstring: scores 组装


def _run_checks(
    *,
    evaluator_input: Mapping[str, Any],
    checks: Sequence[Tuple[str, Any]],
) -> Tuple[List[EvaluationCheck], Optional[str]]:
    """
    [职责] 执行 checks 并收集 EvaluationCheck 列表。
    [边界] 单个 check 异常不会中断全部执行；异常会转为 fail check。
    [上游关系] run_evaluator_pipeline 调用。
    [下游关系] 汇总 status 与落库 checks。
    """
    results: List[EvaluationCheck] = []  # docstring: checks 输出列表
    errors: List[str] = []  # docstring: 错误列表
    for name, fn in checks:
        try:
            check = fn(input=cast(checks_mod.EvaluatorInput, evaluator_input))  # docstring: 执行单条检查
            if check is not None:
                results.append(check)  # docstring: 收集检查结果
        except Exception as exc:
            msg = f"{name}: {exc.__class__.__name__}: {exc}"  # docstring: 异常消息
            errors.append(msg)  # docstring: 记录异常
            results.append(
                EvaluationCheck(
                    name=name,
                    status=cast(CheckStatus, "fail"),
                    message=msg,
                    detail={"error": msg, "exception": exc.__class__.__name__},
                )
            )  # docstring: 异常转为 fail check
    error_message = "; ".join(errors) if errors else None  # docstring: 汇总错误
    return results, error_message


def _build_meta(ctx: PipelineContext) -> Dict[str, Any]:
    """
    [职责] 构建 meta 快照（trace_id/request_id/timing/provider）。
    [边界] 仅做结构化透传；不保证字段完整性。
    [上游关系] run_evaluator_pipeline 调用。
    [下游关系] EvaluationRecord.meta 落库。
    """
    meta: Dict[str, Any] = {
        "trace_id": str(ctx.trace_id),
        "request_id": str(ctx.request_id),
        "parent_request_id": str(ctx.parent_request_id) if ctx.parent_request_id else None,
        "timing_ms": ctx.timing.to_dict(include_total=True, total_key="total"),
    }  # docstring: 基础 meta 快照
    if ctx.trace_tags:
        meta["trace_tags"] = dict(ctx.trace_tags)  # docstring: trace tags
    if ctx.provider_snapshot:
        meta["provider_snapshot"] = dict(ctx.provider_snapshot)  # docstring: provider 快照

    trace_ctx = ctx.as_trace_context()  # docstring: trace context
    if hasattr(trace_ctx, "model_dump"):
        meta["trace"] = trace_ctx.model_dump()  # type: ignore[attr-defined]  # docstring: 兼容 pydantic v2
    elif hasattr(trace_ctx, "dict"):
        meta["trace"] = trace_ctx.dict()  # type: ignore[call-arg]  # docstring: 兼容 pydantic v1
    else:
        meta["trace"] = str(trace_ctx)  # docstring: 兜底为字符串
    return meta


async def run_evaluator_pipeline(
    *,
    session: AsyncSession,
    evaluator_repo: EvaluatorRepo,
    input: checks_mod.EvaluatorInput,
    ctx: Optional[PipelineContext] = None,
) -> EvaluationResult:
    """
    [职责] run_evaluator_pipeline：执行 evaluator checks 并落库 EvaluationRecord。
    [边界] 不提交事务；不调用 retrieval/generation；不访问外部网络。
    [上游关系] services/chat_service 调用；上游提供 retrieval/generation 快照与 config。
    [下游关系] EvaluationResult 返回给 service；EvaluationRecord 写入 DB。
    """
    if input is None:
        raise ValueError("input is required")  # docstring: 必填且不可为空

    conversation_id = _coerce_str(input.get("conversation_id"))  # docstring: conversation_id
    if not conversation_id:
        raise ValueError("conversation_id is required")  # docstring: 必填且不可为空

    retrieval_record = _extract_retrieval_record(input)  # docstring: retrieval_record
    if retrieval_record is None:
        raise ValueError("retrieval_record is required")  # docstring: 必填且不可为空

    message_id = _coerce_str(
        input.get("message_id") or _read_field(retrieval_record, "message_id")
    )  # docstring: message_id
    if not message_id:
        raise ValueError("message_id is required")  # docstring: 必填且不可为空

    retrieval_record_id = _coerce_str(_read_field(retrieval_record, "id"))  # docstring: retrieval_record_id
    if not retrieval_record_id:
        raise ValueError("retrieval_record_id is required")  # docstring: 必填且不可为空

    generation_record = _extract_generation_record(input)  # docstring: generation_record
    generation_record_id = _coerce_str(_read_field(generation_record, "id")) or None  # docstring: generation_record_id

    retrieval_hits = _extract_retrieval_hits(input)  # docstring: retrieval hits
    generation_output = _build_generation_output(input)  # docstring: generation_output
    answer = _extract_generation_answer(input)  # docstring: answer 文本

    cfg = _normalize_evaluator_config(input.get("config"))  # docstring: 归一化配置

    ctx = ctx or PipelineContext.from_session(session)  # docstring: 统一 ctx 装配
    ctx.timing.reset()  # docstring: 清理上次 timing

    evaluator_input: Dict[str, Any] = {
        "conversation_id": conversation_id,
        "message_id": message_id,
        "retrieval_record": retrieval_record,
        "retrieval_hits": retrieval_hits,
        "generation_record": generation_record,
        "generation_output": generation_output,
        "config": cfg,
    }  # docstring: 统一 checks 输入快照

    with ctx.timing.stage("checks"):
        checks, error_message = _run_checks(evaluator_input=evaluator_input, checks=_CHECKS)  # docstring: 执行 checks

    with ctx.timing.stage("scores"):
        scores = _build_scores(
            answer=answer,
            citations=generation_output.get("citations"),
            hits=retrieval_hits,
        )  # docstring: 构建 scores

    status = _aggregate_status(checks)  # docstring: 汇总状态
    meta = _build_meta(ctx)  # docstring: meta 快照

    record_params = {
        "conversation_id": conversation_id,
        "message_id": message_id,
        "retrieval_record_id": retrieval_record_id,
        "generation_record_id": generation_record_id,
        "status": status,
        "rule_version": cfg.rule_version,
        "config": cfg,
        "checks": checks,
        "scores": scores,
        "error_message": error_message,
        "meta": meta,
    }  # docstring: EvaluationRecord 参数快照

    evaluation_record_id = await persist_mod.persist_evaluation(
        evaluator_repo=evaluator_repo,
        record_params=record_params,
    )  # docstring: 写入评估记录

    result = EvaluationResult(
        id=cast(UUIDStr, UUIDStr(evaluation_record_id)),  # docstring: 使用落库 ID
        status=cast(EvaluationStatus, status),
        message_id=cast(MessageId, UUIDStr(str(message_id))),
        retrieval_record_id=cast(RetrievalRecordId, UUIDStr(str(retrieval_record_id))),
        generation_record_id=cast(GenerationRecordId, UUIDStr(str(generation_record_id)))
        if generation_record_id
        else None,
        config=cfg,
        checks=checks,
        scores=scores,
        error_message=error_message,
        meta=meta,
    )  # docstring: 构造 EvaluationResult

    return result  # docstring: 返回 EvaluationResult
