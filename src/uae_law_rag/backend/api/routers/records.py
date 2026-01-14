# src/uae_law_rag/backend/api/routers/records.py

"""
[职责] Records Router：提供检索/生成/评估记录的回放查询接口。
[边界] 只读查询；不触发 pipeline；不做 gate 裁决与事务控制。
[上游关系] 前端通过 /chat debug.records 获取 record_id 后调用。
[下游关系] DB repos 读取记录并映射为 schemas_http/records 视图。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, cast

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from uae_law_rag.backend.api.deps import get_session, get_trace_context
from uae_law_rag.backend.api.errors import to_json_response
from uae_law_rag.backend.api.schemas_http._common import (
    EvaluationRecordId,
    GenerationRecordId,
    KnowledgeBaseId,
    MessageId,
    NodeId,
    RetrievalRecordId,
)
from uae_law_rag.backend.api.schemas_http.records import (
    CitationView,
    CheckStatus,
    EvaluationStatus,
    EvaluationCheckSummary,
    EvaluationRecordView,
    GenerationStatus,
    GenerationRecordView,
    HitSource,
    HitSummary,
    RetrievalRecordView,
    RetrievalStrategySnapshot,
    TimingMs,
)
from uae_law_rag.backend.db.repo import EvaluatorRepo, GenerationRepo, RetrievalRepo
from uae_law_rag.backend.schemas.audit import TraceContext
from uae_law_rag.backend.utils.errors import NotFoundError


router = APIRouter(prefix="/records", tags=["records"])  # docstring: records 路由前缀


def _coerce_hit_source(value: Any) -> HitSource:
    """
    [职责] 规范 hit source（keyword/vector/fused/reranked）。
    [边界] 非法值回退为 fused。
    [上游关系] retrieval 映射时调用。
    [下游关系] HitSummary.source。
    """
    raw = str(value or "").strip().lower()
    if raw in {"keyword", "vector", "fused", "reranked"}:
        return cast(HitSource, raw)  # docstring: 合法 source 直接返回
    return cast(HitSource, "fused")  # docstring: 非法值回退 fused


def _coerce_generation_status(value: Any) -> GenerationStatus:
    """
    [职责] 规范 generation status（success/partial/failed）。
    [边界] 非法值回退 failed。
    [上游关系] generation 映射时调用。
    [下游关系] GenerationRecordView.status。
    """
    raw = str(value or "").strip().lower()
    if raw in {"success", "partial", "failed"}:
        return cast(GenerationStatus, raw)  # docstring: 合法 status 直接返回
    return cast(GenerationStatus, "failed")  # docstring: 非法值回退 failed


def _coerce_evaluation_status(value: Any) -> EvaluationStatus:
    """
    [职责] 规范 evaluation status（pass/partial/fail/skipped）。
    [边界] 非法值回退 fail。
    [上游关系] evaluation 映射时调用。
    [下游关系] EvaluationRecordView.status。
    """
    raw = str(value or "").strip().lower()
    if raw in {"pass", "partial", "fail", "skipped"}:
        return cast(EvaluationStatus, raw)  # docstring: 合法 status 直接返回
    return cast(EvaluationStatus, "fail")  # docstring: 非法值回退 fail


def _coerce_check_status(value: Any) -> CheckStatus:
    """
    [职责] 规范 evaluation check status（pass/fail/warn/skipped）。
    [边界] 非法值回退 skipped。
    [上游关系] evaluation checks 映射时调用。
    [下游关系] EvaluationCheckSummary.status。
    """
    raw = str(value or "").strip().lower()
    if raw in {"pass", "fail", "warn", "skipped"}:
        return cast(CheckStatus, raw)  # docstring: 合法 status 直接返回
    return cast(CheckStatus, "skipped")  # docstring: 非法值回退 skipped


def _build_locator_from_hit(hit: Any) -> Dict[str, Any]:
    """
    [职责] 基于 RetrievalHit 组装 locator 结构。
    [边界] 仅读取 hit 快照字段；不访问 node 关系。
    [上游关系] retrieval 映射调用。
    [下游关系] HitSummary.locator。
    """
    locator: Dict[str, Any] = {}
    page = getattr(hit, "page", None)
    if page is not None:
        locator["page"] = page  # docstring: 页码定位
    start_offset = getattr(hit, "start_offset", None)
    if start_offset is not None:
        locator["start_offset"] = start_offset  # docstring: 起始偏移定位
    end_offset = getattr(hit, "end_offset", None)
    if end_offset is not None:
        locator["end_offset"] = end_offset  # docstring: 结束偏移定位
    return locator  # docstring: 返回 locator


def _normalize_citation_items(citations: Any) -> List[Dict[str, Any]]:
    """
    [职责] 从 citations payload 提取 citation items 列表。
    [边界] 仅处理 dict/list 结构；不校验 node_id 合法性。
    [上游关系] generation 记录映射调用。
    [下游关系] CitationView 列表构建。
    """
    if citations is None:
        return []  # docstring: 缺失 citations 返回空
    if isinstance(citations, dict):
        items = citations.get("items")
        if isinstance(items, list):
            return [dict(item) for item in items if isinstance(item, dict)]  # docstring: payload.items
        nodes = citations.get("nodes")
        if isinstance(nodes, list):
            return [{"node_id": node_id} for node_id in nodes]  # docstring: nodes 兜底为 items
        return []  # docstring: dict 无 items/nodes 回退
    if isinstance(citations, list):
        return [dict(item) for item in citations if isinstance(item, dict)]  # docstring: list 直接当作 items
    return []  # docstring: 其他类型回退空列表


def _build_citations(citations: Any) -> List[CitationView]:
    """
    [职责] 构造 Generation 引用列表（CitationView）。
    [边界] locator 仅做浅映射；不补全文。
    [上游关系] generation 记录映射调用。
    [下游关系] GenerationRecordView.citations。
    """
    items = _normalize_citation_items(citations)  # docstring: 解析 citation items
    views: List[CitationView] = []
    for item in items:
        node_id = str(item.get("node_id") or "").strip()
        if not node_id:
            continue  # docstring: 缺失 node_id 跳过
        locator = (
            dict(item.get("locator") or {}) if isinstance(item.get("locator"), dict) else {}
        )  # docstring: 归一化 locator
        page_value = item.get("page", None)
        if page_value is None:
            page_value = locator.get("page", None)
        article_id_value = item.get("article_id", None)
        if article_id_value is None:
            article_id_value = locator.get("article_id", None)
        section_path_value = item.get("section_path", None)
        if section_path_value is None:
            section_path_value = locator.get("section_path", None)

        views.append(
            CitationView(
                node_id=NodeId(node_id),
                rank=item.get("rank"),
                quote=str(item.get("quote") or ""),
                page=page_value,
                article_id=article_id_value,
                section_path=section_path_value,
            )
        )  # docstring: 映射 CitationView
    return views  # docstring: 返回 citations 视图


def _build_checks_summary(checks: Any) -> List[EvaluationCheckSummary]:
    """
    [职责] 构造评估检查摘要列表。
    [边界] 仅做字段映射；未知字段忽略。
    [上游关系] evaluation 记录映射调用。
    [下游关系] EvaluationRecordView.checks_summary。
    """
    items: List[Any] = []
    if isinstance(checks, dict):
        raw_items = checks.get("items")
        if isinstance(raw_items, list):
            items = raw_items  # docstring: 标准 checks.items
        elif checks:
            items = [checks]  # docstring: 非标准 dict 兜底为单项
    elif isinstance(checks, list):
        items = checks  # docstring: list 直接使用

    summaries: List[EvaluationCheckSummary] = []
    for item in items:
        if not isinstance(item, dict):
            continue  # docstring: 非 dict 跳过
        name = str(item.get("name") or item.get("rule_name") or item.get("rule_id") or "unknown")
        status = _coerce_check_status(item.get("status"))
        message = str(item.get("message") or item.get("reason") or "")
        summaries.append(
            EvaluationCheckSummary(
                name=name,
                status=status,
                message=message,
            )
        )  # docstring: 映射检查摘要
    return summaries  # docstring: 返回检查摘要列表


@router.get("/retrieval/{retrieval_record_id}", response_model=RetrievalRecordView)
async def get_retrieval_record(
    retrieval_record_id: str,
    session: AsyncSession = Depends(get_session),
    trace_context: TraceContext = Depends(get_trace_context),
) -> RetrievalRecordView:
    """
    [职责] 获取检索记录与命中摘要。
    [边界] 仅查询；不做检索重算。
    [上游关系] 前端回放请求。
    [下游关系] RetrievalRepo 读取 record/hits。
    """
    try:
        repo = RetrievalRepo(session)  # docstring: 装配 retrieval repo
        record = await repo.get_record(str(retrieval_record_id))  # docstring: 获取 record
        if record is None:
            raise NotFoundError(message="retrieval record not found")  # docstring: 记录不存在
        hits = await repo.list_hits(str(retrieval_record_id))  # docstring: 获取 hits
    except Exception as exc:
        return to_json_response(  # type: ignore[return-value]
            exc,
            trace_id=str(trace_context.trace_id),
            request_id=str(trace_context.request_id),
        )  # docstring: 异常映射为 ErrorResponse

    def _opt_int(v: Any) -> Optional[int]:
        if v is None:
            return None
        try:
            return int(v)
        except Exception:
            return None

    def _opt_str(v: Any) -> Optional[str]:
        if v is None:
            return None
        s = str(v).strip()
        return s or None

    strategy_snapshot = RetrievalStrategySnapshot(
        keyword_top_k=_opt_int(getattr(record, "keyword_top_k", None)),
        vector_top_k=_opt_int(getattr(record, "vector_top_k", None)),
        fusion_top_k=_opt_int(getattr(record, "fusion_top_k", None)),
        rerank_top_k=_opt_int(getattr(record, "rerank_top_k", None)),
        fusion_strategy=_opt_str(getattr(record, "fusion_strategy", None)),
        rerank_strategy=_opt_str(getattr(record, "rerank_strategy", None)),
        provider_snapshot=dict(getattr(record, "provider_snapshot", None) or {}),
    )  # docstring: 策略快照映射

    timing_ms = TimingMs.model_validate(record.timing_ms or {})  # docstring: timing_ms 映射

    hit_views: List[HitSummary] = []
    for hit in hits:
        hit_views.append(
            HitSummary(
                node_id=NodeId(str(hit.node_id)),
                source=_coerce_hit_source(getattr(hit, "source", None)),
                rank=int(hit.rank),
                score=float(getattr(hit, "score", 0.0) or 0.0),
                excerpt=str(hit.excerpt) if getattr(hit, "excerpt", None) else None,
                locator=_build_locator_from_hit(hit),
            )
        )  # docstring: 映射 HitSummary

    return RetrievalRecordView(
        retrieval_record_id=RetrievalRecordId(str(record.id)),
        message_id=MessageId(str(record.message_id)),
        kb_id=KnowledgeBaseId(str(record.kb_id)),
        query_text=str(record.query_text),
        strategy_snapshot=strategy_snapshot,
        timing_ms=timing_ms,
        hits=hit_views,
    )  # docstring: 返回检索记录视图


@router.get("/generation/{generation_record_id}", response_model=GenerationRecordView)
async def get_generation_record(
    generation_record_id: str,
    session: AsyncSession = Depends(get_session),
    trace_context: TraceContext = Depends(get_trace_context),
) -> GenerationRecordView:
    """
    [职责] 获取生成记录与引用摘要。
    [边界] 仅查询；不触发生成。
    [上游关系] 前端回放请求。
    [下游关系] GenerationRepo 读取 record。
    """
    try:
        repo = GenerationRepo(session)  # docstring: 装配 generation repo
        record = await repo.get_record(str(generation_record_id))  # docstring: 获取 record
        if record is None:
            raise NotFoundError(message="generation record not found")  # docstring: 记录不存在
    except Exception as exc:
        return to_json_response(  # type: ignore[return-value]
            exc,
            trace_id=str(trace_context.trace_id),
            request_id=str(trace_context.request_id),
        )  # docstring: 异常映射为 ErrorResponse

    return GenerationRecordView(
        generation_record_id=GenerationRecordId(str(record.id)),
        message_id=MessageId(str(record.message_id)),
        status=_coerce_generation_status(record.status),
        answer=str(record.output_raw) if record.output_raw is not None else None,
        citations=_build_citations(record.citations),
    )  # docstring: 返回生成记录视图


@router.get("/evaluation/{evaluation_record_id}", response_model=EvaluationRecordView)
async def get_evaluation_record(
    evaluation_record_id: str,
    session: AsyncSession = Depends(get_session),
    trace_context: TraceContext = Depends(get_trace_context),
) -> EvaluationRecordView:
    """
    [职责] 获取评估记录与检查摘要。
    [边界] 仅查询；不触发评估。
    [上游关系] 前端回放请求。
    [下游关系] EvaluatorRepo 读取 record。
    """
    try:
        repo = EvaluatorRepo(session)  # docstring: 装配 evaluator repo
        record = await repo.get_record(str(evaluation_record_id))  # docstring: 获取 record
        if record is None:
            raise NotFoundError(message="evaluation record not found")  # docstring: 记录不存在
    except Exception as exc:
        return to_json_response(  # type: ignore[return-value]
            exc,
            trace_id=str(trace_context.trace_id),
            request_id=str(trace_context.request_id),
        )  # docstring: 异常映射为 ErrorResponse

    return EvaluationRecordView(
        evaluation_record_id=EvaluationRecordId(str(record.id)),
        message_id=MessageId(str(record.message_id)),
        status=_coerce_evaluation_status(record.status),
        rule_version=str(record.rule_version),
        checks_summary=_build_checks_summary(record.checks),
    )  # docstring: 返回评估记录视图
