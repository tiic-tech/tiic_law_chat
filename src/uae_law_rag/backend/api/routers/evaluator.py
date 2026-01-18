# src/uae_law_rag/backend/api/routers/evaluator.py

"""
[职责] evaluator routers：提供面向前端的评估接口（keyword recall v0）。
[边界] 不写入 DB；仅做只读统计与返回；异常使用统一 error envelope。
[上游关系] 前端 EvaluatorPanel / playground gate。
[下游关系] backend/pipelines/evaluator/keyword_recall.py 执行业务逻辑。
"""

from __future__ import annotations


from typing import Dict
from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from uae_law_rag.backend.api.deps import get_session, get_trace_context  # type: ignore
from uae_law_rag.backend.api.errors import to_json_response  # type: ignore
from uae_law_rag.backend.api.schemas_http.evaluator import (
    KeywordRecallAutoRequest,
    KeywordRecallEvalRequest,
    KeywordRecallEvalView,
    QueryAnalysisView,
    QueryPlanRequest,
    QueryPlanResponse,
)  # type: ignore
from uae_law_rag.backend.schemas.audit import TraceContext  # type: ignore
from uae_law_rag.backend.pipelines.evaluator.keyword_recall import (
    evaluate_keyword_recall,
)  # type: ignore
from uae_law_rag.backend.pipelines.evaluator.query_plan import build_query_plan  # type: ignore


router = APIRouter(prefix="/evaluator", tags=["evaluator"])  # docstring: evaluator 路由前缀


@router.post("/query_plan", response_model=QueryPlanResponse)
async def query_plan(
    payload: QueryPlanRequest,
    trace_context: TraceContext = Depends(get_trace_context),
) -> QueryPlanResponse:
    """
    [职责] 生成 QueryPlan（规则版 v1），用于 debug / evaluator 上游。
    [边界] 只读；不写 DB；不依赖 session；不触发 retrieval/generation。
    """
    try:
        plan = build_query_plan(
            raw_query=str(payload.raw_query),
            kb_id=str(payload.kb_id),
        )

        analysis = QueryAnalysisView(
            raw_query=plan.raw_query,
            keywords_list=list(plan.keywords_list),
            enhanced_queries=list(plan.enhanced_queries),
        )

        return QueryPlanResponse(
            kb_id=payload.kb_id,
            analysis=analysis,
            meta=dict(plan.meta),
        )
    except Exception as exc:
        return to_json_response(  # type: ignore[return-value]
            exc,
            trace_id=str(getattr(trace_context, "trace_id", "") or ""),
            request_id=str(getattr(trace_context, "request_id", "") or ""),
        )


@router.post("/keyword_recall", response_model=KeywordRecallEvalView)
async def keyword_recall_evaluator(
    payload: KeywordRecallEvalRequest,
    session: AsyncSession = Depends(get_session),
    trace_context: TraceContext = Depends(get_trace_context),
) -> KeywordRecallEvalView:
    """
    [职责] 评估 keywords_list 的 keyword recall（GT=全文 substring 出现片段）。
    [边界] 只读计算；不写库；不触发 generation；仅依赖 node 表与 keyword_recall。
    """
    try:
        result = await evaluate_keyword_recall(
            session=session,
            kb_id=str(payload.kb_id),
            raw_query=str(payload.raw_query),
            keywords=list(payload.keywords_list or []),
            keyword_top_k=(int(payload.keyword_top_k) if payload.keyword_top_k is not None else None),
            allow_fallback=bool(payload.allow_fallback),
            case_sensitive=bool(payload.case_sensitive),
            sample_n=int(payload.sample_n),
            trace_id=str(getattr(trace_context, "trace_id", "") or ""),
            request_id=str(getattr(trace_context, "request_id", "") or ""),
        )

        analysis = QueryAnalysisView(
            raw_query=str(payload.raw_query),
            keywords_list=list(payload.keywords_list or []),
            enhanced_queries=[],
        )

        return KeywordRecallEvalView(
            kb_id=payload.kb_id,
            analysis=analysis,
            metrics=result["metrics"],
            timing_ms=result["timing_ms"],
            meta=result.get("meta") or {},
        )
    except Exception as exc:
        return to_json_response(  # type: ignore[return-value]
            exc,
            trace_id=str(getattr(trace_context, "trace_id", "") or ""),
            request_id=str(getattr(trace_context, "request_id", "") or ""),
        )


@router.post("/keyword_recall_auto", response_model=KeywordRecallEvalView)
async def keyword_recall_evaluator_auto(
    payload: KeywordRecallAutoRequest,
    session: AsyncSession = Depends(get_session),
    trace_context: TraceContext = Depends(get_trace_context),
) -> KeywordRecallEvalView:
    """
    [职责] 自动从 raw_query 生成 keywords_list，并执行 keyword recall evaluator。
    [边界] 只读计算；不写库；不触发 generation；仅依赖 node 表与 keyword_recall。
    """
    try:
        plan = build_query_plan(
            raw_query=str(payload.raw_query),
            kb_id=str(payload.kb_id),
        )

        result = await evaluate_keyword_recall(
            session=session,
            kb_id=str(payload.kb_id),
            raw_query=str(payload.raw_query),
            keywords=list(plan.keywords_list),
            keyword_top_k=(int(payload.keyword_top_k) if payload.keyword_top_k is not None else None),
            allow_fallback=bool(payload.allow_fallback),
            case_sensitive=bool(payload.case_sensitive),
            sample_n=int(payload.sample_n),
            trace_id=str(getattr(trace_context, "trace_id", "") or ""),
            request_id=str(getattr(trace_context, "request_id", "") or ""),
        )

        analysis = QueryAnalysisView(
            raw_query=str(payload.raw_query),
            keywords_list=list(plan.keywords_list),
            enhanced_queries=list(plan.enhanced_queries),
        )

        # docstring: meta 合并（plan meta 优先级低于 evaluator meta，避免覆盖 trace/request）
        merged_meta: Dict[str, object] = {}
        merged_meta.update(dict(plan.meta or {}))
        merged_meta.update(dict(result.get("meta") or {}))

        return KeywordRecallEvalView(
            kb_id=payload.kb_id,
            analysis=analysis,
            metrics=result["metrics"],
            timing_ms=result["timing_ms"],
            meta=merged_meta,
        )
    except Exception as exc:
        return to_json_response(  # type: ignore[return-value]
            exc,
            trace_id=str(getattr(trace_context, "trace_id", "") or ""),
            request_id=str(getattr(trace_context, "request_id", "") or ""),
        )
