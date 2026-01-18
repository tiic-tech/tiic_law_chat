# src/uae_law_rag/backend/api/schemas_http/evaluator.py

"""
[职责] evaluator http schemas：对外提供可审计、可前端展示的评估视图合同（P0/P1）。
[边界] 仅定义 schema；不包含数据库查询与业务逻辑。
[上游关系] routers/evaluator.py 响应模型。
[下游关系] 前端 EvaluatorPanel / debug 回放。
"""

from __future__ import annotations

from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from uae_law_rag.backend.schemas.ids import KnowledgeBaseId  # type: ignore


__all__ = [
    "QueryAnalysisView",
    "QueryPlanRequest",
    "QueryPlanResponse",
    "KeywordRecallMetricView",
    "KeywordRecallEvalRequest",
    "KeywordRecallEvalView",
]


class QueryAnalysisView(BaseModel):
    """
    [职责] QueryAnalysisView：面向前端的 query 分析结果（v0：仅透传）。
    [边界] v0 不做自动抽词/改写；M2 再扩展 enhanced_queries。
    """

    model_config = ConfigDict(extra="forbid")

    raw_query: str = Field(..., min_length=1, max_length=4096)
    keywords_list: List[str] = Field(default_factory=list)
    enhanced_queries: List[str] = Field(default_factory=list)


class QueryPlanRequest(BaseModel):
    """
    [职责] QueryPlanRequest：query_plan 入参（规则版 v1，后续可扩展 locale/domain）。
    [边界] 只读；不依赖 DB；不触发 retrieval/generation。
    """

    model_config = ConfigDict(extra="forbid")

    kb_id: KnowledgeBaseId = Field(...)
    raw_query: str = Field(..., min_length=1, max_length=4096)


class QueryPlanResponse(BaseModel):
    """
    [职责] QueryPlanResponse：query_plan 输出（面向前端/debug 的稳定合同）。
    [边界] analysis 对齐 QueryAnalysisView；meta 承载策略与审计信息。
    """

    model_config = ConfigDict(extra="forbid")

    kb_id: KnowledgeBaseId = Field(...)
    analysis: QueryAnalysisView = Field(...)
    meta: Dict[str, object] = Field(default_factory=dict)


class KeywordRecallMetricView(BaseModel):
    """
    [职责] KeywordRecallMetricView：单个 keyword 的召回度量（以“全文片段出现”为 GT）。
    [边界] GT 以 substring 匹配定义（对齐“前端展示法条原文片段”的产品目标）。
    """

    model_config = ConfigDict(extra="forbid")

    keyword: str = Field(..., min_length=1, max_length=256)
    gt_mode: Literal["substring"] = Field(default="substring")

    keyword_top_k: int = Field(..., ge=1, le=5000)

    gt_total: int = Field(..., ge=0)
    kw_total: int = Field(..., ge=0)
    overlap: int = Field(..., ge=0)

    # docstring: gt_total==0 时，recall/precision 建议置 None 并由前端显示为 "N/A"
    recall: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    precision: Optional[float] = Field(default=None, ge=0.0, le=1.0)

    # docstring: kw_total 达到 top_k 且 gt_total>kw_total 时，提示可能被截断
    capped: bool = Field(default=False)

    # docstring: 用于前端解释“漏召回/额外召回”的抽样（不要一次性返回全量）
    missing_sample: List[str] = Field(default_factory=list)
    extra_sample: List[str] = Field(default_factory=list)


class KeywordRecallEvalRequest(BaseModel):
    """
    [职责] KeywordRecallEvalRequest：keyword recall evaluator 入参（为前端服务）。
    [边界] v0 要求前端/脚本显式给 keywords_list；M2 再做自动抽词与改写。
    """

    model_config = ConfigDict(extra="forbid")

    kb_id: KnowledgeBaseId = Field(...)
    raw_query: str = Field(..., min_length=1, max_length=4096)
    keywords_list: List[str] = Field(default_factory=list, description="keywords to evaluate")

    keyword_top_k: Optional[int] = Field(default=None, ge=1, le=5000)
    allow_fallback: bool = Field(default=True)

    case_sensitive: bool = Field(default=False)
    sample_n: int = Field(default=20, ge=0, le=200)


class KeywordRecallEvalView(BaseModel):
    """
    [职责] KeywordRecallEvalView：面向前端的 keyword recall 评估结果视图。
    [边界] 不包含 nodes 全量；只返回 metrics 与少量 sample node_ids。
    """

    model_config = ConfigDict(extra="forbid")

    kb_id: KnowledgeBaseId = Field(...)
    analysis: QueryAnalysisView = Field(...)

    metrics: List[KeywordRecallMetricView] = Field(default_factory=list)

    timing_ms: Dict[str, Optional[float]] = Field(default_factory=dict)
    meta: Dict[str, object] = Field(default_factory=dict)
