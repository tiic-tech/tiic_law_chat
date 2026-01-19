# src/uae_law_rag/backend/api/schemas_http/records.py

"""
[职责] HTTP Records 审计视图：定义检索/生成/评估记录的轻量输出结构，供前端回放与调试。
[边界] 不暴露 DB 全量字段；不包含 pipeline 执行逻辑；仅表达可审计的摘要结构。
[上游关系] services/routers 从 DB 或 pipeline record 构建视图对象。
[下游关系] 前端审计页与 debug 回放消费；与 debug.records 字段命名对齐。
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from ._common import (
    EvaluationRecordId,
    GenerationRecordId,
    KnowledgeBaseId,
    MessageId,
    NodeId,
    DocumentId,
    RetrievalRecordId,
)


HitSource = Literal["keyword", "vector", "fused", "reranked"]  # docstring: 命中来源（与 DB 语义一致）
GenerationStatus = Literal["success", "partial", "blocked", "failed"]  # docstring: 生成状态（审计输出）
EvaluationStatus = Literal["pass", "partial", "fail", "skipped"]  # docstring: 评估状态（审计输出）
CheckStatus = Literal["pass", "fail", "warn", "skipped"]  # docstring: 单条规则状态摘要


class TimingMs(BaseModel):
    """
    [职责] TimingMs：统一 timing_ms 容器（允许扩展）。
    [边界] 不强制字段集合；仅承载毫秒级数据。
    [上游关系] services/pipelines 传入 timing 统计。
    [下游关系] 前端审计页展示耗时分布。
    """

    model_config = ConfigDict(extra="allow")  # docstring: 允许扩展阶段字段

    total_ms: Optional[float] = Field(default=None, ge=0.0)  # docstring: 总耗时（ms）


class RetrievalStrategySnapshot(BaseModel):
    """
    [职责] RetrievalStrategySnapshot：检索策略快照（top_k/strategy/provider 摘要）。
    [边界] 不包含全文证据；仅保存策略与 provider 配置信息。
    [上游关系] retrieval pipeline/record 产出策略参数。
    [下游关系] 审计与复现实验参考。
    """

    model_config = ConfigDict(extra="allow")  # docstring: 允许扩展策略字段

    keyword_top_k: Optional[int] = Field(default=None, ge=1)  # docstring: 关键词召回上限
    vector_top_k: Optional[int] = Field(default=None, ge=1)  # docstring: 向量召回上限
    fusion_top_k: Optional[int] = Field(default=None, ge=1)  # docstring: 融合截断上限
    rerank_top_k: Optional[int] = Field(default=None, ge=1)  # docstring: rerank 截断上限

    fusion_strategy: Optional[str] = Field(default=None, min_length=1)  # docstring: 融合策略
    rerank_strategy: Optional[str] = Field(default=None, min_length=1)  # docstring: rerank 策略

    provider_snapshot: Dict[str, Any] = Field(default_factory=dict)  # docstring: provider 快照摘要


class HitSummary(BaseModel):
    """
    [职责] HitSummary：检索命中的最小摘要（用于审计与回放）。
    [边界] 不包含全文与敏感字段；仅保留证据指针与排序信息。
    [上游关系] RetrievalHit 记录摘要映射而来。
    [下游关系] 前端显示“证据命中概览”。
    """

    model_config = ConfigDict(extra="allow")  # docstring: 允许扩展定位字段

    node_id: NodeId = Field(...)  # docstring: 命中文档节点ID
    source: HitSource = Field(default="fused")  # docstring: 命中来源
    rank: int = Field(..., ge=0, le=100000)  # docstring: 命中排序
    score: float = Field(default=0.0)  # docstring: 命中分数

    excerpt: Optional[str] = Field(default=None)  # docstring: 证据片段（可选）
    locator: Dict[str, Any] = Field(default_factory=dict)  # docstring: 定位信息（page/article/offset 等）


class RetrievalRecordView(BaseModel):
    """
    [职责] RetrievalRecordView：检索记录审计视图（record + hits 概览）。
    [边界] 不暴露完整 hits/score 细节；仅输出摘要信息。
    [上游关系] retrieval_record + hits 列表映射。
    [下游关系] 前端审计页、debug 回放使用。
    """

    model_config = ConfigDict(extra="forbid")  # docstring: 锁定对外合同

    retrieval_record_id: RetrievalRecordId = Field(...)  # docstring: 检索记录ID
    message_id: MessageId = Field(...)  # docstring: 归属消息ID
    kb_id: KnowledgeBaseId = Field(...)  # docstring: 知识库ID
    query_text: str = Field(..., min_length=1, max_length=4096)  # docstring: 检索 query 文本

    strategy_snapshot: RetrievalStrategySnapshot = Field(
        default_factory=RetrievalStrategySnapshot
    )  # docstring: 策略快照摘要
    timing_ms: TimingMs = Field(default_factory=TimingMs)  # docstring: 耗时摘要（ms）
    hits: List[HitSummary] = Field(default_factory=list)  # docstring: 命中摘要列表

    # --- v1.1: staged hits for audit/eval (backward compatible) ---
    hits_by_source: Dict[str, List[HitSummary]] = Field(
        default_factory=dict
    )  # docstring: 按 source 分组的命中摘要（keyword/vector/fused/reranked）
    hit_counts: Dict[str, int] = Field(default_factory=dict)  # docstring: 每个 source 的命中数量


class CitationView(BaseModel):
    """
    [职责] CitationView：生成引用的可视化摘要（节点指针 + 定位信息）。
    [边界] 不包含全文；仅用于 UI 展示与回放定位。
    [上游关系] generation_record.citations 映射。
    [下游关系] 前端引用渲染组件。
    """

    model_config = ConfigDict(extra="allow")  # docstring: 允许扩展定位字段

    node_id: NodeId = Field(...)  # docstring: 引用节点ID
    rank: Optional[int] = Field(default=None, ge=0, le=100000)  # docstring: 引用顺序（可选）
    quote: str = Field(default="")  # docstring: 引用片段（可选）
    page: Optional[int] = Field(default=None)  # docstring: 页码定位（可选）
    article_id: Optional[str] = Field(default=None)  # docstring: 条款ID（可选）
    section_path: Optional[str] = Field(default=None)  # docstring: 章节路径（可选）


class GenerationRecordView(BaseModel):
    """
    [职责] GenerationRecordView：生成记录审计视图（状态 + answer + citations）。
    [边界] 不暴露 prompt/messages_snapshot；仅输出对外可解释摘要。
    [上游关系] generation_record 映射。
    [下游关系] 前端审计页与回放工具展示。
    """

    model_config = ConfigDict(extra="forbid")  # docstring: 锁定对外合同

    generation_record_id: GenerationRecordId = Field(...)  # docstring: 生成记录ID
    message_id: MessageId = Field(...)  # docstring: 归属消息ID
    status: GenerationStatus = Field(default="success")  # docstring: 生成状态
    answer: Optional[str] = Field(default=None)  # docstring: 生成答案（可选）
    citations: List[CitationView] = Field(default_factory=list)  # docstring: 引用摘要列表
    output_raw: Optional[str] = Field(default=None)  # docstring: 生成答案（可选）


class EvaluationCheckSummary(BaseModel):
    """
    [职责] EvaluationCheckSummary：评估规则检查摘要（名称 + 状态）。
    [边界] 不包含完整 metrics；仅输出最小可解释字段。
    [上游关系] evaluator checks 列表映射。
    [下游关系] 前端审计页展示规则结果。
    """

    model_config = ConfigDict(extra="allow")  # docstring: 允许扩展摘要字段

    name: str = Field(..., min_length=1, max_length=200)  # docstring: 规则名称
    status: CheckStatus = Field(...)  # docstring: 规则状态
    message: str = Field(default="")  # docstring: 简短说明（可选）


class EvaluationRecordView(BaseModel):
    """
    [职责] EvaluationRecordView：评估记录审计视图（总状态 + 规则摘要）。
    [边界] 不输出完整 config/scores；仅输出审计摘要字段。
    [上游关系] evaluation_record 映射。
    [下游关系] 前端审计页展示评估结论。
    """

    model_config = ConfigDict(extra="forbid")  # docstring: 锁定对外合同

    evaluation_record_id: EvaluationRecordId = Field(...)  # docstring: 评估记录ID
    message_id: MessageId = Field(...)  # docstring: 归属消息ID
    status: EvaluationStatus = Field(default="pass")  # docstring: 评估状态
    rule_version: str = Field(default="v0")  # docstring: 规则版本
    checks_summary: List[EvaluationCheckSummary] = Field(default_factory=list)  # docstring: 规则检查摘要列表


class NodeRecordView(BaseModel):
    """
    [职责] NodeRecordView：Node 回放视图（供 EvidencePanel / NodePreview 使用）。
    [边界] 默认只返回 excerpt（受 max_chars 控制）；不返回全文以避免 payload 失控。
    """

    model_config = ConfigDict(extra="forbid")

    node_id: NodeId = Field(...)
    kb_id: Optional[KnowledgeBaseId] = Field(default=None)

    document_id: DocumentId = Field(...)
    node_index: int = Field(..., ge=0)

    page: Optional[int] = Field(default=None, ge=1)
    start_offset: Optional[int] = Field(default=None, ge=0)
    end_offset: Optional[int] = Field(default=None, ge=0)

    page_start_offset: Optional[int] = Field(default=None, ge=0)
    page_end_offset: Optional[int] = Field(default=None, ge=0)

    article_id: Optional[str] = Field(default=None)
    section_path: Optional[str] = Field(default=None)

    text_excerpt: str = Field(..., description="Truncated node text excerpt")
    text_len: int = Field(..., ge=0)

    meta: Dict[str, Any] = Field(default_factory=dict)
