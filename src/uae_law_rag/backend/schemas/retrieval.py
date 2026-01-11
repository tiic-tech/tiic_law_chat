# src/uae_law_rag/backend/schemas/retrieval.py

"""
[职责] Retrieval 契约层：定义检索阶段的结构化记录（Record/Hit），用于 pipelines 与 DB 写入/读出的一致化对齐。
[边界] 不包含具体检索实现（keyword/vector/fusion/rerank）；不依赖 ORM；仅表达可回放的参数快照与结果快照。
[上游关系] chat 请求（Message）触发 retrieval pipeline；keyword(FTS) 与 vector(Milvus) 返回候选集合。
[下游关系] retrieval_repo 写入 RetrievalRecordModel / RetrievalHitModel；generation pipeline 消费 hits 作为 evidence/citations。
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from .ids import KnowledgeBaseId, MessageId, NodeId, RetrievalRecordId, VectorId


# --- Enums / literals ---

RetrievalStage = Literal["keyword", "vector", "fusion", "rerank"]  # docstring: 命中来源阶段（用于审计与调试）
FusionStrategy = Literal["union", "interleave", "weighted"]  # docstring: 融合策略（MVP 可先用 union）
RerankStrategy = Literal["none", "llm", "bge_reranker"]  # docstring: rerank 策略（MVP 可先 none）


class RetrievalTiming(BaseModel):
    """
    [职责] Retrieval 时延快照：记录各阶段耗时，用于性能分析与回放。
    [边界] 不做 profiling；只存储毫秒级粗粒度信息。
    [上游关系] retrieval pipeline 在每阶段结束时写入 timing。
    [下游关系] audit/监控/回归测试可用该结构做基线对比。
    """

    model_config = ConfigDict(extra="allow")

    total_ms: Optional[float] = Field(default=None)  # docstring: retrieval 总耗时（ms）
    keyword_ms: Optional[float] = Field(default=None)  # docstring: keyword(FTS) 阶段耗时（ms）
    vector_ms: Optional[float] = Field(default=None)  # docstring: vector(Milvus) 阶段耗时（ms）
    fusion_ms: Optional[float] = Field(default=None)  # docstring: fusion 阶段耗时（ms）
    rerank_ms: Optional[float] = Field(default=None)  # docstring: rerank 阶段耗时（ms）


class RetrievalRecord(BaseModel):
    """
    [职责] RetrievalRecord：一次检索的可回放“参数快照 + 策略快照 + provider 快照”。
    [边界] 不包含 hits 列表（hits 为单独实体）；不包含生成回答；仅描述“怎么检索”。
    [上游关系] Message 触发 retrieval pipeline；KB 作用域决定检索范围。
    [下游关系] 写入 RetrievalRecordModel；后续 GenerationRecord 引用 retrieval_record_id。
    """

    model_config = ConfigDict(extra="forbid")

    id: RetrievalRecordId = Field(...)  # docstring: RetrievalRecord 唯一 ID（UUID str）
    message_id: MessageId = Field(...)  # docstring: 归属消息（一致化：1 message ↔ 1 retrieval_record）
    kb_id: KnowledgeBaseId = Field(...)  # docstring: 检索作用域 KB

    query_text: str = Field(..., min_length=1, max_length=4096)  # docstring: 本次检索输入 query（通常=Message.query）

    keyword_top_k: int = Field(default=200, ge=1, le=5000)  # docstring: keyword 全量召回上限（候选数）
    vector_top_k: int = Field(default=200, ge=1, le=5000)  # docstring: vector 召回上限（候选数）
    fusion_top_k: int = Field(default=200, ge=1, le=5000)  # docstring: fusion 后候选上限
    rerank_top_k: int = Field(default=50, ge=1, le=5000)  # docstring: rerank 输出上限（最终 hits 数）

    fusion_strategy: FusionStrategy = Field(default="union")  # docstring: 融合策略
    rerank_strategy: RerankStrategy = Field(default="none")  # docstring: rerank 策略

    provider_snapshot: Dict[str, Any] = Field(default_factory=dict)  # docstring: provider/模型/版本等快照（用于可回放）
    timing: RetrievalTiming = Field(default_factory=RetrievalTiming)  # docstring: 时延快照（各阶段耗时）


class RetrievalHit(BaseModel):
    """
    [职责] RetrievalHit：一次检索的单条命中结果（证据条目），可用于回查 SQL Node 与 UI 展示。
    [边界] 不包含全文（全文在 SQL NodeModel.text）；此处仅存 node_id、可选 snippet 与 score 等轻量信息。
    [上游关系] keyword/vector/fusion/rerank 输出的候选集合经过去重/排序后形成 hits。
    [下游关系] 写入 RetrievalHitModel；generation 使用 hits 组装上下文与 citations。
    """

    model_config = ConfigDict(extra="forbid")

    retrieval_record_id: RetrievalRecordId = Field(...)  # docstring: 归属 RetrievalRecord
    rank: int = Field(..., ge=0, le=100000)  # docstring: 在最终列表中的排序位置（0-based）

    stage: RetrievalStage = Field(...)  # docstring: 命中来源阶段（keyword/vector/fusion/rerank）
    node_id: NodeId = Field(...)  # docstring: 对应 SQL NodeModel.id（证据回查主键）

    vector_id: Optional[VectorId] = Field(default=None)  # docstring: 命中对应的 Milvus vector_id（若来自向量侧）
    score: float = Field(default=0.0)  # docstring: 相似度/相关性分数（来源不同，语义以 stage 解释）

    snippet: str = Field(default="")  # docstring: 关键词命中片段或摘要（可空）
    meta: Dict[str, Any] = Field(default_factory=dict)  # docstring: 轻量元信息（kb/file/doc/page/article/section 等）


class RetrievalBundle(BaseModel):
    """
    [职责] RetrievalBundle：pipeline 内部传输对象（record + hits），便于 services/generation 作为单元消费。
    [边界] 仅用于内部编排；对外 HTTP 可单独映射为更精简结构。
    [上游关系] retrieval pipeline 产出 record 与 hits。
    [下游关系] generation pipeline 消费 bundle 生成回答，并写入 GenerationRecord。
    """

    model_config = ConfigDict(extra="forbid")

    record: RetrievalRecord = Field(...)  # docstring: 检索记录（参数/策略快照）
    hits: List[RetrievalHit] = Field(default_factory=list)  # docstring: 最终证据命中列表（已排序）
