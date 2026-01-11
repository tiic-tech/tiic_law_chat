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

from .ids import KnowledgeBaseId, MessageId, NodeId, RetrievalRecordId, RetrievalHitId


# --- Enums / literals ---

HitSource = Literal["keyword", "vector", "fused", "reranked"]  # docstring: 对齐 DB RetrievalHitModel.source
FusionStrategy = Literal["union", "rrf", "weighted", "interleave"]  # docstring: 可扩展，但先覆盖 DB 可能值
RerankStrategy = Literal["none", "llm", "bge_reranker"]  # docstring: MVP 先 none


class RetrievalTimingMs(BaseModel):
    """
    [职责] Retrieval timing_ms：对齐 DB 的 timing_ms JSON，记录各阶段耗时（ms）。
    [边界] 只存粗粒度毫秒；不做 profiling。
    [上游关系] pipelines/retrieval 在每阶段结束时写入 timing_ms。
    [下游关系] DB 落库到 RetrievalRecordModel.timing_ms；监控与回归测试可用。
    """

    model_config = ConfigDict(extra="allow")

    total: Optional[float] = Field(default=None)  # docstring: 总耗时（ms）
    keyword: Optional[float] = Field(default=None)  # docstring: keyword 阶段耗时（ms）
    vector: Optional[float] = Field(default=None)  # docstring: vector 阶段耗时（ms）
    fusion: Optional[float] = Field(default=None)  # docstring: fusion 阶段耗时（ms）
    rerank: Optional[float] = Field(default=None)  # docstring: rerank 阶段耗时（ms）


class RetrievalRecord(BaseModel):
    """
    [职责] RetrievalRecord：一次检索的参数/策略/provider/timing 快照（对齐 DB 可落库形态）。
    [边界] 不包含 hits 列表；hits 为独立实体。
    [上游关系] Message/KB 触发 retrieval pipeline。
    [下游关系] 写入 RetrievalRecordModel；GenerationRecord 引用 retrieval_record_id。
    """

    model_config = ConfigDict(extra="forbid")

    id: RetrievalRecordId = Field(...)
    message_id: MessageId = Field(...)
    kb_id: KnowledgeBaseId = Field(...)

    query_text: str = Field(..., min_length=1, max_length=4096)

    # --- defaults MUST match DB models ---
    keyword_top_k: int = Field(default=200, ge=1, le=5000)
    vector_top_k: int = Field(default=50, ge=1, le=5000)
    fusion_top_k: int = Field(default=50, ge=1, le=5000)
    rerank_top_k: int = Field(default=10, ge=1, le=5000)

    fusion_strategy: FusionStrategy = Field(default="union")
    rerank_strategy: RerankStrategy = Field(default="none")

    provider_snapshot: Dict[str, Any] = Field(default_factory=dict)
    timing_ms: RetrievalTimingMs = Field(default_factory=RetrievalTimingMs)  # docstring: 对齐 DB timing_ms


class RetrievalHit(BaseModel):
    """
    [职责] RetrievalHit：一次检索的单条命中（对齐 DB RetrievalHitModel 可落库形态）。
    [边界] 不要求携带全文；必要证据指针以 node_id + page/offset 快照表达。
    [上游关系] keyword/vector/fusion/rerank 输出候选集合。
    [下游关系] 写入 RetrievalHitModel；generation/evaluator 消费该结构做 evidence/citations。
    """

    model_config = ConfigDict(extra="forbid")

    # 建议补齐 id（读库/审计场景有用；写入时可不填）
    id: Optional[RetrievalHitId] = Field(default=None)  # docstring: 命中ID（DB 生成）

    retrieval_record_id: RetrievalRecordId = Field(...)
    node_id: NodeId = Field(...)

    source: HitSource = Field(default="fused")  # docstring: 对齐 DB source
    rank: int = Field(..., ge=0, le=100000)
    score: float = Field(default=0.0)

    score_details: Dict[str, Any] = Field(default_factory=dict)  # docstring: 对齐 DB score_details JSON
    excerpt: Optional[str] = Field(default=None)  # docstring: 对齐 DB excerpt

    page: Optional[int] = Field(default=None)  # docstring: 证据定位快照
    start_offset: Optional[int] = Field(default=None)  # docstring: 证据定位快照
    end_offset: Optional[int] = Field(default=None)  # docstring: 证据定位快照


class RetrievalBundle(BaseModel):
    """
    [职责] RetrievalBundle：pipeline 内部传输对象（record + hits）。
    [边界] 仅内部使用；HTTP 层可映射为更轻结构。
    [上游关系] retrieval pipeline 产出。
    [下游关系] generation pipeline 消费。
    """

    model_config = ConfigDict(extra="forbid")

    record: RetrievalRecord = Field(...)
    hits: List[RetrievalHit] = Field(default_factory=list)
