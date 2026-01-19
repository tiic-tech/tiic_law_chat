# src/uae_law_rag/backend/db/models/retrieval.py

from __future__ import annotations

import uuid
from typing import List, Optional, TYPE_CHECKING

from sqlalchemy import Float, ForeignKey, Integer, JSON, String, Text, UniqueConstraint
from sqlalchemy.orm import Mapped, mapped_column, relationship

from uae_law_rag.backend.utils.constants import MESSAGE_ID_KEY

from ..base import Base, TimestampMixin

if TYPE_CHECKING:
    from .message import MessageModel
    from .doc import KnowledgeBaseModel, NodeModel
    from .evaluator import EvaluationRecordModel


class RetrievalRecordModel(Base, TimestampMixin):
    """
    [职责] 检索记录：一次“关键词全量召回 + 向量检索 + 融合 + rerank”的可回放审计单元。
    [边界] 不存生成结果；生成进入 GenerationRecord。此表只记录 retrieval pipeline 的输入、参数与产物索引。
    [上游关系] ChatService 在执行 retrieval pipeline 后创建记录；MessageModel 可引用此记录。
    [下游关系] RetrievalHitModel（1-N）记录每个命中证据；GenerationRecord 通过 retrieval_record_id 引用证据集合。
    """

    __tablename__ = "retrieval_record"
    __table_args__ = (UniqueConstraint(MESSAGE_ID_KEY, name="uq_retrieval_record_message"),)

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
        comment="检索记录ID（UUID字符串）",  # docstring: 一次检索的唯一标识
    )

    message_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("message.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment="消息ID（可选外键）",  # docstring: 若先建 message 再检索可关联
    )

    kb_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("knowledge_base.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment="KB ID（外键）",  # docstring: 检索作用域
    )

    query_text: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        comment="检索查询文本",  # docstring: 用户问题/重写后的检索 query
    )

    keyword_top_k: Mapped[int] = mapped_column(
        Integer,
        default=200,
        nullable=False,
        comment="关键词召回 top_k（全量召回可设为大值）",  # docstring: keyword recall 配置
    )

    vector_top_k: Mapped[int] = mapped_column(
        Integer,
        default=50,
        nullable=False,
        comment="向量召回 top_k",  # docstring: vector recall 配置
    )

    fusion_top_k: Mapped[int] = mapped_column(
        Integer,
        default=50,
        nullable=False,
        comment="融合后截断 top_k",  # docstring: fusion 结果规模
    )

    rerank_top_k: Mapped[int] = mapped_column(
        Integer,
        default=10,
        nullable=False,
        comment="rerank 后保留 top_k",  # docstring: 最终 evidence 数量
    )

    fusion_strategy: Mapped[str] = mapped_column(
        String(64),
        default="union",
        nullable=False,
        comment="融合策略（union/rrf/weighted等）",  # docstring: fusion 算法标识
    )

    rerank_strategy: Mapped[str] = mapped_column(
        String(64),
        default="none",
        nullable=False,
        comment="rerank 策略（none/llm/model等）",  # docstring: rerank 算法标识
    )

    provider_snapshot: Mapped[dict] = mapped_column(
        JSON,
        default=dict,
        nullable=False,
        comment="provider/model 快照（embed/rerank等）",  # docstring: 回放与审计必需
    )

    timing_ms: Mapped[dict] = mapped_column(
        JSON,
        default=dict,
        nullable=False,
        comment="各阶段耗时（ms）：keyword/vector/fusion/rerank",  # docstring: 性能分析与 gate tests
    )

    # --- explicit ownership relationships (消费 Conversation/Message/KB) ---
    kb: Mapped["KnowledgeBaseModel"] = relationship(
        "KnowledgeBaseModel",
    )

    hits: Mapped[List["RetrievalHitModel"]] = relationship(
        "RetrievalHitModel",
        back_populates="record",
        cascade="all, delete-orphan",
        order_by="RetrievalHitModel.rank",
    )

    message: Mapped["MessageModel"] = relationship(
        "MessageModel",
        back_populates="retrieval_record",
    )

    evaluation_record: Mapped[Optional["EvaluationRecordModel"]] = relationship(
        "EvaluationRecordModel",
        back_populates="retrieval_record",
        uselist=False,
        cascade="all, delete-orphan",
    )  # docstring: 本次检索的评估记录（一对一）


class RetrievalHitModel(Base, TimestampMixin):
    """
    [职责] 检索命中：记录每个证据节点在不同阶段（keyword/vector/fusion/rerank）的排名与分数细节。
    [边界] 不做引用格式化；引用渲染在 generation/postprocess 或 API 层完成。
    [上游关系] Retrieval pipeline 产出 hit 列表并写入本表。
    [下游关系] Generation 依据 hit 的 node_id 构建上下文；Evaluator 可用 hit 计算覆盖率/可解释性。
    """

    __tablename__ = "retrieval_hit"

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
        comment="命中ID（UUID字符串）",  # docstring: 命中记录唯一标识
    )

    retrieval_record_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("retrieval_record.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment="检索记录ID（外键）",  # docstring: 命中归属某次检索
    )

    node_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("node.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment="命中文档节点ID（外键）",  # docstring: 证据引用的核心指针
    )

    source: Mapped[str] = mapped_column(
        String(32),
        nullable=False,
        default="fused",
        comment="命中来源（keyword/vector/fused/reranked）",  # docstring: hit 来源阶段
    )

    rank: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        comment="命中排名（从0或1开始；建议统一为1开始）",  # docstring: 展示与 rerank 结果顺序
    )

    score: Mapped[float] = mapped_column(
        Float,
        nullable=False,
        default=0.0,
        comment="综合分数（融合/重排后的最终分）",  # docstring: 用于排序的主分数
    )

    score_details: Mapped[dict] = mapped_column(
        JSON,
        default=dict,
        nullable=False,
        comment="分数细节（kw_score/vector_score/rrf/rerank等）",  # docstring: 白箱解释与调参依据
    )

    excerpt: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
        comment="命中摘要/片段（可选，便于UI与评估）",  # docstring: 可存截断文本，避免每次 join node.text
    )

    page: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
        comment="页码快照（可选，冗余自 Node）",  # docstring: 便于直接展示而不 join
    )

    start_offset: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
        comment="页内起始偏移快照（冗余自 Node）",  # docstring: 证据定位辅助
    )

    end_offset: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
        comment="页内结束偏移快照（冗余自 Node）",  # docstring: 证据定位辅助
    )

    article_id: Mapped[Optional[str]] = mapped_column(
        String(64),
        nullable=True,
        index=True,
        comment="法条/条款标识（可选，如 Article 12）",  # docstring: UAE 法律结构化定位
    )

    section_path: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
        comment="层级路径（可选，如 Chapter/Section）",  # docstring: 结构化导航与展示
    )

    record: Mapped["RetrievalRecordModel"] = relationship(
        "RetrievalRecordModel",
        back_populates="hits",
    )

    node: Mapped["NodeModel"] = relationship(
        "NodeModel",
    )
