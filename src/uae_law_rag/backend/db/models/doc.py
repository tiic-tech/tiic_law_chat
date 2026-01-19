# src/uae_law_rag/backend/db/models/doc.py

from __future__ import annotations

import uuid
from datetime import datetime
from typing import List, Optional, TYPE_CHECKING

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    JSON,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from ..base import Base, TimestampMixin

if TYPE_CHECKING:
    from .user import UserModel


class KnowledgeBaseModel(Base, TimestampMixin):
    """
    [职责] 知识库实体：定义检索作用域与向量库（Milvus）绑定及 embedding/rerank 配置快照。
    [边界] 不直接存储向量；向量存 Milvus。DB 仅存“collection/index/profile/统计”等可回放信息。
    [上游关系] UserModel 创建 KB；Ingest 将文件与节点写入 KB；Chat 使用 KB 执行检索与生成。
    [下游关系] KnowledgeFileModel / DocumentModel / RetrievalRecordModel 通过 kb_id 归属；Milvus collection 由 kb 绑定。
    """

    __tablename__ = "knowledge_base"
    __table_args__ = (UniqueConstraint("user_id", "kb_name", name="uq_kb_user_name"),)

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
        comment="知识库ID（UUID字符串）",  # docstring: KB 全局唯一标识
    )

    user_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("user.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment="用户ID（外键）",  # docstring: KB 归属用户
    )

    kb_name: Mapped[str] = mapped_column(
        String(80),
        nullable=False,
        comment="知识库名称（同用户下唯一）",  # docstring: UI 展示与选择
    )

    kb_info: Mapped[Optional[str]] = mapped_column(
        String(200),
        nullable=True,
        comment="知识库简介（可选，用于Agent/提示词）",  # docstring: 简述 KB 内容与用途
    )

    vs_type: Mapped[str] = mapped_column(
        String(50),
        nullable=False,
        default="milvus",
        comment="向量库类型（当前固定 milvus）",  # docstring: 便于未来替换实现
    )

    milvus_collection: Mapped[str] = mapped_column(
        String(128),
        nullable=False,
        comment="Milvus collection 名称",  # docstring: 向量数据所在 collection
    )

    milvus_partition: Mapped[Optional[str]] = mapped_column(
        String(128),
        nullable=True,
        comment="Milvus partition（可选）",  # docstring: 多租户/分片时使用
    )

    embed_provider: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        default="ollama",
        comment="Embedding provider（回放用）",  # docstring: 记录 embedding 提供方
    )

    embed_model: Mapped[str] = mapped_column(
        String(128),
        nullable=False,
        comment="Embedding 模型名称",  # docstring: 记录 embedding 模型
    )

    embed_dim: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=1024,
        comment="Embedding 向量维度",  # docstring: Milvus schema 关键参数
    )

    rerank_provider: Mapped[Optional[str]] = mapped_column(
        String(64),
        nullable=True,
        comment="Rerank provider（可选）",  # docstring: 记录 rerank 提供方
    )

    rerank_model: Mapped[Optional[str]] = mapped_column(
        String(128),
        nullable=True,
        comment="Rerank 模型名称（可选）",  # docstring: 记录 rerank 模型
    )

    chunking_config: Mapped[dict] = mapped_column(
        JSON,
        default=dict,
        nullable=False,
        comment="切分配置快照（chunk_size/overlap/结构化规则）",  # docstring: 复现实验的最小配置
    )

    file_count: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
        comment="KB 内文件数量（冗余统计）",  # docstring: UI/统计用，可由 query 重算
    )

    user: Mapped["UserModel"] = relationship(
        "UserModel",
        back_populates="knowledge_bases",
    )

    files: Mapped[List["KnowledgeFileModel"]] = relationship(
        "KnowledgeFileModel",
        back_populates="kb",
        cascade="all, delete-orphan",
    )

    documents: Mapped[List["DocumentModel"]] = relationship(
        "DocumentModel",
        back_populates="kb",
        cascade="all, delete-orphan",
    )


class KnowledgeFileModel(Base, TimestampMixin):
    """
    [职责] 知识文件实体：记录源文件版本、指纹、解析/导入状态与统计。
    [边界] 不存储解析后的节点文本（在 NodeModel）；不存储向量（在 Milvus）。
    [上游关系] Ingest Pipeline 导入时创建/更新文件记录并写入指纹/页数/状态。
    [下游关系] DocumentModel/NodeModel 归属到 file；NodeVectorMapModel 通过 file/node 映射到 Milvus。
    """

    __tablename__ = "knowledge_file"
    __table_args__ = (UniqueConstraint("kb_id", "sha256", name="uq_file_kb_sha256"),)

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
        comment="文件ID（UUID字符串）",  # docstring: 文件记录唯一标识
    )

    kb_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("knowledge_base.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment="所属KB ID（外键）",  # docstring: 文件归属 KB
    )

    file_name: Mapped[str] = mapped_column(
        String(255),
        nullable=False,
        comment="文件名（展示用）",  # docstring: 原始文件名
    )

    file_ext: Mapped[Optional[str]] = mapped_column(
        String(16),
        nullable=True,
        comment="文件扩展名",  # docstring: pdf/txt 等
    )

    source_uri: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
        comment="源文件路径/URI（本地路径或对象存储URI）",  # docstring: 可追溯来源
    )

    sha256: Mapped[str] = mapped_column(
        String(64),
        nullable=False,
        index=True,
        comment="文件内容指纹（sha256）",  # docstring: 幂等导入与版本判断
    )

    file_version: Mapped[int] = mapped_column(
        Integer,
        default=1,
        nullable=False,
        comment="文件版本号（同名更新时递增）",  # docstring: 便于 UI 展示与回滚策略
    )

    file_mtime: Mapped[float] = mapped_column(
        Float,
        default=0.0,
        nullable=False,
        comment="文件修改时间（epoch seconds）",  # docstring: 与源文件对齐的 mtime
    )

    file_size: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
        comment="文件大小（bytes）",  # docstring: 基础统计
    )

    pages: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
        comment="页数（可选）",  # docstring: 证据定位与 UI 展示
    )

    ingest_profile: Mapped[dict] = mapped_column(
        JSON,
        default=dict,
        nullable=False,
        comment="导入配置快照（parser/segment/embed等）",  # docstring: 复现导入过程
    )

    node_count: Mapped[int] = mapped_column(
        Integer,
        default=0,
        nullable=False,
        comment="切分节点数量（冗余统计）",  # docstring: UI/统计用
    )

    ingest_status: Mapped[str] = mapped_column(
        String(32),
        default="pending",
        nullable=False,
        comment="导入状态（pending/success/failed）",  # docstring: gate tests 与任务状态
    )

    last_ingested_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime,
        nullable=True,
        comment="最近一次导入完成时间",  # docstring: 便于增量/重试策略
    )

    kb: Mapped["KnowledgeBaseModel"] = relationship(
        "KnowledgeBaseModel",
        back_populates="files",
    )

    documents: Mapped[List["DocumentModel"]] = relationship(
        "DocumentModel",
        back_populates="file",
        cascade="all, delete-orphan",
    )


class DocumentModel(Base, TimestampMixin):
    """
    [职责] 逻辑文档实体：承载“一个可检索的文档单元”（通常对应一个 PDF 文件的解析结果）。
    [边界] 不存储向量；不做检索记录；仅作为 Node 的父容器与元信息载体。
    [上游关系] Ingest Pipeline 在解析文件后创建 Document，并生成 Node。
    [下游关系] NodeModel 归属 Document；RetrievalHit 通过 node_id 间接定位到 document/file/page。
    """

    __tablename__ = "document"
    __table_args__ = (UniqueConstraint("kb_id", "file_id", name="uq_document_kb_file"),)

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
        comment="文档ID（UUID字符串）",  # docstring: 文档唯一标识
    )

    kb_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("knowledge_base.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment="所属KB ID（外键）",  # docstring: 文档归属 KB
    )

    file_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("knowledge_file.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment="源文件ID（外键）",  # docstring: 文档来源文件
    )

    title: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
        comment="文档标题（可选）",  # docstring: 从 PDF 元信息/首段推断
    )

    source_name: Mapped[Optional[str]] = mapped_column(
        String(255),
        nullable=True,
        comment="来源展示名（可选）",  # docstring: UI 展示友好名称
    )

    meta_data: Mapped[dict] = mapped_column(
        JSON,
        default=dict,
        nullable=False,
        comment="文档元数据（解析信息/法条结构摘要等）",  # docstring: 非关键字段的扩展区
    )

    kb: Mapped["KnowledgeBaseModel"] = relationship(
        "KnowledgeBaseModel",
        back_populates="documents",
    )

    file: Mapped["KnowledgeFileModel"] = relationship(
        "KnowledgeFileModel",
        back_populates="documents",
    )

    nodes: Mapped[List["NodeModel"]] = relationship(
        "NodeModel",
        back_populates="document",
        cascade="all, delete-orphan",
    )


class NodeModel(Base, TimestampMixin):
    """
    [职责] 节点（Chunk）实体：关键词全量召回与证据引用的最小单元（包含 page/offset 等定位信息）。
    [边界] 不存向量本体；向量在 Milvus。Node 仅存文本与结构化定位元信息。
    [上游关系] Ingest Pipeline 从 pymupdf4llm 解析结果切分生成 Node，并写入 DB + Milvus 映射。
    [下游关系] RetrievalHit 引用 node_id；Generation citations 引用 node_id；UI 可展示 node 文本与定位。
    """

    __tablename__ = "node"
    __table_args__ = (UniqueConstraint("document_id", "node_index", name="uq_node_document_index"),)

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
        comment="节点ID（UUID字符串）",  # docstring: node 全局唯一标识（证据引用主键）
    )

    document_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("document.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment="所属文档ID（外键）",  # docstring: 节点归属文档
    )

    node_index: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        comment="节点序号（文档内从0递增）",  # docstring: 保证稳定排序与可复现切分
    )

    text: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        comment="节点文本内容",  # docstring: keyword 全量召回与引用证据原文
    )

    page: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
        index=True,
        comment="页码（从1开始，可选）",  # docstring: 证据定位（PDF page）
    )

    start_offset: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
        comment="在文档中的起始偏移（全量）",  # docstring: 精准定位文本位置
    )

    end_offset: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
        comment="在文档中的结束偏移（全量）",  # docstring: 精准定位文本位置
    )

    page_start_offset: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
        comment="在页面中的起始偏移（页内）",  # docstring: 精准定位文本位置
    )

    page_end_offset: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
        comment="在页面中的结束偏移（页内）",  # docstring: 精准定位文本位置
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

    meta_data: Mapped[dict] = mapped_column(
        JSON,
        default=dict,
        nullable=False,
        comment="节点元数据（解析细节/置信度/标题等）",  # docstring: 扩展字段
    )

    document: Mapped["DocumentModel"] = relationship(
        "DocumentModel",
        back_populates="nodes",
    )

    vector_maps: Mapped[List["NodeVectorMapModel"]] = relationship(
        "NodeVectorMapModel",
        back_populates="node",
        cascade="all, delete-orphan",
    )


class NodeVectorMapModel(Base, TimestampMixin):
    """
    [职责] Node↔Milvus 映射：维护 node_id 与向量主键/collection 的对应关系，保证 upsert/回查一致。
    [边界] 不存 embedding；只存映射与必要元信息（可用于排错与一致性校验）。
    [上游关系] Ingest Pipeline 在写入 Milvus 后创建映射记录。
    [下游关系] Vector Retrieval 返回 vector_id 后可回查 node_id；一致性检查与重建索引依赖本表。
    """

    __tablename__ = "node_vector_map"
    __table_args__ = (UniqueConstraint("kb_id", "node_id", name="uq_map_kb_node"),)

    id: Mapped[str] = mapped_column(
        String(36),
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
        comment="映射ID（UUID字符串）",  # docstring: 映射记录唯一标识
    )

    kb_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("knowledge_base.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment="KB ID（外键）",  # docstring: 映射所属 KB
    )

    file_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("knowledge_file.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment="文件ID（外键）",  # docstring: 映射所属文件（便于批量操作）
    )

    node_id: Mapped[str] = mapped_column(
        String(36),
        ForeignKey("node.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
        comment="节点ID（外键）",  # docstring: 关联 NodeModel
    )

    vector_id: Mapped[str] = mapped_column(
        String(128),
        nullable=False,
        index=True,
        comment="Milvus 向量主键/ID",  # docstring: Milvus 插入后的主键（字符串化）
    )

    is_active: Mapped[bool] = mapped_column(
        Boolean,
        default=True,
        nullable=False,
        comment="映射是否有效（重建/删除时可置False）",  # docstring: 支持软失效与重建策略
    )

    meta_data: Mapped[dict] = mapped_column(
        JSON,
        default=dict,
        nullable=False,
        comment="映射元数据（collection/partition等冗余信息）",  # docstring: 排错与一致性校验
    )

    node: Mapped["NodeModel"] = relationship(
        "NodeModel",
        back_populates="vector_maps",
    )
