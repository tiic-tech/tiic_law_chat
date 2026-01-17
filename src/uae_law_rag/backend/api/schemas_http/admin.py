# src/uae_law_rag/backend/api/schemas_http/admin.py

"""
[职责] HTTP Admin Schema：提供运营/审计用的 KB/File/Document 视图结构。
[边界] 不暴露内部 DB 全量字段；不承载业务编排或权限逻辑。
[上游关系] routers/admin.py 从 DB 聚合数据并映射为本模块结构。
[下游关系] 前端运营/审计页面展示与回放使用。
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from pydantic import BaseModel, ConfigDict, Field

from ._common import DocumentId, KnowledgeBaseId, KnowledgeFileId, UUIDStr
from .ingest import IngestStatus


class KBView(BaseModel):
    """
    [职责] KBView：知识库运营视图（基础配置与统计）。
    [边界] 不包含向量数据/索引细节；仅输出可审计配置摘要。
    [上游关系] admin router 从 KnowledgeBaseModel 映射。
    [下游关系] 运营后台展示与筛选。
    """

    model_config = ConfigDict(extra="forbid")  # docstring: 锁定对外字段

    id: UUIDStr = Field(...)  # docstring: DB 主键（KnowledgeBaseModel.id）
    kb_id: KnowledgeBaseId = Field(...)  # docstring: KB 业务ID（当前等于 kb_name，如 default）
    user_id: Optional[UUIDStr] = Field(default=None)  # docstring: 归属用户ID（可选）
    kb_name: str = Field(..., min_length=1, max_length=80)  # docstring: KB 名称
    kb_info: Optional[str] = Field(default=None, max_length=200)  # docstring: KB 简介

    vs_type: Optional[str] = Field(default=None, max_length=50)  # docstring: 向量库类型
    milvus_collection: Optional[str] = Field(default=None, max_length=128)  # docstring: Milvus collection
    milvus_partition: Optional[str] = Field(default=None, max_length=128)  # docstring: Milvus partition

    embed_provider: Optional[str] = Field(default=None, max_length=64)  # docstring: embedding provider
    embed_model: Optional[str] = Field(default=None, max_length=128)  # docstring: embedding model
    embed_dim: Optional[int] = Field(default=None, ge=1)  # docstring: embedding 维度

    rerank_provider: Optional[str] = Field(default=None, max_length=64)  # docstring: rerank provider
    rerank_model: Optional[str] = Field(default=None, max_length=128)  # docstring: rerank model

    chunking_config: Dict[str, Any] = Field(default_factory=dict)  # docstring: 切分配置快照
    file_count: Optional[int] = Field(default=None, ge=0)  # docstring: 文件数量统计


class FileView(BaseModel):
    """
    [职责] FileView：文件运营视图（导入状态与指纹信息）。
    [边界] 不包含节点/向量细节；仅提供审计所需摘要字段。
    [上游关系] admin router 从 KnowledgeFileModel 映射。
    [下游关系] 运营后台展示导入状态与统计。
    """

    model_config = ConfigDict(extra="forbid")  # docstring: 锁定对外字段

    file_id: KnowledgeFileId = Field(...)  # docstring: 文件ID
    kb_id: KnowledgeBaseId = Field(...)  # docstring: 所属 KB ID
    file_name: str = Field(..., min_length=1, max_length=255)  # docstring: 文件名
    file_ext: Optional[str] = Field(default=None, max_length=16)  # docstring: 文件扩展名
    source_uri: Optional[str] = Field(default=None, max_length=2048)  # docstring: 源 URI

    sha256: str = Field(..., min_length=64, max_length=64)  # docstring: 文件指纹
    file_version: int = Field(default=1, ge=1)  # docstring: 文件版本
    file_mtime: float = Field(default=0.0, ge=0.0)  # docstring: 文件 mtime（epoch）
    file_size: int = Field(default=0, ge=0)  # docstring: 文件大小（bytes）

    pages: Optional[int] = Field(default=None, ge=0)  # docstring: 页数（可选）
    ingest_profile: Dict[str, Any] = Field(default_factory=dict)  # docstring: 导入配置快照
    node_count: int = Field(default=0, ge=0)  # docstring: 节点数量
    ingest_status: IngestStatus = Field(...)  # docstring: 导入状态
    last_ingested_at: Optional[str] = Field(
        default=None
    )  # docstring: 最近导入完成时间（ISO）; 建议在 routers 映射时保证格式为 ISO 8601（最好带时区），并保持一致（例如 YYYY-MM-DDTHH:MM:SSZ），避免前端解析歧义


class DocumentView(BaseModel):
    """
    [职责] DocumentView：文档运营视图（文档基础信息与元数据摘要）。
    [边界] 不包含节点全文或向量；仅提供可回放元信息。
    [上游关系] admin router 从 DocumentModel 映射。
    [下游关系] 运营后台展示文档信息。
    """

    model_config = ConfigDict(extra="forbid")  # docstring: 锁定对外字段

    document_id: DocumentId = Field(...)  # docstring: 文档ID
    kb_id: KnowledgeBaseId = Field(...)  # docstring: 所属 KB ID
    file_id: KnowledgeFileId = Field(...)  # docstring: 来源文件ID

    title: Optional[str] = Field(default=None, max_length=255)  # docstring: 文档标题
    source_name: Optional[str] = Field(default=None, max_length=255)  # docstring: 来源展示名
    meta_data: Dict[str, Any] = Field(
        default_factory=dict
    )  # docstring: 文档元数据;建议后续 routers 映射时避免与 Python/Pydantic 内置 model_dump 等语义混淆；并确保该字段始终为 JSON-safe dict
    node_count: Optional[int] = Field(default=None, ge=0)  # docstring: 节点数量（可选）
