# src/uae_law_rag/backend/api/schemas_http/ingest.py

"""
[职责] HTTP Ingest Schema：定义导入接口的输入/输出合同（IngestRequest/IngestResponse）。
[边界] 仅表达 HTTP 语义；不包含 pipeline 实现；不负责 trace 注入与错误映射。
[上游关系] 前端/外部调用方提交 ingest 请求。
[下游关系] routers/ingest.py 调用 ingest_service 并映射到本模块输出结构。
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from ._common import DebugEnvelope, KnowledgeBaseId, KnowledgeFileId, DocumentId


IngestStatus = Literal["pending", "success", "failed"]  # docstring: 导入状态（对外合同）


class IngestProfile(BaseModel):
    """
    [职责] IngestProfile：导入策略快照（parser/parse_version/segment_version）。
    [边界] 仅描述可覆盖字段；不包含运行时配置解析。
    [上游关系] IngestRequest.ingest_profile。
    [下游关系] ingest_service 将其归一化并写入 ingest_profile 快照。
    """

    model_config = ConfigDict(extra="allow")  # docstring: 允许未来扩展策略字段

    parser: Optional[str] = Field(default=None)  # docstring: 解析器名称（可选覆盖）
    parse_version: Optional[str] = Field(default=None)  # docstring: 解析版本（可选覆盖）
    segment_version: Optional[str] = Field(default=None)  # docstring: 切分版本（可选覆盖）


class IngestRequest(BaseModel):
    """
    [职责] IngestRequest：导入请求输入结构（kb + 文件信息 + 可选策略）。
    [边界] 不承载文件上传内容；仅承载 source_uri 引用与策略参数。
    [上游关系] 前端/外部系统触发 ingest。
    [下游关系] routers/ingest.py 校验后调用 ingest_service.ingest_file。
    """

    model_config = ConfigDict(extra="forbid")  # docstring: 锁定对外输入合同

    kb_id: KnowledgeBaseId = Field(...)  # docstring: 知识库ID
    file_name: str = Field(..., min_length=1, max_length=512)  # docstring: 文件名（展示/审计）
    source_uri: str = Field(..., min_length=1, max_length=2048)  # docstring: 文件 URI 或 路径

    ingest_profile: Optional[IngestProfile] = Field(default=None)  # docstring: 可选策略覆盖
    dry_run: bool = Field(default=False)  # docstring: 仅校验/不落库（保留字段）


class IngestTimingMs(BaseModel):
    """
    [职责] IngestTimingMs：导入耗时统计容器（ms）。
    [边界] 不强制具体阶段字段；仅承载时间结构。
    [上游关系] ingest_service 返回 timing_ms。
    [下游关系] 前端/审计页面展示耗时分布。
    """

    model_config = ConfigDict(extra="allow")  # docstring: 允许扩展阶段字段

    total_ms: Optional[float] = Field(default=None, ge=0.0)  # docstring: 总耗时（ms）


class IngestResponse(BaseModel):
    """
    [职责] IngestResponse：导入响应输出结构（状态 + 数量 + timing + debug）。
    [边界] 不返回内部 DB 全量字段；仅输出前端可消费的摘要。
    [上游关系] ingest_service 返回结果经 routers 映射到本结构。
    [下游关系] 前端展示导入状态与结果。
    """

    model_config = ConfigDict(extra="forbid")  # docstring: 锁定对外输出合同

    kb_id: KnowledgeBaseId = Field(...)  # docstring: 知识库ID
    file_id: KnowledgeFileId = Field(...)  # docstring: 导入文件ID
    file_name: str = Field(..., min_length=1, max_length=512)  # docstring: 文件名（展示/审计）
    document_id: Optional[DocumentId] = Field(default=None)  # docstring: 导入产生的文档ID（可选）
    status: IngestStatus = Field(...)  # docstring: 导入状态
    node_count: int = Field(..., ge=0)  # docstring: 节点数量

    timing_ms: IngestTimingMs = Field(default_factory=IngestTimingMs)  # docstring: 耗时统计
    debug: Optional[DebugEnvelope] = Field(default=None)  # docstring: debug 输出（仅 debug=true）
