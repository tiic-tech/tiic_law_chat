# src/uae_law_rag/backend/api/schemas_http/_common.py

"""
[职责] HTTP Schema 公共组件：定义 ErrorResponse、DebugEnvelope 与通用 ID 类型，作为 API 契约基础。
[边界] 仅描述 HTTP 输入/输出结构；不负责 trace 注入、异常映射或业务逻辑。
[上游关系] api/middleware 注入 trace/request；api/errors 将 DomainError 映射到 ErrorResponse。
[下游关系] api/schemas_http/{chat,ingest,records,admin} 复用本模块结构。
"""

from __future__ import annotations

from typing import Any, Dict, Literal, NewType, Optional

from pydantic import BaseModel, ConfigDict, Field


UUIDStr = NewType("UUIDStr", str)  # docstring: 通用 UUID 字符串类型（运行时仍为 str）

TraceId = UUIDStr  # docstring: trace_id（跨请求链路）
RequestId = UUIDStr  # docstring: request_id（单次请求）
ConversationId = UUIDStr  # docstring: conversation_id（会话ID）
MessageId = UUIDStr  # docstring: message_id（消息ID）
KnowledgeBaseId = NewType("KnowledgeBaseId", str)  # docstring: kb_id（知识库ID）
KnowledgeFileId = UUIDStr  # docstring: file_id（知识文件ID）
DocumentId = UUIDStr  # docstring: document_id（文档ID）
NodeId = UUIDStr  # docstring: node_id（引用节点ID）
RetrievalRecordId = UUIDStr  # docstring: retrieval_record_id（检索记录ID）
GenerationRecordId = UUIDStr  # docstring: generation_record_id（生成记录ID）
EvaluationRecordId = UUIDStr  # docstring: evaluation_record_id（评估记录ID）

ErrorCode = Literal[
    "bad_request",
    "not_found",
    "external_dependency",
    "pipeline_error",
    "internal_error",
]  # docstring: HTTP 层标准错误码

ErrorDetail = Dict[str, Any]  # docstring: ErrorResponse.error.detail 结构（必须 JSON-safe）


class ErrorInfo(BaseModel):
    """
    [职责] ErrorInfo：统一错误载体（code/message/trace_id/detail）。
    [边界] 不包含 HTTP status/retryable；这些由 api/errors.py 决定。
    [上游关系] api/errors.py 将 DomainError 映射为 ErrorInfo。
    [下游关系] 前端/UI/审计系统消费统一错误结构。
    """

    model_config = ConfigDict(extra="forbid")  # docstring: 锁死错误字段，避免 drift

    code: ErrorCode = Field(...)  # docstring: 错误码（标准枚举）
    message: str = Field(..., min_length=1)  # docstring: 人类可读错误信息
    trace_id: TraceId = Field(...)  # docstring: 全链路追踪ID（由 middleware 注入）
    detail: ErrorDetail = Field(default_factory=dict)  # docstring: 结构化细节（可为空）


class ErrorResponse(BaseModel):
    """
    [职责] ErrorResponse：HTTP 错误响应的顶层包裹结构。
    [边界] 仅包含 error 字段；trace_id/request_id 由 header 透传。
    [上游关系] routers/api/errors.py 返回此结构。
    [下游关系] 前端统一处理错误展示与告警。
    """

    model_config = ConfigDict(extra="forbid")  # docstring: 保持响应结构稳定

    error: ErrorInfo = Field(...)  # docstring: 错误主体


class DebugRecords(BaseModel):
    """
    [职责] DebugRecords：记录 ID 的稳定容器（允许按场景扩展）。
    [边界] 仅承载 ID 字段；不包含 record 内容。
    [上游关系] services/pipelines 在 debug 模式下回填记录 ID。
    [下游关系] schemas_http debug 输出复用。
    """

    model_config = ConfigDict(extra="allow")  # docstring: 允许新增 record_id 字段

    retrieval_record_id: Optional[RetrievalRecordId] = Field(default=None)  # docstring: 检索记录ID
    generation_record_id: Optional[GenerationRecordId] = Field(default=None)  # docstring: 生成记录ID
    evaluation_record_id: Optional[EvaluationRecordId] = Field(default=None)  # docstring: 评估记录ID
    document_id: Optional[DocumentId] = Field(default=None)  # docstring: 导入产生的文档ID（可选）


class DebugEnvelope(BaseModel):
    """
    [职责] DebugEnvelope：debug=true 时的统一调试封装（trace/request/records/timing）。
    [边界] 不保证字段齐全性；仅保证结构稳定与可扩展。
    [上游关系] services 返回 debug 对象；middleware 注入 trace/request。
    [下游关系] 前端 debug 模式/审计页面解析。
    """

    model_config = ConfigDict(extra="allow")  # docstring: 允许扩展 gate/timing 细节

    trace_id: TraceId = Field(...)  # docstring: 全链路追踪ID
    request_id: RequestId = Field(...)  # docstring: 单次请求ID
    records: DebugRecords = Field(default_factory=DebugRecords)  # docstring: record_id 集合
    timing_ms: Dict[str, Any] = Field(default_factory=dict)  # docstring: timing 明细（ms，可嵌套）
