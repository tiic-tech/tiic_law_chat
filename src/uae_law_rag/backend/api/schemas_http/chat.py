# src/uae_law_rag/backend/api/schemas_http/chat.py

"""
[职责] HTTP Chat Schema：定义对外 chat 请求/响应合同，承接产品侧输入与输出。
[边界] 不包含业务编排与 DB 语义；不暴露内部 schema 全量结构。
[上游关系] 前端或外部调用方发起 chat 请求。
[下游关系] routers/chat.py 调用 chat_service 并映射为本模块输出结构。
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from ._common import (
    ConversationId,
    DebugEnvelope,
    KnowledgeBaseId,
    MessageId,
    NodeId,
    RequestId,
    TraceId,
)


ChatStatus = Literal["success", "blocked", "partial", "failed"]  # docstring: chat 状态（对外合同）
EvaluatorStatus = Literal["pass", "partial", "fail", "skipped"]  # docstring: evaluator 状态摘要


class ChatContextConfig(BaseModel):
    """
    [职责] ChatContextConfig：HTTP 层可选配置（检索/生成/evaluator 的轻量覆盖）。
    [边界] 仅表达“建议/覆盖”；最终策略由服务端解析并落库。
    [上游关系] 前端高级设置输入。
    [下游关系] chat_service 将其映射为 retrieval/generation/evaluator config。
    """

    model_config = ConfigDict(extra="allow")  # docstring: 允许未来扩展字段

    keyword_top_k: Optional[int] = Field(default=None)  # docstring: keyword 召回上限
    vector_top_k: Optional[int] = Field(default=None)  # docstring: vector 召回上限
    fusion_top_k: Optional[int] = Field(default=None)  # docstring: fusion 截断上限
    rerank_top_k: Optional[int] = Field(default=None)  # docstring: rerank 输出上限

    fusion_strategy: Optional[str] = Field(default=None)  # docstring: fusion 策略
    rerank_strategy: Optional[str] = Field(default=None)  # docstring: rerank 策略

    embed_provider: Optional[str] = Field(default=None)  # docstring: embed provider
    embed_model: Optional[str] = Field(default=None)  # docstring: embed 模型名称
    embed_dim: Optional[int] = Field(default=None)  # docstring: embed 维度

    model_provider: Optional[str] = Field(default=None)  # docstring: LLM provider
    model_name: Optional[str] = Field(default=None)  # docstring: LLM 模型名称

    prompt_name: Optional[str] = Field(default=None)  # docstring: prompt 名称
    prompt_version: Optional[str] = Field(default=None)  # docstring: prompt 版本

    evaluator_config: Optional[Dict[str, Any]] = Field(default=None)  # docstring: evaluator 配置覆盖
    return_records: Optional[bool] = Field(default=None)  # docstring: 仅返回 record_id（不输出 gate/provider）
    return_hits: Optional[bool] = Field(default=None)  # docstring: 预留字段（是否返回 hits 详情）


class ChatRequest(BaseModel):
    """
    [职责] ChatRequest：对外 chat 请求结构（query + 可选 KB/会话/上下文）。
    [边界] 不携带历史消息全量；不携带内部 record/hit 结构。
    [上游关系] 前端输入 query 发起请求。
    [下游关系] chat_service 解析上下文并编排 retrieval/generation/evaluator。
    """

    model_config = ConfigDict(extra="forbid")  # docstring: 锁定对外输入合同

    query: str = Field(..., min_length=1, max_length=4096)  # docstring: 用户问题
    conversation_id: Optional[ConversationId] = Field(default=None)  # docstring: 续聊会话ID
    kb_id: Optional[KnowledgeBaseId] = Field(default=None)  # docstring: KB 作用域
    context: Optional[ChatContextConfig] = Field(default=None)  # docstring: 可选配置覆盖
    debug: bool = Field(default=False)  # docstring: debug 开关


class EvaluatorSummary(BaseModel):
    """
    [职责] EvaluatorSummary：评估摘要（status/rule_version/warnings）。
    [边界] 不输出完整 checks/scores，仅提供最小可解释字段。
    [上游关系] evaluator_service 生成 summary。
    [下游关系] ChatResponse.evaluator 输出给前端。
    """

    model_config = ConfigDict(extra="forbid")  # docstring: 锁定摘要结构

    status: EvaluatorStatus = Field(...)  # docstring: evaluator 总状态
    rule_version: str = Field(default="v0")  # docstring: 规则版本
    warnings: List[str] = Field(default_factory=list)  # docstring: 警告摘要列表


class CitationView(BaseModel):
    """
    [职责] CitationView：引用证据摘要（节点指针 + 定位信息）。
    [边界] 不包含全文；仅保留引用片段与定位字段。
    [上游关系] generation 输出 citations。
    [下游关系] 前端引用渲染与可解释性展示。
    """

    model_config = ConfigDict(extra="allow")  # docstring: 允许扩展定位字段

    node_id: NodeId = Field(...)  # docstring: 引用节点ID
    rank: Optional[int] = Field(default=None, ge=0, le=100000)  # docstring: 引用排序（可选）
    quote: str = Field(default="")  # docstring: 引用片段（可选）

    page: Optional[int] = Field(default=None)  # docstring: 页码定位（可选）
    article_id: Optional[str] = Field(default=None)  # docstring: 法条ID（可选）
    section_path: Optional[str] = Field(default=None)  # docstring: 章节路径（可选）

    locator: Dict[str, Any] = Field(
        default_factory=dict
    )  # docstring: 兼容 locator 结构；locator 是冗余兼容字段，page/article/section 是主字段


class ChatGateSummary(BaseModel):
    """
    [职责] ChatGateSummary：debug 下 gate 状态摘要（retrieval/generation/evaluator）。
    [边界] 仅用于 debug 输出；不参与业务裁决。
    [上游关系] chat_service 生成 gate 记录。
    [下游关系] debug.gate 输出给前端。
    """

    model_config = ConfigDict(extra="allow")  # docstring: 允许扩展 gate 字段

    retrieval: Optional[Dict[str, Any]] = Field(default=None)  # docstring: retrieval gate 摘要
    generation: Optional[Dict[str, Any]] = Field(default=None)  # docstring: generation gate 摘要
    evaluator: Optional[Dict[str, Any]] = Field(default=None)  # docstring: evaluator gate 摘要


class ChatDebugEnvelope(DebugEnvelope):
    """
    [职责] ChatDebugEnvelope：chat debug 输出（trace/request/records/gate/timing）。
    [边界] 不保证字段齐全；仅保证结构稳定与可扩展。
    [上游关系] chat_service 在 debug=true 时返回。
    [下游关系] 前端调试模式与审计回放。
    """

    gate: Optional[ChatGateSummary] = Field(default=None)  # docstring: gate 摘要


class ChatTimingMs(BaseModel):
    """
    [职责] ChatTimingMs：chat 响应耗时统计容器（ms）。
    [边界] 不强制阶段字段；仅输出总耗时。
    [上游关系] chat_service 返回 timing_ms。
    [下游关系] 前端/审计展示耗时。
    """

    model_config = ConfigDict(extra="allow")  # docstring: 允许扩展阶段字段

    total_ms: Optional[float] = Field(default=None, ge=0.0)  # docstring: 总耗时（ms）


class ChatResponse(BaseModel):
    """
    [职责] ChatResponse：对外 chat 响应结构（answer/citations/status/evaluator）。
    [边界] 不输出内部 provider_snapshot/messages_snapshot；仅输出对外契约字段。
    [上游关系] chat_service 输出结果映射。
    [下游关系] 前端渲染回答、引用与状态。
    """

    model_config = ConfigDict(extra="forbid")  # docstring: 锁定对外输出合同

    conversation_id: ConversationId = Field(...)  # docstring: 会话ID
    message_id: MessageId = Field(...)  # docstring: 消息ID
    kb_id: KnowledgeBaseId = Field(...)  # docstring: 知识库ID

    status: ChatStatus = Field(...)  # docstring: chat 状态
    answer: str = Field(default="")  # docstring: 模型回答
    citations: List[CitationView] = Field(default_factory=list)  # docstring: 引用证据列表
    evaluator: EvaluatorSummary = Field(...)  # docstring: evaluator 摘要
    timing_ms: ChatTimingMs = Field(default_factory=ChatTimingMs)  # docstring: timing 统计
    trace_id: TraceId = Field(...)  # docstring: trace_id
    request_id: RequestId = Field(...)  # docstring: request_id
    debug: Optional[ChatDebugEnvelope] = Field(default=None)  # docstring: debug 输出（可选）
