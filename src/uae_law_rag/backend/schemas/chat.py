# src/uae_law_rag/backend/schemas/chat.py

"""
[职责] Chat HTTP 契约层：定义对外接口的请求/响应结构（ChatRequest/ChatResponse），并对齐内部 Message/Retrieval/Generation 引用关系。
[边界] 仅表达 HTTP 输入输出；不包含 DB 写入、不包含 pipeline 编排、不包含 LLM 调用。
[上游关系] 前端（React/Vite）或外部调用方发起 chat 请求；可携带 conversation_id 续聊或创建新会话。
[下游关系] services/chat_service.py 消费 ChatRequest 调用 retrieval/generation pipeline 并返回 ChatResponse。
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from .generation import Citation
from .ids import ConversationId, KnowledgeBaseId, MessageId, RetrievalRecordId, UserId


ChatType = Literal["chat", "agent_chat"]  # docstring: 对话类型（MVP 先用 chat）


class ChatContextConfig(BaseModel):
    """
    [职责] ChatContextConfig：对话级检索/生成策略的轻量配置（HTTP 层可选覆盖）。
    [边界] 仅作为“建议/覆盖”；最终策略由服务端决定并写入 provider_snapshot/record。
    [上游关系] 前端可在高级设置面板传入。
    [下游关系] chat_service 将其转换为 retrieval/generation 的 profile/params，并落入记录对象。
    """

    model_config = ConfigDict(extra="allow")

    keyword_top_k: Optional[int] = Field(default=None)  # docstring: keyword 召回上限（可选覆盖）
    vector_top_k: Optional[int] = Field(default=None)  # docstring: vector 召回上限（可选覆盖）
    rerank_top_k: Optional[int] = Field(default=None)  # docstring: rerank 输出上限（可选覆盖）

    model_provider: Optional[str] = Field(default=None)  # docstring: 生成模型提供方（可选覆盖）
    model_name: Optional[str] = Field(default=None)  # docstring: 生成模型名称（可选覆盖）
    temperature: Optional[float] = Field(default=None, ge=0.0, le=2.0)  # docstring: 生成温度（可选覆盖）


class ChatRequest(BaseModel):
    """
    [职责] ChatRequest：对外 chat 请求结构，支持新建/续聊，并显式指定 KB 作用域。
    [边界] 不携带历史 messages 全量（后端可按 conversation_id 自行加载）；不携带内部 record/hit 结构。
    [上游关系] 前端输入 query 发起请求。
    [下游关系] chat_service 创建 MessageModel（user message）并驱动 retrieval/generation。
    """

    model_config = ConfigDict(extra="forbid")

    user_id: Optional[UserId] = Field(default=None)  # docstring: 用户ID（若有鉴权体系可从 token 推断）
    conversation_id: Optional[ConversationId] = Field(default=None)  # docstring: 续聊会话ID；为空则创建新会话

    chat_type: ChatType = Field(default="chat")  # docstring: chat 类型
    query: str = Field(..., min_length=1, max_length=4096)  # docstring: 用户输入问题

    kb_id: Optional[KnowledgeBaseId] = Field(
        default=None
    )  # docstring: 显式 KB 作用域；为空则使用 conversation.default_kb_id
    context: ChatContextConfig = Field(default_factory=ChatContextConfig)  # docstring: 高级配置（可选覆盖）
    meta: Dict[str, Any] = Field(default_factory=dict)  # docstring: 扩展元信息（前端埋点/渠道等）


class ChatDebugInfo(BaseModel):
    """
    [职责] ChatDebugInfo：面向调试/评估的可选输出（记录 ID、命中数量、provider 快照摘要）。
    [边界] 不输出敏感凭据；不输出完整 prompt；仅输出可审计的摘要信息。
    [上游关系] chat_service 在完成 retrieval/generation 后组装。
    [下游关系] 前端调试模式/评估脚本可消费。
    """

    model_config = ConfigDict(extra="allow")

    message_id: Optional[MessageId] = Field(default=None)  # docstring: 本次 user message id
    retrieval_record_id: Optional[RetrievalRecordId] = Field(default=None)  # docstring: 本次 retrieval record id
    hits_count: Optional[int] = Field(default=None, ge=0)  # docstring: 最终命中条数（rerank 后）

    model_provider: Optional[str] = Field(default=None)  # docstring: 实际使用的 provider
    model_name: Optional[str] = Field(default=None)  # docstring: 实际使用的 model name
    latency_ms: Optional[float] = Field(default=None)  # docstring: 本次请求端到端耗时（粗粒度）


class ChatResponse(BaseModel):
    """
    [职责] ChatResponse：对外 chat 响应结构，返回 answer + citations，并携带会话/消息引用以便前端续聊。
    [边界] 不返回 retrieval hits 全量（除非 debug 扩展）；不返回内部 provider_snapshot 全量（仅摘要）。
    [上游关系] generation pipeline 产出 answer 与 citations。
    [下游关系] 前端渲染回答、引用证据；后续请求携带 conversation_id 续聊。
    """

    model_config = ConfigDict(extra="forbid")

    conversation_id: ConversationId = Field(...)  # docstring: 会话ID（新建或续聊）
    message_id: MessageId = Field(...)  # docstring: 本次 message（user+assistant 对应的主键引用）

    chat_type: ChatType = Field(default="chat")  # docstring: chat 类型
    kb_id: KnowledgeBaseId = Field(...)  # docstring: 实际使用的 KB 作用域

    answer: str = Field(default="")  # docstring: 模型回答（通常=GenerationRecord.output_raw）
    citations: List[Citation] = Field(default_factory=list)  # docstring: 引用证据列表（node_id 级别）

    debug: Optional[ChatDebugInfo] = Field(default=None)  # docstring: 调试信息（可选，默认不输出）
