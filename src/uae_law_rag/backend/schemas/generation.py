# src/uae_law_rag/backend/schemas/generation.py

"""
[职责] Generation 契约层：定义生成阶段的结构化记录（GenerationRecord）与引用证据（Citation/Evidence）结构。
[边界] 不实现 LLM 调用；不实现 prompt 模板；仅表达可回放输入快照与输出快照（raw/structured/citations/status）。
[上游关系] retrieval bundle 提供上下文证据；chat message 提供 query；prompt/render 提供 messages_snapshot。
[下游关系] generation_repo 写入 GenerationRecordModel；chat service 回写 MessageModel.response 与结构化字段（可选）。
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from .ids import GenerationRecordId, MessageId, NodeId, RetrievalRecordId


GenerationStatus = Literal["success", "failed", "partial"]  # docstring: 生成状态（MVP 与 evaluator 对齐）


class Citation(BaseModel):
    """
    [职责] Citation：回答中引用的证据指针（面向可回查与可解释性）。
    [边界] 不存储全文；只存 node_id 与可选定位信息；具体文本回查走 SQL NodeModel。
    [上游关系] retrieval hits（node_id/page/article_id 等）为 citations 提供来源。
    [下游关系] evaluator/审计/前端高亮引用使用该结构。
    """

    model_config = ConfigDict(extra="allow")

    node_id: NodeId = Field(...)  # docstring: 证据节点（SQL NodeModel.id）
    rank: Optional[int] = Field(default=None, ge=0, le=100000)  # docstring: 在 retrieval hits 中的排序位置（可选）
    quote: str = Field(default="")  # docstring: 可选引用片段（短文本，用于 UI；不应超过少量字符）
    locator: Dict[str, Any] = Field(default_factory=dict)  # docstring: 定位信息（page/article_id/section_path 等）


class GenerationRecord(BaseModel):
    """
    [职责] GenerationRecord：一次生成的可回放“输入快照 + 模型快照 + 输出快照 + 引用证据”。
    [边界] 不包含完整 prompt 模板系统实现；messages_snapshot 只作为回放输入，不保证可直接执行。
    [上游关系] Message + RetrievalRecord 决定输入；prompt/render 与 model/router 决定 messages_snapshot/provider。
    [下游关系] 写入 GenerationRecordModel；chat response 可直接返回 output_raw 与 citations。
    """

    model_config = ConfigDict(extra="forbid")

    id: GenerationRecordId = Field(...)  # docstring: GenerationRecord 唯一 ID（UUID str）
    message_id: MessageId = Field(...)  # docstring: 归属消息（一致化：1 message ↔ 1 generation_record）
    retrieval_record_id: RetrievalRecordId = Field(...)  # docstring: 关联检索记录（证据来源）

    prompt_name: str = Field(..., min_length=1, max_length=200)  # docstring: prompt 模板名称（版本化的逻辑名）
    prompt_version: Optional[str] = Field(default=None, max_length=50)  # docstring: prompt 版本（可选）

    model_provider: str = Field(..., min_length=1, max_length=50)  # docstring: 模型提供方（ollama/openai/...）
    model_name: str = Field(..., min_length=1, max_length=200)  # docstring: 模型名称（例如 llama3 / gpt-4.1）
    provider_snapshot: Dict[str, Any] = Field(
        default_factory=dict
    )  # docstring: provider 参数快照（temperature/top_p 等）

    messages_snapshot: Dict[str, Any] = Field(default_factory=dict)  # docstring: 输入消息快照（system/user/context 等）
    output_raw: str = Field(default="")  # docstring: 模型原始输出文本
    output_structured: Optional[Dict[str, Any]] = Field(default=None)  # docstring: 结构化输出（JSON dict），可选

    citations: List[Citation] = Field(default_factory=list)  # docstring: 引用证据列表（可回查）
    status: GenerationStatus = Field(default="success")  # docstring: 生成状态
    error_message: Optional[str] = Field(default=None)  # docstring: 失败原因（若 status != success）


class GenerationBundle(BaseModel):
    """
    [职责] GenerationBundle：pipeline 内部传输对象（record + 便捷字段），供 services 统一返回或落库。
    [边界] 不强制用于对外 HTTP；可由 chat schema 映射为更轻量结构。
    [上游关系] generator pipeline 产出 GenerationRecord。
    [下游关系] message 回写、evaluator、前端展示。
    """

    model_config = ConfigDict(extra="forbid")

    record: GenerationRecord = Field(...)  # docstring: 生成记录（可回放）
    answer: str = Field(default="")  # docstring: 便捷字段（通常=record.output_raw）
