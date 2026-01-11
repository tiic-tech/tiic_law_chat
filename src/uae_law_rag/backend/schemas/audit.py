# src/uae_law_rag/backend/schemas/audit.py

"""
[职责] Audit 契约层：统一记录 provider_snapshot、timing、trace/request 标识等可回放与可观测字段。
[边界] 不负责日志落盘/上报（由 utils/logging_ 或 observability 系统负责）；仅提供结构化字段定义。
[上游关系] services/pipelines 在一次请求/一次 step 中生成 trace_id/request_id 并填充 timing/provider 信息。
[下游关系] DB 记录（RetrievalRecord/GenerationRecord/Message.meta_data）与前端 debug 输出可复用该结构。
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from pydantic import BaseModel, ConfigDict, Field

from .ids import UUIDStr, new_uuid


class TraceContext(BaseModel):
    """
    [职责] TraceContext：一次请求/一次会话编排的追踪上下文（trace_id/request_id）。
    [边界] 仅标识与轻量 tags；不包含 span 级别细节（可由外部 APM 承担）。
    [上游关系] API 层或 service 层创建；也可从前端透传 request_id。
    [下游关系] 各 pipeline step 的 record/provider_snapshot 引用，便于串联排障。
    """

    model_config = ConfigDict(extra="allow")

    trace_id: UUIDStr = Field(default_factory=new_uuid)  # docstring: 全链路追踪ID（跨 steps）
    request_id: UUIDStr = Field(default_factory=new_uuid)  # docstring: 单次 HTTP 请求ID（或单轮对话ID）

    parent_request_id: Optional[UUIDStr] = Field(default=None)  # docstring: 上游请求ID（可用于批处理/链式调用）
    tags: Dict[str, Any] = Field(default_factory=dict)  # docstring: 任意扩展 tags（env/build/user-agent 等）


class ProviderSnapshot(BaseModel):
    """
    [职责] ProviderSnapshot：统一的“模型/向量/重排器”等 provider 配置快照（可回放）。
    [边界] 不存储敏感密钥；仅存储模型名、参数、版本、endpoint 等非敏感信息。
    [上游关系] adapter/router 在 resolve provider 后生成快照；pipelines 写入 record.provider_snapshot。
    [下游关系] 回归测试、审计、问题复现使用；前端 debug 可输出摘要信息。
    """

    model_config = ConfigDict(extra="allow")

    kind: str = Field(..., min_length=1, max_length=50)  # docstring: provider 类型（llm/embedder/reranker/vector_db）
    provider: str = Field(..., min_length=1, max_length=50)  # docstring: 提供方（ollama/openai/milvus/...）
    name: str = Field(
        ..., min_length=1, max_length=200
    )  # docstring: 模型/组件名称（model_name/embed_model/index_type）

    params: Dict[str, Any] = Field(default_factory=dict)  # docstring: 非敏感参数快照（temperature/top_k/nprobe 等）
    version: Optional[str] = Field(default=None, max_length=50)  # docstring: 版本号（可选）
    endpoint: Optional[str] = Field(default=None, max_length=500)  # docstring: endpoint/host（可选，避免敏感信息）


class TimingSnapshot(BaseModel):
    """
    [职责] TimingSnapshot：统一时间统计快照（端到端或 step 粒度）。
    [边界] 不强制统计方式；仅提供字段容器；允许 extra 扩展。
    [上游关系] service/pipeline 在关键节点填充。
    [下游关系] DB/meta_data/debug 输出；性能回归对比。
    """

    model_config = ConfigDict(extra="allow")

    total_ms: Optional[float] = Field(default=None)  # docstring: 总耗时（ms）
    breakdown: Dict[str, float] = Field(default_factory=dict)  # docstring: 分段耗时（key=阶段名, value=ms）
    started_at: Optional[str] = Field(default=None)  # docstring: 开始时间（ISO 字符串，可选）
    ended_at: Optional[str] = Field(default=None)  # docstring: 结束时间（ISO 字符串，可选）
