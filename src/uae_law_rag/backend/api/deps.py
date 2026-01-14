# src/uae_law_rag/backend/api/deps.py

"""
[职责] API 依赖装配：提供 session、trace_context、repos 与 MilvusRepo 注入。
[边界] 不做业务逻辑；不提交事务；不进行网络/DB 读写以外的处理。
[上游关系] FastAPI 路由层调用依赖注入。
[下游关系] services/routers 通过本模块获取依赖实例。
"""

from __future__ import annotations

from typing import AsyncIterator

from fastapi import Request
from sqlalchemy.ext.asyncio import AsyncSession

from uae_law_rag.backend.db.engine import SessionLocal
from uae_law_rag.backend.db.repo import (
    EvaluatorRepo,
    GenerationRepo,
    IngestRepo,
    MessageRepo,
    RetrievalRepo,
)
from uae_law_rag.backend.kb.client import MilvusClient
from uae_law_rag.backend.kb.repo import MilvusRepo
from uae_law_rag.backend.schemas.audit import TraceContext
from uae_law_rag.backend.schemas.ids import UUIDStr, new_uuid


async def get_session() -> AsyncIterator[AsyncSession]:
    """
    [职责] 获取数据库会话（每个 request 一个 session）。
    [边界] 不提交/回滚事务；仅负责创建与关闭。
    [上游关系] FastAPI dependency 注入调用。
    [下游关系] repos/services 使用该 session。
    """
    async with SessionLocal() as session:
        yield session  # docstring: 输出 session 给下游使用


def get_trace_context(request: Request) -> TraceContext:
    """
    [职责] 获取或创建 TraceContext（优先使用 middleware 注入）。
    [边界] 不写入日志；不校验 UUID 格式。
    [上游关系] middleware 注入 request.state.trace_context。
    [下游关系] services 需要 trace_id/request_id 时调用。
    """
    existing = getattr(request.state, "trace_context", None)
    if isinstance(existing, TraceContext):
        return existing  # docstring: 复用 middleware 注入的 TraceContext

    trace_id = str(getattr(request.state, "trace_id", "") or "").strip()
    request_id = str(getattr(request.state, "request_id", "") or "").strip()
    parent_request_id = str(getattr(request.state, "parent_request_id", "") or "").strip()

    if not trace_id:
        trace_id = str(new_uuid())  # docstring: 缺失 trace_id 则生成
    if not request_id:
        request_id = str(new_uuid())  # docstring: 缺失 request_id 则生成

    ctx = TraceContext(
        trace_id=UUIDStr(trace_id),
        request_id=UUIDStr(request_id),
        parent_request_id=UUIDStr(parent_request_id) if parent_request_id else None,
        tags={},
    )  # docstring: 兜底构造 TraceContext
    request.state.trace_context = ctx  # docstring: 写回 state 以复用
    request.state.trace_id = trace_id  # docstring: 便捷字段
    request.state.request_id = request_id  # docstring: 便捷字段
    if parent_request_id:
        request.state.parent_request_id = parent_request_id  # docstring: 写回 parent_request_id
    return ctx


def get_ingest_repo(session: AsyncSession) -> IngestRepo:
    """
    [职责] 获取 IngestRepo 实例。
    [边界] 不执行 DB I/O；仅装配仓储。
    [上游关系] routers/ingest 调用。
    [下游关系] ingest_service/pipelines 使用。
    """
    return IngestRepo(session)  # docstring: 装配 ingest repo


def get_message_repo(session: AsyncSession) -> MessageRepo:
    """
    [职责] 获取 MessageRepo 实例。
    [边界] 不执行 DB I/O；仅装配仓储。
    [上游关系] routers/chat 调用。
    [下游关系] chat_service 使用。
    """
    return MessageRepo(session)  # docstring: 装配 message repo


def get_retrieval_repo(session: AsyncSession) -> RetrievalRepo:
    """
    [职责] 获取 RetrievalRepo 实例。
    [边界] 不执行 DB I/O；仅装配仓储。
    [上游关系] routers/chat 调用。
    [下游关系] retrieval_service 使用。
    """
    return RetrievalRepo(session)  # docstring: 装配 retrieval repo


def get_generation_repo(session: AsyncSession) -> GenerationRepo:
    """
    [职责] 获取 GenerationRepo 实例。
    [边界] 不执行 DB I/O；仅装配仓储。
    [上游关系] routers/chat 调用。
    [下游关系] generation_service 使用。
    """
    return GenerationRepo(session)  # docstring: 装配 generation repo


def get_evaluator_repo(session: AsyncSession) -> EvaluatorRepo:
    """
    [职责] 获取 EvaluatorRepo 实例。
    [边界] 不执行 DB I/O；仅装配仓储。
    [上游关系] routers/chat 调用。
    [下游关系] evaluator_service 使用。
    """
    return EvaluatorRepo(session)  # docstring: 装配 evaluator repo


def get_milvus_repo() -> MilvusRepo:
    """
    [职责] 获取 MilvusRepo 实例（向量库接口）。
    [边界] 不做 collection 初始化；仅建立连接与仓储封装。
    [上游关系] routers/ingest/chat 调用。
    [下游关系] ingest_service/retrieval_service 使用。
    """
    client = MilvusClient.from_env()  # docstring: 从环境加载 Milvus 连接
    return MilvusRepo(client)  # docstring: 装配 Milvus repo
