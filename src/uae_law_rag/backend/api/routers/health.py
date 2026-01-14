# src/uae_law_rag/backend/api/routers/health.py

"""
[职责] Health Router：提供服务健康检查（DB/Milvus）与版本摘要。
[边界] 不执行业务逻辑；不触发 pipeline；仅探测依赖健康状态。
[上游关系] 运维/监控系统调用健康检查接口。
[下游关系] 依赖 DB/Milvus 客户端执行轻量检查。
"""

from __future__ import annotations

from typing import Any, Dict

from fastapi import APIRouter, Depends
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from uae_law_rag.backend.api.deps import get_milvus_repo, get_session
from uae_law_rag.backend.kb.repo import MilvusRepo


router = APIRouter(prefix="/health", tags=["health"])  # docstring: health 路由前缀


@router.get("")
async def health_check(
    session: AsyncSession = Depends(get_session),
    milvus_repo: MilvusRepo = Depends(get_milvus_repo),
) -> Dict[str, Any]:
    """
    [职责] 检测 DB/Milvus 可用性并返回健康摘要。
    [边界] 只做轻量探测；不做业务写入或耗时操作。
    [上游关系] 运维/监控系统或 CI 探针调用。
    [下游关系] DB/Milvus 客户端执行健康检查。
    """
    status = "ok"  # docstring: 默认健康状态
    db_status: Dict[str, Any] = {"ok": True}  # docstring: DB 健康容器
    milvus_status: Dict[str, Any] = {"ok": True, "optional": True}  # docstring: Milvus 健康容器

    try:
        await session.execute(text("SELECT 1"))  # docstring: DB ping（最小读）
    except Exception as exc:
        db_status["ok"] = False  # docstring: 标记 DB 不可用
        db_status["error"] = f"{exc.__class__.__name__}: {exc}"  # docstring: 记录 DB 错误摘要

    try:
        client = getattr(milvus_repo, "_client", None)  # docstring: 获取底层 MilvusClient
        if client is None:
            raise RuntimeError("milvus client not available")  # docstring: 缺失 client 视为不可用
        await client.healthcheck()  # docstring: Milvus ping
    except Exception as exc:
        milvus_status["ok"] = False  # docstring: 标记 Milvus 不可用
        milvus_status["error"] = f"{exc.__class__.__name__}: {exc}"  # docstring: 记录 Milvus 错误摘要

    if not db_status.get("ok") or not milvus_status.get("ok"):
        status = "degraded"  # docstring: 任一依赖异常则降级

    return {
        "status": status,
        "db": db_status,
        "milvus": milvus_status,
        "version": {"api": "v1"},
    }  # docstring: 健康检查响应
