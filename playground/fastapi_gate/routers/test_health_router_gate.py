# playground/fastapi_gate/routers/test_health_router_gate.py

"""
[职责] Health router gate：验证 /health 输出结构与健康标记。
[边界] Stub Milvus/DB 依赖；不触发真实外部连接。
[上游关系] backend/api/routers/health.py。
[下游关系] 确保健康检查契约稳定。
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import AsyncIterator

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SRC_ROOT = _REPO_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))  # docstring: ensure local src import

from uae_law_rag.backend.api.deps import get_milvus_repo, get_session
from uae_law_rag.backend.api.middleware import TraceContextMiddleware
from uae_law_rag.backend.api.routers.health import router as health_router
from uae_law_rag.backend.schemas.ids import new_uuid


pytestmark = pytest.mark.fastapi_gate


class _MilvusClientStub:
    """Milvus client stub."""  # docstring: avoid Milvus access

    async def healthcheck(self) -> None:
        return None  # docstring: stub healthcheck


class _MilvusRepoStub:
    """Milvus repo stub."""  # docstring: avoid Milvus access

    def __init__(self) -> None:
        self._client = _MilvusClientStub()  # docstring: provide client stub


@pytest.mark.asyncio
async def test_health_router_gate(session: AsyncSession, monkeypatch: pytest.MonkeyPatch) -> None:
    """
    [职责] 验证 health router 输出结构与 header 透传。
    [边界] 不覆盖真实 DB/Milvus；仅验证 HTTP 输出。
    [上游关系] /health router。
    [下游关系] 运维/监控依赖此响应结构。
    """
    trace_id = new_uuid()
    request_id = new_uuid()

    async def _override_session() -> AsyncIterator[AsyncSession]:
        yield session  # docstring: reuse test session

    app = FastAPI()
    app.add_middleware(TraceContextMiddleware)  # docstring: inject trace/request headers
    app.include_router(health_router)  # docstring: mount health router
    app.dependency_overrides[get_session] = _override_session  # docstring: override session dep
    app.dependency_overrides[get_milvus_repo] = lambda: _MilvusRepoStub()  # docstring: override Milvus dep

    transport = ASGITransport(app=app)  # docstring: ASGI transport for httpx
    headers = {
        "x-trace-id": str(trace_id),
        "x-request-id": str(request_id),
    }  # docstring: explicit trace/request headers
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/health", headers=headers)  # docstring: invoke health endpoint

    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["db"]["ok"] is True
    assert data["milvus"]["ok"] is True
    assert data["version"]["api"] == "v1"
    assert resp.headers["x-trace-id"] == str(trace_id)  # docstring: trace_id must propagate
    assert resp.headers["x-request-id"] == str(request_id)  # docstring: request_id must propagate
