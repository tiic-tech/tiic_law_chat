# playground/fastapi_gate/routers/test_ingest_router_gate.py

"""
[职责] Ingest router gate: verify /ingest returns mapped response fields.
[边界] Stub ingest_service to avoid pipeline/Milvus dependencies.
[上游关系] backend/api/routers/ingest.py.
[下游关系] Ensures HTTP mapping contract is stable.
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
from uae_law_rag.backend.api.routers.ingest import router as ingest_router
from uae_law_rag.backend.schemas.ids import new_uuid
from uae_law_rag.backend.utils.constants import REQUEST_ID_KEY, TIMING_MS_KEY, TRACE_ID_KEY


pytestmark = pytest.mark.fastapi_gate


class _MilvusStub:
    """Milvus stub for router gate."""  # docstring: avoid Milvus access


@pytest.mark.asyncio
async def test_ingest_router_gate(session: AsyncSession, monkeypatch: pytest.MonkeyPatch) -> None:
    """
    [职责] 验证 ingest router 输出字段映射正确。
    [边界] 不覆盖 pipeline；仅验证 HTTP 映射。
    [上游关系] /ingest router。
    [下游关系] 前端依赖此响应结构。
    """
    trace_id = new_uuid()
    request_id = new_uuid()

    async def _override_session() -> AsyncIterator[AsyncSession]:
        yield session  # docstring: reuse test session

    def _override_milvus_repo() -> _MilvusStub:
        return _MilvusStub()  # docstring: stub Milvus repo

    async def _fake_ingest_file(**_kwargs):
        return {
            "kb_id": str(new_uuid()),
            "file_id": str(new_uuid()),
            "status": "success",
            "node_count": 3,
            TIMING_MS_KEY: {"total_ms": 12.0},
            TRACE_ID_KEY: str(trace_id),
            REQUEST_ID_KEY: str(request_id),
        }  # docstring: stub service output

    monkeypatch.setattr(
        "uae_law_rag.backend.api.routers.ingest.ingest_file",
        _fake_ingest_file,
    )  # docstring: patch service call

    app = FastAPI()
    app.add_middleware(TraceContextMiddleware)  # docstring: inject trace/request headers
    app.include_router(ingest_router)  # docstring: mount ingest router
    app.dependency_overrides[get_session] = _override_session  # docstring: override session dep
    app.dependency_overrides[get_milvus_repo] = _override_milvus_repo  # docstring: override Milvus dep

    payload = {
        "kb_id": str(new_uuid()),
        "file_name": "test.pdf",
        "source_uri": "file:///tmp/test.pdf",
        "dry_run": False,
    }

    transport = ASGITransport(app=app)  # docstring: ASGI transport for httpx
    headers = {
        "x-trace-id": str(trace_id),
        "x-request-id": str(request_id),
    }  # docstring: explicit trace/request headers
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/ingest", json=payload, headers=headers)  # docstring: invoke ingest endpoint

    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"
    assert data["file_id"]
    assert data["file_name"] == "test.pdf"
    assert data["node_count"] == 3
    assert "timing_ms" in data
    assert "x-trace-id" in resp.headers  # docstring: header must include trace_id
    assert "x-request-id" in resp.headers  # docstring: header must include request_id
    assert resp.headers["x-trace-id"] == str(trace_id)  # docstring: trace_id must propagate
    assert resp.headers["x-request-id"] == str(request_id)  # docstring: request_id must propagate
