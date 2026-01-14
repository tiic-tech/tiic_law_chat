# playground/fastapi_gate/routers/test_chat_router_gate.py

"""
[职责] Chat router gate：验证 /chat 输出字段与 debug 映射。
[边界] Stub chat_service，避免 pipeline/Milvus 依赖。
[上游关系] backend/api/routers/chat.py。
[下游关系] 确保 HTTP 映射契约稳定。
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
from uae_law_rag.backend.api.routers.chat import router as chat_router
from uae_law_rag.backend.schemas.ids import new_uuid
from uae_law_rag.backend.utils.constants import (
    DEBUG_KEY,
    EVALUATION_RECORD_ID_KEY,
    GENERATION_RECORD_ID_KEY,
    REQUEST_ID_KEY,
    RETRIEVAL_RECORD_ID_KEY,
    TIMING_MS_KEY,
    TRACE_ID_KEY,
)


pytestmark = pytest.mark.fastapi_gate


class _MilvusStub:
    """Milvus stub for router gate."""  # docstring: avoid Milvus access


@pytest.mark.asyncio
async def test_chat_router_gate(session: AsyncSession, monkeypatch: pytest.MonkeyPatch) -> None:
    """
    [职责] 验证 chat router 输出字段映射与 debug 结构。
    [边界] 不覆盖 pipeline；仅验证 HTTP 映射。
    [上游关系] /chat router。
    [下游关系] 前端依赖此响应结构。
    """
    trace_id = new_uuid()
    request_id = new_uuid()

    async def _override_session() -> AsyncIterator[AsyncSession]:
        yield session  # docstring: reuse test session

    def _override_milvus_repo() -> _MilvusStub:
        return _MilvusStub()  # docstring: stub Milvus repo

    async def _fake_chat(**_kwargs):
        return {
            "conversation_id": str(new_uuid()),
            "message_id": str(new_uuid()),
            "kb_id": str(new_uuid()),
            "status": "success",
            "answer": "ok",
            "citations": [
                {
                    "node_id": str(new_uuid()),
                    "rank": 1,
                    "quote": "quote",
                }
            ],
            "evaluator": {"status": "pass", "rule_version": "v0", "warnings": []},
            TIMING_MS_KEY: {"total_ms": 8.5},
            TRACE_ID_KEY: str(trace_id),
            REQUEST_ID_KEY: str(request_id),
            DEBUG_KEY: {
                RETRIEVAL_RECORD_ID_KEY: str(new_uuid()),
                GENERATION_RECORD_ID_KEY: str(new_uuid()),
                EVALUATION_RECORD_ID_KEY: str(new_uuid()),
                "gate": {
                    "retrieval": {"passed": True},
                    "generation": {"status": "success"},
                    "evaluator": {"status": "pass"},
                },
            },
        }  # docstring: stub service output

    monkeypatch.setattr(
        "uae_law_rag.backend.api.routers.chat.chat",
        _fake_chat,
    )  # docstring: patch service call

    app = FastAPI()
    app.add_middleware(TraceContextMiddleware)  # docstring: inject trace/request headers
    app.include_router(chat_router)  # docstring: mount chat router
    app.dependency_overrides[get_session] = _override_session  # docstring: override session dep
    app.dependency_overrides[get_milvus_repo] = _override_milvus_repo  # docstring: override Milvus dep

    payload = {
        "query": "hello",
        "conversation_id": str(new_uuid()),
        "kb_id": str(new_uuid()),
        "debug": False,
    }

    transport = ASGITransport(app=app)  # docstring: ASGI transport for httpx
    headers = {
        "x-trace-id": str(trace_id),
        "x-request-id": str(request_id),
    }  # docstring: explicit trace/request headers
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post(
            "/chat?debug=true",
            json=payload,
            headers=headers,
        )  # docstring: invoke chat endpoint

    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "success"
    assert data["answer"] == "ok"
    assert data["citations"] and data["citations"][0]["quote"] == "quote"
    assert data["evaluator"]["status"] == "pass"
    assert "timing_ms" in data
    assert data["trace_id"] == str(trace_id)
    assert data["request_id"] == str(request_id)
    assert data["debug"]["records"][RETRIEVAL_RECORD_ID_KEY]
    assert data["debug"]["gate"]["retrieval"]["passed"] is True
    assert "x-trace-id" in resp.headers  # docstring: header must include trace_id
    assert "x-request-id" in resp.headers  # docstring: header must include request_id
    assert resp.headers["x-trace-id"] == str(trace_id)  # docstring: trace_id must propagate
    assert resp.headers["x-request-id"] == str(request_id)  # docstring: request_id must propagate
