# playground/fastapi_gate/routers/test_records_router_gate.py

"""
[职责] Records router gate：验证 /records 回放接口输出结构。
[边界] 使用 repo stub；不依赖真实 DB 结构或 pipeline。
[上游关系] backend/api/routers/records.py。
[下游关系] 前端 EvidencePanel 回放依赖该契约。
"""

from __future__ import annotations

import sys
from types import SimpleNamespace
from pathlib import Path
from typing import AsyncIterator, List

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SRC_ROOT = _REPO_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))  # docstring: ensure local src import

from uae_law_rag.backend.api.deps import get_session
from uae_law_rag.backend.api.middleware import TraceContextMiddleware
from uae_law_rag.backend.api.routers.records import router as records_router
from uae_law_rag.backend.schemas.ids import new_uuid


pytestmark = pytest.mark.fastapi_gate


@pytest.mark.asyncio
async def test_records_router_gate(session: AsyncSession, monkeypatch: pytest.MonkeyPatch) -> None:
    """
    [职责] 覆盖 retrieval/generation/evaluation 三个回放接口。
    [边界] 仅验证 HTTP 映射与 header 透传。
    [上游关系] /records 路由。
    [下游关系] 前端审计面板依赖此结构。
    """
    retrieval_record_id = str(new_uuid())
    generation_record_id = str(new_uuid())
    evaluation_record_id = str(new_uuid())

    retrieval_record = SimpleNamespace(
        id=retrieval_record_id,
        message_id=str(new_uuid()),
        kb_id=str(new_uuid()),
        query_text="query",
        keyword_top_k=5,
        vector_top_k=3,
        fusion_top_k=3,
        rerank_top_k=2,
        fusion_strategy="union",
        rerank_strategy="none",
        provider_snapshot={"embed": {"provider": "hash"}},
        timing_ms={"total_ms": 1.0},
    )  # docstring: retrieval record stub

    hit = SimpleNamespace(
        node_id=str(new_uuid()),
        source="keyword",
        rank=1,
        score=0.9,
        excerpt="excerpt",
        page=1,
        start_offset=0,
        end_offset=10,
    )  # docstring: retrieval hit stub

    generation_record = SimpleNamespace(
        id=generation_record_id,
        message_id=str(new_uuid()),
        status="success",
        output_raw="answer",
        citations={
            "version": "v1",
            "nodes": [str(new_uuid())],
            "items": [{"node_id": str(new_uuid()), "rank": 1, "quote": "quote"}],
        },
    )  # docstring: generation record stub

    evaluation_record = SimpleNamespace(
        id=evaluation_record_id,
        message_id=str(new_uuid()),
        status="pass",
        rule_version="v0",
        checks={"items": [{"name": "rule1", "status": "pass", "message": "ok"}]},
    )  # docstring: evaluation record stub

    class _RetrievalRepoStub:
        def __init__(self, _session: AsyncSession) -> None:
            self._session = _session  # docstring: stub session

        async def get_record(self, _record_id: str):
            return retrieval_record  # docstring: return stub record

        async def list_hits(self, _record_id: str) -> List[object]:
            return [hit]  # docstring: return stub hits

    class _GenerationRepoStub:
        def __init__(self, _session: AsyncSession) -> None:
            self._session = _session  # docstring: stub session

        async def get_record(self, _record_id: str):
            return generation_record  # docstring: return stub record

    class _EvaluatorRepoStub:
        def __init__(self, _session: AsyncSession) -> None:
            self._session = _session  # docstring: stub session

        async def get_record(self, _record_id: str):
            return evaluation_record  # docstring: return stub record

    monkeypatch.setattr(
        "uae_law_rag.backend.api.routers.records.RetrievalRepo",
        _RetrievalRepoStub,
    )  # docstring: patch retrieval repo
    monkeypatch.setattr(
        "uae_law_rag.backend.api.routers.records.GenerationRepo",
        _GenerationRepoStub,
    )  # docstring: patch generation repo
    monkeypatch.setattr(
        "uae_law_rag.backend.api.routers.records.EvaluatorRepo",
        _EvaluatorRepoStub,
    )  # docstring: patch evaluator repo

    async def _override_session() -> AsyncIterator[AsyncSession]:
        yield session  # docstring: reuse test session

    app = FastAPI()
    app.add_middleware(TraceContextMiddleware)  # docstring: inject trace/request headers
    app.include_router(records_router)  # docstring: mount records router
    app.dependency_overrides[get_session] = _override_session  # docstring: override session dep

    trace_id = str(new_uuid())
    request_id = str(new_uuid())
    headers = {"x-trace-id": trace_id, "x-request-id": request_id}  # docstring: trace headers

    transport = ASGITransport(app=app)  # docstring: ASGI transport for httpx
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        retrieval_resp = await client.get(f"/records/retrieval/{retrieval_record_id}", headers=headers)
        generation_resp = await client.get(f"/records/generation/{generation_record_id}", headers=headers)
        evaluation_resp = await client.get(f"/records/evaluation/{evaluation_record_id}", headers=headers)

    assert retrieval_resp.status_code == 200
    retrieval_data = retrieval_resp.json()
    assert retrieval_data["retrieval_record_id"] == retrieval_record_id
    assert retrieval_data["hits"] and retrieval_data["hits"][0]["node_id"]

    assert generation_resp.status_code == 200
    generation_data = generation_resp.json()
    assert generation_data["generation_record_id"] == generation_record_id
    assert generation_data["citations"] is not None

    assert evaluation_resp.status_code == 200
    evaluation_data = evaluation_resp.json()
    assert evaluation_data["evaluation_record_id"] == evaluation_record_id
    assert evaluation_data["checks_summary"] is not None

    assert retrieval_resp.headers["x-trace-id"] == trace_id  # docstring: trace_id must propagate
    assert retrieval_resp.headers["x-request-id"] == request_id  # docstring: request_id must propagate
    assert generation_resp.headers["x-trace-id"] == trace_id  # docstring: trace_id must propagate
    assert generation_resp.headers["x-request-id"] == request_id  # docstring: request_id must propagate
    assert evaluation_resp.headers["x-trace-id"] == trace_id  # docstring: trace_id must propagate
    assert evaluation_resp.headers["x-request-id"] == request_id  # docstring: request_id must propagate
