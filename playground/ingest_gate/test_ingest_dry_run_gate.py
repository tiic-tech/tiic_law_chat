# playground/ingest_gate/test_ingest_dry_run_gate.py

"""
[Responsibility] Ingest dry_run gate: verify dry_run avoids persistence and returns deterministic file_id.
[Boundary] Use monkeypatch to stub parse/segment/embed to avoid external deps and Milvus writes.
[Upstream] ingest_service.ingest_file is the entrypoint.
[Downstream] Keep dry_run semantics stable without DB/Milvus pollution.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import pytest
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from uae_law_rag.backend.db.models.doc import (
    DocumentModel,
    KnowledgeFileModel,
    NodeModel,
    NodeVectorMapModel,
)
from uae_law_rag.backend.db.repo import IngestRepo, UserRepo
from uae_law_rag.backend.pipelines.ingest import embed as embed_mod
from uae_law_rag.backend.pipelines.ingest import pdf_parse as pdf_parse_mod
from uae_law_rag.backend.pipelines.ingest import segment as segment_mod
from uae_law_rag.backend.services.ingest_service import ingest_file


pytestmark = pytest.mark.ingest_gate


class _MilvusStub:
    """Milvus stub for dry_run (should never be called)."""  # docstring: dry_run should not touch Milvus

    def __getattr__(self, name: str) -> Any:
        raise AssertionError(f"milvus method should not be called: {name}")  # docstring: forbid calls


async def _count_rows(session: AsyncSession, model: Any) -> int:
    """
    [Responsibility] Count rows for a model.
    [Boundary] Test-only helper; no data mutation.
    [Upstream] dry_run gate test calls.
    [Downstream] Asserts DB stays unchanged.
    """
    stmt = select(func.count()).select_from(model)  # docstring: COUNT(*) query
    return int((await session.execute(stmt)).scalar_one())  # docstring: return count


@pytest.mark.asyncio
async def test_ingest_dry_run_no_persist(
    session: AsyncSession,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    [Responsibility] Validate dry_run avoids persistence, returns deterministic file_id, and correct node_count.
    [Boundary] Covers dry_run only; does not validate Milvus/FTS.
    [Upstream] ingest_service.ingest_file.
    [Downstream] Ensures dry_run does not pollute DB.
    """

    async def _fake_parse_pdf(*_args: Any, **_kwargs: Any) -> Dict[str, Any]:
        return {"pages": 1, "content": "stub"}  # docstring: fake parse output

    async def _fake_segment_nodes(*_args: Any, **_kwargs: Any) -> List[Dict[str, Any]]:
        return [
            {"node_index": 0, "text": "A", "page": 1},
            {"node_index": 1, "text": "B", "page": 1},
        ]  # docstring: fake segment output

    async def _fake_embed_texts(*_args: Any, **_kwargs: Any) -> List[List[float]]:
        return [[0.0, 0.0], [0.0, 0.0]]  # docstring: fake embedding output

    monkeypatch.setattr(pdf_parse_mod, "parse_pdf", _fake_parse_pdf)  # docstring: stub parser
    monkeypatch.setattr(segment_mod, "segment_nodes", _fake_segment_nodes)  # docstring: stub segmenter
    monkeypatch.setattr(embed_mod, "embed_texts", _fake_embed_texts)  # docstring: stub embedder

    pdf_path = tmp_path / "dry_run.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 dry run")  # docstring: create temp PDF file

    user_repo = UserRepo(session)  # docstring: user repo
    ingest_repo = IngestRepo(session)  # docstring: ingest repo
    u = await user_repo.create(username="dry_run_user")  # docstring: create test user
    kb = await ingest_repo.create_kb(
        user_id=u.id,
        kb_name="dry_run_kb",
        milvus_collection="dry_run_collection",
        embed_model="hash",
        embed_dim=2,
        embed_provider="hash",
        chunking_config={"enable_sentence_window": False},
    )  # docstring: create KB config
    await session.commit()  # docstring: commit KB row

    counts_before = {
        "file": await _count_rows(session, KnowledgeFileModel),
        "doc": await _count_rows(session, DocumentModel),
        "node": await _count_rows(session, NodeModel),
        "map": await _count_rows(session, NodeVectorMapModel),
    }  # docstring: record counts before dry_run

    res1 = await ingest_file(
        session=session,
        kb_id=str(kb.id),
        source_uri=str(pdf_path),
        file_name="dry_run.pdf",
        ingest_profile=None,
        milvus_repo=_MilvusStub(),
        trace_context=None,
        dry_run=True,
        debug=False,
    )  # docstring: first dry_run call
    res2 = await ingest_file(
        session=session,
        kb_id=str(kb.id),
        source_uri=str(pdf_path),
        file_name="dry_run.pdf",
        ingest_profile=None,
        milvus_repo=_MilvusStub(),
        trace_context=None,
        dry_run=True,
        debug=False,
    )  # docstring: second dry_run call (deterministic id)

    assert res1["status"] == "success"
    assert res1["node_count"] == 2
    assert res1["timing_ms"]["dry_run"] is True
    assert res1["file_id"] == res2["file_id"]

    # dry_run must not leave any pending ORM state
    assert len(session.new) == 0
    assert len(session.dirty) == 0
    assert len(session.deleted) == 0

    counts_after = {
        "file": await _count_rows(session, KnowledgeFileModel),
        "doc": await _count_rows(session, DocumentModel),
        "node": await _count_rows(session, NodeModel),
        "map": await _count_rows(session, NodeVectorMapModel),
    }  # docstring: record counts after dry_run
    assert counts_before == counts_after
