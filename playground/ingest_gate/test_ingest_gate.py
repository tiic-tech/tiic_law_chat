# playground/ingest_gate/test_ingest_gate.py

from __future__ import annotations

import os
import time
from dotenv import load_dotenv
from pathlib import Path

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from uae_law_rag.backend.db.repo import IngestRepo, UserRepo

from uae_law_rag.backend.kb.schema import build_collection_spec, build_expr_for_scope


pytestmark = pytest.mark.ingest_gate

load_dotenv()


def _milvus_env_ok() -> bool:
    uri = os.getenv("MILVUS_URI", "").strip()
    host = os.getenv("MILVUS_HOST", "").strip()
    port = os.getenv("MILVUS_PORT", "").strip()
    return bool(uri) or (bool(host) and bool(port))


def _sample_pdf_path() -> Path:
    p = os.getenv("UAE_LAW_RAG_SAMPLE_PDF", "").strip()
    if p:
        return Path(p).expanduser().resolve()
    return Path("playground/assets/sample.pdf").resolve()


@pytest.mark.asyncio
async def test_ingest_gate_pdf_to_db_and_milvus(session: AsyncSession) -> None:
    """
    Contract:
      - ingest once: creates file/document/nodes/node_vector_maps + upserts Milvus entities
      - ingest twice (same kb + same pdf content): idempotent (no new nodes/vectors)
    """
    if not _milvus_env_ok():
        pytest.skip("Milvus env not configured. Set MILVUS_URI or MILVUS_HOST/MILVUS_PORT.")

    pdf_path = _sample_pdf_path()
    if not pdf_path.exists():
        pytest.skip(f"Sample PDF not found: {pdf_path}. Set UAE_LAW_RAG_SAMPLE_PDF or add playground/assets/sample.pdf")

    from uae_law_rag.backend.kb.client import MilvusClient  # type: ignore
    from uae_law_rag.backend.kb.index import MilvusIndexManager  # type: ignore
    from uae_law_rag.backend.kb.repo import MilvusRepo  # type: ignore

    from uae_law_rag.backend.pipelines.ingest.pipeline import run_ingest_pdf  # type: ignore

    user_repo = UserRepo(session)
    ingest_repo = IngestRepo(session)

    u = await user_repo.create(username=f"ingest_u_{int(time.time())}", password_hash=None, is_active=True)
    kb = await ingest_repo.create_kb(
        user_id=u.id,
        kb_name=f"ingest_kb_{int(time.time())}",
        milvus_collection=f"ingest_collection_{int(time.time())}",
        embed_model="bge-m3",
        embed_dim=1024,
        chunking_config={"chunk_size": 800, "overlap": 100},
    )

    spec = build_collection_spec(
        name=kb.milvus_collection,
        embed_dim=kb.embed_dim,
        metric_type="COSINE",
        index_type="HNSW",
        default_top_k=50,
    )

    client = MilvusClient.from_env()
    await client.create_collection(spec, drop_if_exists=True)
    idx = MilvusIndexManager(client)
    await idx.ensure_index(spec)
    await idx.load_collection(spec.name)
    milvus_repo = MilvusRepo(client)

    # ---- ingest #1 ----
    r1 = await run_ingest_pdf(
        session=session,
        kb_id=kb.id,
        pdf_path=str(pdf_path),
        file_name=pdf_path.name,
        source_uri=f"file://{pdf_path.name}",
        parser_name="pymupdf4llm",
        parse_version="v1",
        segment_version="v1",
        milvus_repo=milvus_repo,
        milvus_collection=kb.milvus_collection,
    )

    assert r1.kb_id == kb.id
    assert r1.file_id
    assert r1.document_id
    assert r1.node_count > 0
    assert r1.vector_count > 0

    # DB checks
    f = await ingest_repo.get_file(r1.file_id)
    assert f is not None
    assert f.kb_id == kb.id
    assert f.file_ext == "pdf"
    assert f.sha256 and len(f.sha256) == 64

    doc = await ingest_repo.get_document(r1.document_id)
    assert doc is not None
    assert doc.kb_id == kb.id
    assert doc.file_id == r1.file_id

    nodes = await ingest_repo.list_nodes_by_document(r1.document_id)
    assert len(nodes) == r1.node_count
    assert any(getattr(n, "article_id", None) for n in nodes)

    maps = await ingest_repo.list_node_vector_maps_by_file(r1.file_id)
    assert len(maps) == r1.vector_count

    # Milvus smoke search (optional but recommended for stability if pipeline returns a sample vector)
    expr = build_expr_for_scope(kb_id=kb.id)
    if getattr(r1, "sample_query_vector", None) is not None:
        res = await milvus_repo.search(
            collection=kb.milvus_collection,
            query_vectors=[r1.sample_query_vector],
            top_k=3,
            expr=expr,
            output_fields=spec.search.output_fields,
        )
        assert len(res) == 1
        assert len(res[0]) >= 1
        assert res[0][0]["payload"]["kb_id"] == kb.id

    # ---- ingest #2 (idempotent) ----
    r2 = await run_ingest_pdf(
        session=session,
        kb_id=kb.id,
        pdf_path=str(pdf_path),
        file_name=pdf_path.name,
        source_uri=f"file://{pdf_path.name}",
        parser_name="pymupdf4llm",
        parse_version="v1",
        segment_version="v1",
        milvus_repo=milvus_repo,
        milvus_collection=kb.milvus_collection,
    )

    assert r2.file_id == r1.file_id
    assert r2.document_id == r1.document_id
    assert r2.node_count == 0
    assert r2.vector_count == 0

    await idx.release_collection(spec.name)
    await client.drop_collection(spec.name)
