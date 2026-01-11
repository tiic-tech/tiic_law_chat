# playground/sql_gate/test_fts_gate.py

"""
[职责] fts gate：验证 SQLite FTS5 表/触发器可创建，并能对 Node.text 做关键词检索。
[边界] 只测试 SQL 侧 FTS；不涉及 Milvus；不做 fusion/rerank。
[上游关系] 依赖 db/fts.py + ingest repo 写入 node 表。
[下游关系] retrieval/keyword.py 将依赖 search_nodes() 返回候选集合。
"""

from __future__ import annotations

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from uae_law_rag.backend.db.fts import FTS_TABLE, ensure_sqlite_fts, search_nodes
from uae_law_rag.backend.db.repo import ConversationRepo, IngestRepo, MessageRepo, UserRepo


pytestmark = pytest.mark.sql_gate


@pytest.mark.asyncio
async def test_fts_ensure_creates_table_and_triggers(session: AsyncSession) -> None:
    """Ensure FTS table and triggers exist."""  # docstring: DDL/trigger 创建验证
    await ensure_sqlite_fts(session)

    # Check virtual table
    row = (
        await session.execute(text("SELECT name FROM sqlite_master WHERE type='table' AND name=:n"), {"n": FTS_TABLE})
    ).fetchone()
    assert row is not None

    # Check triggers
    trigger_rows = (
        await session.execute(
            text("SELECT name FROM sqlite_master WHERE type='trigger' AND name IN ('node_ai','node_ad','node_au')")
        )
    ).fetchall()
    trigger_names = {r[0] for r in trigger_rows}
    assert {"node_ai", "node_ad", "node_au"}.issubset(trigger_names)


@pytest.mark.asyncio
async def test_fts_search_hits_after_ingest(session: AsyncSession) -> None:
    """
    Insert nodes, verify FTS triggers sync, then search_nodes returns hits.
    """  # docstring: “关键词全量召回”底座验证
    await ensure_sqlite_fts(session)

    user_repo = UserRepo(session)
    ingest_repo = IngestRepo(session)
    conv_repo = ConversationRepo(session)
    msg_repo = MessageRepo(session)

    u = await user_repo.create(username="fts_u")
    kb = await ingest_repo.create_kb(
        user_id=u.id,
        kb_name="fts_kb",
        milvus_collection="fts_collection",
        embed_model="bge-m3",
        embed_dim=1024,
    )
    c = await conv_repo.create(user_id=u.id, chat_type="chat", default_kb_id=kb.id, settings={})
    _m = await msg_repo.create_user_message(conversation_id=c.id, chat_type="chat", query="q")

    f = await ingest_repo.create_file(
        kb_id=kb.id,
        file_name="fts.pdf",
        file_ext="pdf",
        sha256="b" * 64,
        source_uri="file://fts.pdf",
        file_version=1,
        file_mtime=0.0,
        file_size=10,
        pages=1,
        ingest_profile={"parser": "pymupdf4llm"},
    )
    doc = await ingest_repo.create_document(
        kb_id=kb.id,
        file_id=f.id,
        title="FTS Test Doc",
        source_name="fts.pdf",
        meta_data={},
    )
    nodes = await ingest_repo.bulk_create_nodes(
        document_id=doc.id,
        nodes=[
            {
                "node_index": 0,
                "text": "Article 2: Scope of application for real beneficiary procedures.",
                "page": 1,
                "start_offset": 0,
                "end_offset": 70,
                "article_id": "Article 2",
                "section_path": "Chapter 1",
                "meta_data": {},
            },
            {
                "node_index": 1,
                "text": "Article 3: Competent authority and obligations.",
                "page": 1,
                "start_offset": 71,
                "end_offset": 120,
                "article_id": "Article 3",
                "section_path": "Chapter 1",
                "meta_data": {},
            },
        ],
    )

    # Commit so that triggers & FTS rows are definitely persisted for subsequent queries.
    await session.commit()

    hits = await search_nodes(session, kb_id=kb.id, query="Scope", top_k=10)
    assert len(hits) >= 1
    assert any(h.node_id == nodes[0].id for h in hits)

    # bm25: smaller is better; we just assert score is numeric
    assert all(isinstance(h.score, float) for h in hits)

    # snippet should be a string (may include brackets if snippet() formatting applied)
    assert isinstance(hits[0].snippet, str)
