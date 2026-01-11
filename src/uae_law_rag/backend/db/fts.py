# src/uae_law_rag/backend/db/fts.py

"""
[职责] SQLite FTS（全文检索）最小实现：为 NodeModel.text 提供关键词检索能力（全量召回的 SQL 侧底座）。
[边界] 当前仅实现 SQLite FTS5；PostgreSQL tsvector/GIN 作为生产替换点（后续扩展）。
[上游关系] ingest pipeline 写入 NodeModel；本模块通过触发器同步 FTS 索引。
[下游关系] retrieval/keyword.py 调用 search_nodes() 获取候选 nodes；fusion/rerank 在其基础上进行融合排序。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession


# --- SQLite FTS table design (minimal & robust) ---
# 说明：
# - NodeModel.id 是 UUID(str)，不适合作为 SQLite rowid；因此 FTS 表使用独立主键 node_id。
# - 为保证一致性：使用 INSERT/UPDATE/DELETE triggers 同步 node_fts。
# - 仅索引 node.text（以及可扩展字段）；过滤（kb_id/file_id/page）走 join + where。


FTS_TABLE = "node_fts"  # docstring: FTS 虚表名（SQLite FTS5）


@dataclass(frozen=True)
class KeywordHit:
    """Keyword search hit (DB-side)."""  # docstring: keyword stage 的最小返回结构

    node_id: str  # docstring: 命中节点ID（UUID str）
    score: float  # docstring: BM25 分数（越小越相关，后续可转换/归一化）
    snippet: str  # docstring: 关键词命中片段（用于 UI/debug，可空）
    meta: Dict[str, Any]  # docstring: 结构化元信息（kb/doc/file/page/article_id 等）


async def ensure_sqlite_fts(session: AsyncSession) -> None:
    """
    Ensure SQLite FTS5 structures exist.

    Creates:
      - node_fts virtual table
      - triggers to sync from node table
    """  # docstring: 应在 app startup 或 ingest 前调用一次
    # Enable FK for current connection (SQLite-specific). Not harmful if already enabled.
    await session.execute(text("PRAGMA foreign_keys=ON;"))  # docstring: SQLite FK 开启

    # 1) Create FTS virtual table
    # UNINDEXED node_id 仍可存储/查询，但不参与倒排索引；text 才是索引字段。
    await session.execute(
        text(
            f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS {FTS_TABLE}
            USING fts5(
              node_id UNINDEXED,
              text,
              tokenize = 'unicode61'
            );
            """
        )
    )

    # 2) Create triggers to keep FTS in sync with node table
    # NOTE: node table名假设为 "node"（与你的 NodeModel.__tablename__ 一致）
    await session.execute(
        text(
            f"""
            CREATE TRIGGER IF NOT EXISTS node_ai AFTER INSERT ON node BEGIN
              INSERT INTO {FTS_TABLE}(node_id, text) VALUES (new.id, new.text);
            END;
            """
        )
    )

    await session.execute(
        text(
            f"""
            CREATE TRIGGER IF NOT EXISTS node_ad AFTER DELETE ON node BEGIN
              DELETE FROM {FTS_TABLE} WHERE node_id = old.id;
            END;
            """
        )
    )

    await session.execute(
        text(
            f"""
            CREATE TRIGGER IF NOT EXISTS node_au AFTER UPDATE OF text ON node BEGIN
              UPDATE {FTS_TABLE} SET text = new.text WHERE node_id = new.id;
            END;
            """
        )
    )

    await session.commit()  # docstring: DDL/trigger 需要提交以生效


async def rebuild_sqlite_fts(session: AsyncSession) -> None:
    """
    Rebuild FTS index from existing node table.

    Use cases:
      - legacy data existed before triggers
      - bulk import with triggers disabled (not recommended)
    """  # docstring: 运维/修复工具
    await session.execute(text(f"DELETE FROM {FTS_TABLE};"))  # docstring: 清空索引
    await session.execute(
        text(
            f"""
            INSERT INTO {FTS_TABLE}(node_id, text)
            SELECT id, text FROM node;
            """
        )
    )
    await session.commit()  # docstring: 写入索引提交


async def search_nodes_sqlite(
    session: AsyncSession,
    *,
    kb_id: str,
    query: str,
    top_k: int,
    file_id: Optional[str] = None,
) -> List[KeywordHit]:
    """
    SQLite FTS search for nodes within a KB (and optional file scope).

    Returns:
      - node_id
      - bm25 score
      - snippet
      - meta (kb/doc/file/page/article_id/section_path)
    """  # docstring: keyword 全量召回的入口（DB-level）
    if not query.strip():
        return []

    # Join path:
    # node -> document -> knowledge_base
    # 你们的 doc schema：NodeModel(document_id) -> DocumentModel(kb_id, file_id)
    sql = f"""
    SELECT
      n.id AS node_id,
      bm25({FTS_TABLE}) AS score,
      snippet({FTS_TABLE}, 1, '[', ']', '…', 20) AS snippet,
      d.kb_id AS kb_id,
      d.id AS document_id,
      d.file_id AS file_id,
      n.page AS page,
      n.article_id AS article_id,
      n.section_path AS section_path
    FROM {FTS_TABLE}
    JOIN node n ON n.id = {FTS_TABLE}.node_id
    JOIN document d ON d.id = n.document_id
    WHERE {FTS_TABLE} MATCH :q
      AND d.kb_id = :kb_id
    """

    params: Dict[str, Any] = {
        "q": query,
        "kb_id": kb_id,
        "limit": int(top_k),
    }

    if file_id:
        sql += " AND d.file_id = :file_id"
        params["file_id"] = file_id

    # FTS5 排序：bm25 越小越相关，ASC
    sql += " ORDER BY score ASC LIMIT :limit"

    rows = (await session.execute(text(sql), params)).mappings().all()

    hits: List[KeywordHit] = []
    for r in rows:
        meta = {
            "kb_id": r["kb_id"],  # docstring: KB 作用域
            "document_id": r["document_id"],  # docstring: 文档ID
            "file_id": r["file_id"],  # docstring: 文件ID
            "page": r["page"],  # docstring: 页码（可空）
            "article_id": r["article_id"],  # docstring: 法条编号（可空）
            "section_path": r["section_path"],  # docstring: 结构路径（可空）
        }
        hits.append(
            KeywordHit(
                node_id=r["node_id"],
                score=float(r["score"] or 0.0),
                snippet=str(r["snippet"] or ""),
                meta=meta,
            )
        )
    return hits


async def search_nodes(
    session: AsyncSession,
    *,
    kb_id: str,
    query: str,
    top_k: int = 200,
    file_id: Optional[str] = None,
    dialect: str = "sqlite",
) -> List[KeywordHit]:
    """
    Dialect router for keyword search.

    Current:
      - sqlite: FTS5
    Future:
      - postgres: tsvector/GIN
    """  # docstring: 为未来 PostgreSQL 保留扩展点
    if dialect != "sqlite":
        raise NotImplementedError("Only sqlite FTS5 is implemented in MVP")
    return await search_nodes_sqlite(session, kb_id=kb_id, query=query, top_k=top_k, file_id=file_id)
