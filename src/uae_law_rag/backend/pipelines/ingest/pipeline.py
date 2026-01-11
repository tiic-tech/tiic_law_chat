# src/uae_law_rag/backend/pipelines/ingest/pipeline.py

"""
[职责] Ingest pipeline 编排层：端到端导入 PDF（parse -> segment -> persist_db -> embed -> persist_milvus），并输出可审计结果。
[边界] 不实现具体解析算法/切分算法/embedding 模型推理/Milvus client 细节；仅负责把各子步骤按合同串起来，并实现幂等策略。
[上游关系] services 或脚本触发 ingest；依赖 db.repo.IngestRepo、kb.repo.MilvusRepo、ingest 子模块（pdf_parse/segment/embed/persist_*）。
[下游关系] DB 产物用于 FTS keyword 检索与证据回查；Milvus 向量产物用于 vector 检索；后续 retrieval/generation 依赖 node_id/vector_id 稳定性。
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, List, Optional, Sequence

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from uae_law_rag.backend.db.repo.ingest_repo import IngestRepo
from uae_law_rag.backend.db.models.doc import DocumentModel

# MilvusRepo 只在运行时注入，避免在无 Milvus 环境下 import 失败导致 gate 之外的测试崩溃
# from uae_law_rag.backend.kb.repo import MilvusRepo


@dataclass(frozen=True)
class IngestResult:
    """
    [职责] IngestResult：导入结果快照（供 gate tests / services / 调试回放使用）。
    [边界] 不携带大文本；不携带全部 embedding；只返回关键引用与计数。
    [上游关系] run_ingest_pdf 产出。
    [下游关系] ingest_gate 断言；services 可据此展示状态或写入审计。
    """

    kb_id: str
    file_id: str
    document_id: str
    node_count: int
    vector_count: int
    sha256: str
    pages: Optional[int] = None
    sample_query_vector: Optional[List[float]] = None  # docstring: 供 gate 做 Milvus smoke search（可选）


# -----------------------------
# Public entrypoint
# -----------------------------


async def run_ingest_pdf(
    *,
    session: AsyncSession,
    kb_id: str,
    pdf_path: str,
    file_name: Optional[str] = None,
    source_uri: Optional[str] = None,
    parser_name: str = "pymupdf4llm",
    parse_version: str = "v1",
    segment_version: str = "v1",
    milvus_repo: Any,  # expected: kb.repo.MilvusRepo
    milvus_collection: str,
) -> IngestResult:
    """
    [职责] run_ingest_pdf：导入单个 PDF 的主编排入口（幂等、落库、Milvus upsert）。
    [边界] 不负责创建 KB；不负责选择 collection spec；不负责 retry/队列化；仅完成一次“同步导入闭环”。
    [上游关系] ingest_gate / services 调用；上游提供 kb_id 与 MilvusRepo/collection。
    [下游关系] 写入 knowledge_file/document/node/node_vector_map；写入 Milvus 向量实体；返回 IngestResult。
    """
    # --- normalize inputs ---
    pdf_file = Path(pdf_path).expanduser().resolve()
    if not pdf_file.exists():
        raise FileNotFoundError(str(pdf_file))

    _file_name = file_name or pdf_file.name
    _source_uri = source_uri or f"file://{_file_name}"
    file_ext = _safe_ext(pdf_file.name)

    # --- deps ---
    ingest_repo = IngestRepo(session)

    # --- compute sha256 (idempotency key) ---
    sha256 = _sha256_file(pdf_file)

    # --- idempotent check: same KB + sha256 => treat as same file in MVP ---
    existing = await ingest_repo.get_file_by_sha256(kb_id=kb_id, sha256=sha256)
    if existing is not None:
        # For idempotent return, we still need document_id. Prefer the first document for this file.
        document_id = await _get_document_id_by_file_id(session=session, file_id=existing.id)

        # Mark as success is optional; we keep DB unchanged by default in idempotent path.
        return IngestResult(
            kb_id=kb_id,
            file_id=existing.id,
            document_id=document_id,
            node_count=0,
            vector_count=0,
            sha256=sha256,
            pages=existing.pages,
            sample_query_vector=None,
        )

    # --- load KB config (embed_dim/model/provider, chunking_config...) ---
    kb = await ingest_repo.get_kb(kb_id)
    if kb is None:
        raise ValueError(f"KB not found: {kb_id}")

    # --- Step 1: create file row (pending) ---
    f = await ingest_repo.create_file(
        kb_id=kb_id,
        file_name=_file_name,
        file_ext=file_ext,
        sha256=sha256,
        source_uri=_source_uri,
        file_version=1,
        file_mtime=_safe_mtime(pdf_file),
        file_size=_safe_size(pdf_file),
        pages=None,  # will fill if parser returns pages
        ingest_profile={
            "parser": parser_name,
            "parse_version": parse_version,
            "segment_version": segment_version,
        },
    )

    # --- Step 2: parse pdf -> raw units (pages/blocks) ---
    parsed = await _call_pdf_parse(
        pdf_path=str(pdf_file),
        parser_name=parser_name,
        parse_version=parse_version,
    )
    pages = _infer_pages(parsed)
    if pages is not None:
        f.pages = pages  # docstring: 回填页数快照（可选）

    # --- Step 3: segment -> node drafts (list[dict]) ---
    node_dicts = await _call_segment(
        parsed=parsed,
        chunking_config=getattr(kb, "chunking_config", {}) or {},
        segment_version=segment_version,
    )

    # --- Step 4: persist_db (document + nodes) ---
    # title/source_name 可在 parsed 中提取；MVP 先用文件名
    doc = await ingest_repo.create_document(
        kb_id=kb_id,
        file_id=f.id,
        title=_file_name,
        source_name=_file_name,
        meta_data={"parser": parser_name, "parse_version": parse_version},
    )
    nodes = await ingest_repo.bulk_create_nodes(document_id=doc.id, nodes=node_dicts)

    # --- Step 5: embed nodes (list[list[float]]) ---
    texts: List[str] = [n.text for n in nodes]
    embeddings = await _call_embed(
        texts=texts,
        embed_provider=getattr(kb, "embed_provider", "ollama"),
        embed_model=getattr(kb, "embed_model", ""),
        embed_dim=getattr(kb, "embed_dim", None),
    )

    if len(embeddings) != len(nodes):
        raise ValueError(f"embedding count mismatch: {len(embeddings)} != {len(nodes)}")

    # --- Step 6: persist_milvus (upsert entities + map vector_id) ---
    vector_ids: List[str] = []
    entities: List[dict] = []

    for i, n in enumerate(nodes):
        vector_id = _new_vector_id(kb_id=kb_id, node_id=n.id)
        vector_ids.append(vector_id)
        entities.append(
            {
                "vector_id": vector_id,
                "embedding": embeddings[i],
                "node_id": n.id,
                "kb_id": kb_id,
                "file_id": f.id,
                "document_id": doc.id,
                "page": getattr(n, "page", None),
                "article_id": getattr(n, "article_id", None) or "",
                "section_path": getattr(n, "section_path", None) or "",
            }
        )

    # upsert into Milvus
    await _call_persist_milvus(
        milvus_repo=milvus_repo,
        collection=milvus_collection,
        entities=entities,
    )

    # persist node_vector_map in SQL
    maps = [{"node_id": nodes[i].id, "vector_id": vector_ids[i]} for i in range(len(nodes))]
    await ingest_repo.bulk_create_node_vector_maps(kb_id=kb_id, file_id=f.id, maps=maps)

    # --- Step 7: mark file ingested ---
    await ingest_repo.mark_file_ingested(
        f.id,
        status="success",
        node_count=len(nodes),
        last_ingested_at=datetime.now(timezone.utc),
    )

    # Note: session commit is controlled by caller fixture/service; we only flush in repo methods.

    sample_vec = embeddings[0] if embeddings else None
    return IngestResult(
        kb_id=kb_id,
        file_id=f.id,
        document_id=doc.id,
        node_count=len(nodes),
        vector_count=len(nodes),
        sha256=sha256,
        pages=pages,
        sample_query_vector=sample_vec,
    )


# -----------------------------
# Private helpers (adapters)
# -----------------------------


def _sha256_file(path: Path) -> str:
    """
    [职责] 计算文件 sha256（幂等 key）。
    [边界] 不读取整文件到内存；按块流式计算。
    [上游关系] run_ingest_pdf 调用。
    [下游关系] get_file_by_sha256 幂等判定；KnowledgeFile.sha256 落库。
    """
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(1024 * 1024)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _safe_ext(name: str) -> str:
    """
    [职责] 归一化扩展名。
    [边界] 仅做字符串处理。
    [上游关系] run_ingest_pdf 调用。
    [下游关系] KnowledgeFile.file_ext 落库。
    """
    ext = os.path.splitext(name)[1].lstrip(".").lower().strip()
    return ext or "pdf"


def _safe_mtime(p: Path) -> float:
    """
    [职责] 获取文件 mtime（尽量不抛异常）。
    [边界] 文件系统可能不可用；失败则返回 0。
    [上游关系] run_ingest_pdf 调用。
    [下游关系] KnowledgeFile.file_mtime 落库。
    """
    try:
        return float(p.stat().st_mtime)
    except Exception:
        return 0.0


def _safe_size(p: Path) -> int:
    """
    [职责] 获取文件 size（尽量不抛异常）。
    [边界] 文件系统可能不可用；失败则返回 0。
    [上游关系] run_ingest_pdf 调用。
    [下游关系] KnowledgeFile.file_size 落库。
    """
    try:
        return int(p.stat().st_size)
    except Exception:
        return 0


async def _get_document_id_by_file_id(*, session: AsyncSession, file_id: str) -> str:
    """
    [职责] 根据 file_id 找到对应 document_id（幂等返回需要）。
    [边界] MVP 假设一个 file 对应一个 document；多文档场景后续扩展。
    [上游关系] run_ingest_pdf 幂等分支调用。
    [下游关系] IngestResult.document_id 返回；后续 retrieval/generation 可能用于作用域/审计。
    """
    stmt = select(DocumentModel).where(DocumentModel.file_id == file_id).order_by(DocumentModel.created_at.asc())
    doc = (await session.execute(stmt)).scalars().first()
    if doc is None:
        # MVP fallback: if no document found, this indicates an inconsistent DB state; fail fast.
        raise ValueError(f"document not found for file_id={file_id}")
    return doc.id


def _new_vector_id(*, kb_id: str, node_id: str) -> str:
    """
    [职责] 生成稳定的 vector_id（Milvus PK），确保可复现与去重。
    [边界] MVP 用 hash(kb_id:node_id) 派生；不依赖 Milvus auto_id（你 schema_gate 已锁 auto_id=False）。
    [上游关系] run_ingest_pdf 在写入 Milvus 前生成。
    [下游关系] Milvus payload.vector_id；SQL node_vector_map.vector_id。
    """
    raw = f"{kb_id}:{node_id}".encode("utf-8")
    return hashlib.sha1(raw).hexdigest()[:36]  # docstring: 36 chars to look like UUID-ish; stable & deterministic


async def _call_pdf_parse(*, pdf_path: str, parser_name: str, parse_version: str) -> Any:
    """
    [职责] 调用 pdf_parse 子模块（适配层）。
    [边界] 不假设具体 ParsedPdf 类型；只要求返回对象可被 segment 消费，且可推断 pages（可选）。
    [上游关系] run_ingest_pdf。
    [下游关系] _call_segment。
    """
    from . import pdf_parse as mod

    # Preferred function names (implement in pdf_parse.py)
    if hasattr(mod, "parse_pdf"):
        return await mod.parse_pdf(pdf_path=pdf_path, parser_name=parser_name, parse_version=parse_version)  # type: ignore[misc]

    if hasattr(mod, "parse"):
        return await mod.parse(pdf_path=pdf_path, parser_name=parser_name, parse_version=parse_version)  # type: ignore[misc]

    raise AttributeError("pdf_parse module must provide async parse_pdf(...) or parse(...)")


def _infer_pages(parsed: Any) -> Optional[int]:
    """
    [职责] 从 parsed 对象推断页数（尽量兼容不同 parser 输出）。
    [边界] 推断失败返回 None。
    [上游关系] run_ingest_pdf。
    [下游关系] KnowledgeFile.pages 快照与 IngestResult.pages。
    """
    # common patterns: parsed.pages, parsed["pages"], parsed["page_count"]
    try:
        v = getattr(parsed, "pages", None)
        if isinstance(v, int):
            return v
    except Exception:
        pass
    try:
        if isinstance(parsed, dict):
            for k in ("pages", "page_count", "num_pages"):
                if isinstance(parsed.get(k), int):
                    return int(parsed[k])
    except Exception:
        pass
    return None


async def _call_segment(*, parsed: Any, chunking_config: dict, segment_version: str) -> List[dict]:
    """
    [职责] 调用 segment 子模块，把 parsed 输出切分为 Node dict 列表（对齐 IngestRepo.bulk_create_nodes 输入）。
    [边界] 只要求输出 dict keys 满足 bulk_create_nodes 所需；不规定具体算法。
    [上游关系] _call_pdf_parse。
    [下游关系] IngestRepo.bulk_create_nodes。
    """
    from . import segment as mod

    if hasattr(mod, "segment_nodes"):
        nodes = await mod.segment_nodes(parsed=parsed, chunking_config=chunking_config, segment_version=segment_version)  # type: ignore[misc]
        return list(nodes)

    if hasattr(mod, "segment"):
        nodes = await mod.segment(parsed=parsed, chunking_config=chunking_config, segment_version=segment_version)  # type: ignore[misc]
        return list(nodes)

    raise AttributeError("segment module must provide async segment_nodes(...) or segment(...)")


async def _call_embed(
    *, texts: Sequence[str], embed_provider: str, embed_model: str, embed_dim: Any
) -> List[List[float]]:
    """
    [职责] 调用 embed 子模块，对 texts 生成 embedding 向量。
    [边界] 不关心 provider 细节；只要求输出 list[list[float]]，长度与 texts 一致。
    [上游关系] NodeModel.text 列表。
    [下游关系] Milvus upsert payload.embedding。
    """
    from . import embed as mod

    if hasattr(mod, "embed_texts"):
        vecs = await mod.embed_texts(texts=list(texts), provider=embed_provider, model=embed_model, dim=embed_dim)  # type: ignore[misc]
        return list(vecs)

    if hasattr(mod, "embed"):
        vecs = await mod.embed(texts=list(texts), provider=embed_provider, model=embed_model, dim=embed_dim)  # type: ignore[misc]
        return list(vecs)

    raise AttributeError("embed module must provide async embed_texts(...) or embed(...)")


async def _call_persist_milvus(*, milvus_repo: Any, collection: str, entities: List[dict]) -> None:
    """
    [职责] 调用 persist_milvus 子模块或直接调用 repo.upsert_embeddings 完成写入。
    [边界] 不关心索引/加载生命周期；由上游 gate/服务保证 collection ready。
    [上游关系] embedding 产物 + payload keys（kb/schema 常量已锁）。
    [下游关系] Milvus 向量检索的数据底座。
    """
    # Prefer a dedicated module hook if you want to add retries/logging
    from . import persist_milvus as mod

    if hasattr(mod, "upsert"):
        await mod.upsert(milvus_repo=milvus_repo, collection=collection, entities=entities)  # type: ignore[misc]
        return

    # fallback: call repo directly (as per milvus_gate contract)
    if hasattr(milvus_repo, "upsert_embeddings"):
        await milvus_repo.upsert_embeddings(collection=collection, entities=entities)
        return

    raise AttributeError("persist_milvus must provide upsert(...) or milvus_repo must have upsert_embeddings(...)")
