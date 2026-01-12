# src/uae_law_rag/backend/pipelines/ingest/pipeline.py

"""
[职责] ingest pipeline：编排 PDF→Markdown→Node→Embedding→Milvus→DB 的完整导入闭环。
[边界] 不做 API 级事务提交；不实现队列/重试；不负责 Milvus collection 生命周期管理。
[上游关系] services/scripts 调用入口 run_ingest_pdf；依赖 PipelineContext 与 IngestRepo/MilvusRepo。
[下游关系] retrieval/generation/evaluator 依赖 Node/VectorMap 与 provider/timing 可回放数据。
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from uae_law_rag.backend.db.models.doc import DocumentModel, KnowledgeFileModel
from uae_law_rag.backend.pipelines.base.context import PipelineContext

from . import embed as embed_mod
from . import pdf_parse as pdf_parse_mod
from . import persist_db as persist_db_mod
from . import persist_milvus as persist_milvus_mod
from . import segment as segment_mod


@dataclass(frozen=True)
class IngestResult:
    """
    [职责] IngestResult：导入结果快照（供 gate/tests/services 调用）。
    [边界] 不携带大文本与全部向量；仅返回关键引用与计数。
    [上游关系] run_ingest_pdf 产出。
    [下游关系] ingest_gate/服务层可用于状态展示与回放。
    """

    kb_id: str
    file_id: str
    document_id: str
    node_count: int
    vector_count: int
    sha256: str
    pages: Optional[int] = None
    sample_query_vector: Optional[List[float]] = None  # docstring: gate 可选 smoke search
    trace_id: Optional[str] = None  # docstring: pipeline trace id
    request_id: Optional[str] = None  # docstring: pipeline request id
    timing_ms: Dict[str, float] = None  # type: ignore[assignment]  # docstring: timing 快照
    provider_snapshot: Dict[str, Any] = None  # type: ignore[assignment]  # docstring: provider 快照


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
    milvus_repo: Any,
    milvus_collection: str,
    ctx: Optional[PipelineContext] = None,
) -> IngestResult:
    """
    [职责] run_ingest_pdf：执行单次 PDF 导入（幂等、可观测、可回放）。
    [边界] 不创建 KB；不做事务提交；不做队列化。
    [上游关系] services 或脚本调用；上游提供 KB/Milvus 依赖。
    [下游关系] 写入 file/document/node/vector_map；Milvus 向量可用于检索。
    """
    ctx = ctx or PipelineContext.from_session(session)  # docstring: 统一 ctx 装配
    ingest_repo = ctx.ingest_repo  # docstring: SQL repo 入口

    pdf_file = Path(pdf_path).expanduser().resolve()  # docstring: 标准化路径
    if not pdf_file.exists():
        raise FileNotFoundError(str(pdf_file))  # docstring: 文件必须存在

    _file_name = file_name or pdf_file.name  # docstring: 文件名快照
    _source_uri = source_uri or f"file://{_file_name}"  # docstring: 源 URI 兜底
    file_ext = _safe_ext(pdf_file.name)  # docstring: 扩展名归一化

    with ctx.timing.stage("sha256"):
        sha256 = _sha256_file(pdf_file)  # docstring: 幂等 key

    with ctx.timing.stage("idempotency"):
        existing = await ingest_repo.get_file_by_sha256(kb_id=kb_id, sha256=sha256)  # docstring: 幂等判定
    if existing is not None and existing.ingest_status == "success":
        document_id = await _get_document_id_by_file_id(session=session, file_id=existing.id)  # docstring: 回查文档
        return IngestResult(
            kb_id=kb_id,
            file_id=existing.id,
            document_id=document_id,
            node_count=int(existing.node_count or 0),
            vector_count=int(existing.node_count or 0),
            sha256=sha256,
            pages=existing.pages,
            sample_query_vector=None,
            trace_id=str(ctx.trace_id),
            request_id=str(ctx.request_id),
            timing_ms=ctx.timing_ms(),
            provider_snapshot=dict(ctx.provider_snapshot),
        )

    with ctx.timing.stage("load_kb"):
        kb = await ingest_repo.get_kb(kb_id)  # docstring: 加载 KB 配置
    if kb is None:
        raise ValueError(f"KB not found: {kb_id}")  # docstring: KB 必须存在

    file_row: Optional[KnowledgeFileModel] = None
    entities: List[Dict[str, Any]] = []
    vector_ids: List[str] = []
    nodes_out: List[Any] = []
    parsed_pages: Optional[int] = None

    try:
        with ctx.timing.stage("db", accumulate=True):
            file_row = await persist_db_mod.create_file(
                repo=ingest_repo,
                kb_id=kb_id,
                file_name=_file_name,
                sha256=sha256,
                source_uri=_source_uri,
                file_ext=file_ext,
                file_version=1,
                file_mtime=_safe_mtime(pdf_file),
                file_size=_safe_size(pdf_file),
                pages=None,
                ingest_profile={
                    "parser": parser_name,
                    "parse_version": parse_version,
                    "segment_version": segment_version,
                },
            )  # docstring: 创建文件记录（pending）

        ctx.with_provider(
            "parser", {"name": parser_name, "parse_version": parse_version}
        )  # docstring: 记录 parser 快照
        with ctx.timing.stage("parse"):
            parsed = await pdf_parse_mod.parse_pdf(
                pdf_path=str(pdf_file), parser_name=parser_name, parse_version=parse_version
            )  # docstring: PDF → Markdown
        parsed_pages = _infer_pages(parsed)  # docstring: 解析页数
        if parsed_pages is not None and file_row is not None:
            file_row.pages = parsed_pages  # docstring: 回填页数
            await session.flush()  # docstring: 持久化页数

        ctx.meta["segment_version"] = segment_version  # docstring: 记录切分版本
        with ctx.timing.stage("segment"):
            node_dicts = await segment_mod.segment_nodes(
                parsed=parsed, chunking_config=getattr(kb, "chunking_config", {}) or {}, segment_version=segment_version
            )  # docstring: Markdown → Node payloads
        if not node_dicts:
            raise ValueError("segment produced no nodes")  # docstring: 禁止空节点集合

        ctx.with_provider(
            "embed",
            {
                "provider": getattr(kb, "embed_provider", "ollama"),
                "model": getattr(kb, "embed_model", ""),
                "dim": getattr(kb, "embed_dim", None),
            },
        )  # docstring: 记录 embedding 快照
        with ctx.timing.stage("embed"):
            embeddings = await embed_mod.embed_texts(
                texts=[n["text"] for n in node_dicts],
                provider=getattr(kb, "embed_provider", "ollama"),
                model=getattr(kb, "embed_model", ""),
                dim=getattr(kb, "embed_dim", None),
            )  # docstring: 生成 embedding
        if len(embeddings) != len(node_dicts):
            raise ValueError("embedding count mismatch")  # docstring: 向量数量必须与节点一致

        with ctx.timing.stage("db", accumulate=True):
            doc, nodes_out = await persist_db_mod.persist_document_nodes(
                repo=ingest_repo,
                kb_id=kb_id,
                file_id=file_row.id if file_row else "",
                nodes=node_dicts,
                title=_file_name,
                source_name=_file_name,
                meta_data={"parser": parser_name, "parse_version": parse_version},
            )  # docstring: 落库 Document + Nodes

        for i, node in enumerate(nodes_out):
            payload = node_dicts[i]  # docstring: 通过索引对齐 payload
            page = payload.get("page")
            if page is None:
                page = getattr(node, "page", None)  # docstring: 回退到 DB 节点页码
            if page is None and parsed_pages == 1:
                page = 1  # docstring: 单页文档兜底 page=1

            vector_id = _new_vector_id(kb_id=kb_id, node_id=node.id)  # docstring: 稳定向量主键
            vector_ids.append(vector_id)
            entities.append(
                {
                    "vector_id": vector_id,
                    "embedding": embeddings[i],
                    "node_id": node.id,
                    "kb_id": kb_id,
                    "file_id": file_row.id if file_row else "",
                    "document_id": doc.id,
                    "page": page,
                    "article_id": getattr(node, "article_id", None) or payload.get("article_id") or "",
                    "section_path": getattr(node, "section_path", None) or payload.get("section_path") or "",
                }
            )  # docstring: Milvus payload（与 schema 对齐）

        with ctx.timing.stage("milvus"):
            await persist_milvus_mod.upsert(
                milvus_repo=milvus_repo,
                collection=milvus_collection,
                entities=entities,
                embed_dim=getattr(kb, "embed_dim", None),
            )  # docstring: 写入 Milvus 向量实体

        maps = [{"node_id": nodes_out[i].id, "vector_id": vector_ids[i]} for i in range(len(nodes_out))]
        with ctx.timing.stage("db", accumulate=True):
            await persist_db_mod.persist_node_vector_maps(
                repo=ingest_repo, kb_id=kb_id, file_id=file_row.id if file_row else "", maps=maps
            )  # docstring: 写入 node↔vector 映射

        with ctx.timing.stage("db", accumulate=True):
            await persist_db_mod.mark_file_ingested(
                repo=ingest_repo,
                file_id=file_row.id if file_row else "",
                node_count=len(nodes_out),
                last_ingested_at=datetime.now(timezone.utc),
            )  # docstring: 标记文件导入成功

        sample_vec = embeddings[0] if embeddings else None  # docstring: 返回 sample 向量
        return IngestResult(
            kb_id=kb_id,
            file_id=file_row.id if file_row else "",
            document_id=doc.id,
            node_count=len(nodes_out),
            vector_count=len(vector_ids),
            sha256=sha256,
            pages=parsed_pages,
            sample_query_vector=sample_vec,
            trace_id=str(ctx.trace_id),
            request_id=str(ctx.request_id),
            timing_ms=ctx.timing_ms(),
            provider_snapshot=dict(ctx.provider_snapshot),
        )
    except Exception:
        if file_row is not None:
            with ctx.timing.stage("db", accumulate=True):
                await persist_db_mod.mark_file_failed(
                    repo=ingest_repo,
                    file_id=file_row.id,
                    node_count=len(nodes_out),
                    last_ingested_at=datetime.now(timezone.utc),
                )  # docstring: 标记失败便于回滚与观测
        if entities and hasattr(milvus_repo, "delete_by_expr"):
            expr = f"file_id == '{file_row.id if file_row else ''}'"  # docstring: 按 file_id 清理向量
            await milvus_repo.delete_by_expr(collection=milvus_collection, expr=expr)  # docstring: best-effort 清理
        raise


def _sha256_file(path: Path) -> str:
    """
    [职责] 计算文件 sha256（幂等 key）。
    [边界] 不读取整文件到内存；按块流式计算。
    [上游关系] run_ingest_pdf 调用。
    [下游关系] KnowledgeFile.sha256。
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
    [下游关系] KnowledgeFile.file_ext。
    """
    ext = os.path.splitext(name)[1].lstrip(".").lower().strip()
    return ext or "pdf"


def _safe_mtime(p: Path) -> float:
    """
    [职责] 获取文件 mtime（尽量不抛异常）。
    [边界] 文件系统可能不可用；失败则返回 0。
    [上游关系] run_ingest_pdf 调用。
    [下游关系] KnowledgeFile.file_mtime。
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
    [下游关系] KnowledgeFile.file_size。
    """
    try:
        return int(p.stat().st_size)
    except Exception:
        return 0


def _new_vector_id(*, kb_id: str, node_id: str) -> str:
    """
    [职责] 生成稳定的 vector_id（Milvus PK）。
    [边界] 使用 hash(kb_id:node_id) 派生；不依赖 Milvus auto_id。
    [上游关系] run_ingest_pdf 生成 Milvus payload。
    [下游关系] NodeVectorMapModel.vector_id / Milvus PK。
    """
    raw = f"{kb_id}:{node_id}".encode("utf-8")
    return hashlib.sha1(raw).hexdigest()[:36]


def _infer_pages(parsed: Any) -> Optional[int]:
    """
    [职责] 从 parsed 对象推断页数。
    [边界] 推断失败返回 None。
    [上游关系] run_ingest_pdf 调用。
    [下游关系] file.pages 与 Milvus page 兜底。
    """
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


async def _get_document_id_by_file_id(*, session: AsyncSession, file_id: str) -> str:
    """
    [职责] 根据 file_id 查找 document_id（幂等返回路径）。
    [边界] MVP 假设 file 对应单 document；找不到则抛错。
    [上游关系] run_ingest_pdf 幂等分支调用。
    [下游关系] IngestResult.document_id。
    """
    stmt = select(DocumentModel).where(DocumentModel.file_id == file_id).order_by(DocumentModel.created_at.asc())
    doc = (await session.execute(stmt)).scalars().first()
    if doc is None:
        raise ValueError(f"document not found for file_id={file_id}")  # docstring: DB 状态异常
    return doc.id
