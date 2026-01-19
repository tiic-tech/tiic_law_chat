# src/uae_law_rag/backend/pipelines/ingest/persist_db.py

"""
[职责] persist_db：负责 ingest 产物的 SQL 落库（file/document/node/node_vector_map）。
[边界] 不做 PDF 解析/切分/embedding；不提交事务；不处理 Milvus。
[上游关系] ingest/pipeline.py 或 service 层调用；依赖 IngestRepo 与节点/向量映射产物。
[下游关系] retrieval pipeline 依赖 Node/NodeVectorMap 做检索与一致性校验。
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence

from uae_law_rag.backend.db.models.doc import DocumentModel, KnowledgeFileModel, NodeModel, NodeVectorMapModel
from uae_law_rag.backend.db.repo.ingest_repo import IngestRepo
from uae_law_rag.backend.utils.constants import META_DATA_KEY


_ARTICLE_RE = re.compile(r"\bArticle\s*[\(\[]?\s*(\d+)\s*[\)\]]?\b", flags=re.IGNORECASE)


def _extract_article_id(s: Optional[str]) -> Optional[str]:
    if not s:
        return None
    m = _ARTICLE_RE.search(str(s))
    if not m:
        return None
    num = m.group(1)
    return f"Article {num}" if num else None


def _coerce_int(v: Any) -> Optional[int]:
    if v is None:
        return None
    if isinstance(v, bool):
        return None
    if isinstance(v, int):
        return v
    s = str(v).strip()
    if not s:
        return None
    try:
        return int(s)
    except Exception:
        return None


def _coerce_page(payload: dict) -> Optional[int]:
    # accept multiple upstream keys to avoid silent drift
    for k in ("page", "page_no", "page_number", "pageIndex", "page_idx"):
        if k in payload:
            p = _coerce_int(payload.get(k))
            if p is None or p <= 0:
                return None
            return p
    return None


def _normalize_node_payloads(nodes: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    [职责] 归一化节点 payload（对齐 NodeModel 所需字段）。
    [边界] 不补充业务语义；仅保证字段存在与类型可用。
    [上游关系] persist_nodes 调用。
    [下游关系] IngestRepo.bulk_create_nodes。
    """
    normalized: List[Dict[str, Any]] = []
    for i, n in enumerate(nodes):
        text_raw = str(n.get("text") or "")  # docstring: 节点文本（keyword 召回基础）
        if not text_raw.strip():
            raise ValueError("node.text is required")  # docstring: 禁止空节点
        node_index = int(n.get("node_index", i))  # docstring: node_index 兜底

        # --- page: tolerate upstream key drift + enforce int/positive ---
        page = _coerce_page(n)

        # --- article_id: keep upstream if present; else derive from section_path/text ---
        section_path = n.get("section_path")
        article_id = n.get("article_id")
        if not (article_id and str(article_id).strip()):
            article_id = _extract_article_id(str(section_path or "")) or _extract_article_id(text_raw)

        normalized.append(
            {
                "node_index": node_index,
                "text": text_raw,
                "page": page,
                "article_id": article_id,
                "section_path": section_path,
                "start_offset": _coerce_int(n.get("start_offset")),
                "end_offset": _coerce_int(n.get("end_offset")),
                "page_start_offset": _coerce_int(n.get("page_start_offset")),
                "page_end_offset": _coerce_int(n.get("page_end_offset")),
                META_DATA_KEY: n.get(META_DATA_KEY) or {},
            }
        )

    _validate_node_indices(normalized)  # docstring: 校验节点序号连续性
    return normalized


def _validate_node_indices(nodes: Sequence[Dict[str, Any]]) -> None:
    """
    [职责] 校验 node_index 连续性（从 0 递增）。
    [边界] 仅校验顺序与重复；不修复。
    [上游关系] _normalize_node_payloads 调用。
    [下游关系] gate 断言前置保障。
    """
    if not nodes:
        return
    indices = [int(n["node_index"]) for n in nodes]  # docstring: 保持原顺序校验
    if len(set(indices)) != len(indices):
        raise ValueError("node_index must be unique")  # docstring: 禁止重复
    expected = list(range(len(indices)))  # docstring: 期望 0..N-1
    if indices != expected:
        raise ValueError("node_index must be continuous from 0")  # docstring: 保证稳定排序


def _normalize_maps(maps: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    [职责] 归一化 node↔vector 映射 payload。
    [边界] 不校验 Milvus 存在性；仅保证字段存在。
    [上游关系] persist_node_vector_maps 调用。
    [下游关系] IngestRepo.bulk_create_node_vector_maps。
    """
    normalized: List[Dict[str, Any]] = []
    for m in maps:
        node_id = str(m.get("node_id") or "")  # docstring: 节点ID
        vector_id = str(m.get("vector_id") or "")  # docstring: Milvus 主键
        if not node_id or not vector_id:
            raise ValueError("node_id/vector_id are required")  # docstring: 映射字段必填
        normalized.append(
            {
                "node_id": node_id,
                "vector_id": vector_id,
                META_DATA_KEY: m.get(META_DATA_KEY) or {},
            }
        )
    return normalized


async def create_file(
    *,
    repo: IngestRepo,
    kb_id: str,
    file_name: str,
    sha256: str,
    source_uri: Optional[str] = None,
    file_ext: Optional[str] = None,
    file_version: int = 1,
    file_mtime: float = 0.0,
    file_size: int = 0,
    pages: Optional[int] = None,
    ingest_profile: Optional[Dict[str, Any]] = None,
) -> KnowledgeFileModel:
    """
    [职责] 创建 KnowledgeFile 记录（pending 状态）。
    [边界] 不做幂等判定；不提交事务。
    [上游关系] ingest/pipeline.py 调用（入库起点）。
    [下游关系] Document/Node/NodeVectorMap 归属该 file。
    """
    return await repo.create_file(
        kb_id=kb_id,  # docstring: KB 归属
        file_name=file_name,  # docstring: 文件名
        sha256=sha256,  # docstring: 内容指纹
        source_uri=source_uri,  # docstring: 源 URI
        file_ext=file_ext,  # docstring: 扩展名
        file_version=file_version,  # docstring: 文件版本
        file_mtime=file_mtime,  # docstring: 文件 mtime
        file_size=file_size,  # docstring: 文件 size
        pages=pages,  # docstring: 页数
        ingest_profile=ingest_profile or {},  # docstring: 导入配置快照
    )


async def create_document(
    *,
    repo: IngestRepo,
    kb_id: str,
    file_id: str,
    title: Optional[str] = None,
    source_name: Optional[str] = None,
    meta_data: Optional[Dict[str, Any]] = None,
) -> DocumentModel:
    """
    [职责] 创建 Document 记录。
    [边界] 不推断标题；不提交事务。
    [上游关系] ingest/pipeline.py 或 persist_document_nodes 调用。
    [下游关系] NodeModel 归属该 document。
    """
    return await repo.create_document(
        kb_id=kb_id,  # docstring: KB 归属
        file_id=file_id,  # docstring: 文件归属
        title=title,  # docstring: 文档标题
        source_name=source_name,  # docstring: 展示名
        meta_data=meta_data or {},  # docstring: 文档元数据
    )


async def persist_nodes(
    *,
    repo: IngestRepo,
    document_id: str,
    nodes: Sequence[Dict[str, Any]],
) -> List[NodeModel]:
    """
    [职责] 批量写入 Node 记录。
    [边界] 不做 embedding；不提交事务。
    [上游关系] ingest/segment.py 产出 node payload。
    [下游关系] NodeModel 为 retrieval/generation 提供证据。
    """
    normalized = _normalize_node_payloads(nodes)  # docstring: 对齐 NodeModel 字段
    return await repo.bulk_create_nodes(document_id=document_id, nodes=normalized)  # docstring: 批量落库


async def persist_node_vector_maps(
    *,
    repo: IngestRepo,
    kb_id: str,
    file_id: str,
    maps: Sequence[Dict[str, Any]],
) -> List[NodeVectorMapModel]:
    """
    [职责] 批量写入 node↔vector 映射记录。
    [边界] 不校验 Milvus 状态；不提交事务。
    [上游关系] ingest/persist_milvus.py 完成 upsert 后调用。
    [下游关系] NodeVectorMapModel 支撑向量回查与一致性检查。
    """
    normalized = _normalize_maps(maps)  # docstring: 对齐映射字段
    return await repo.bulk_create_node_vector_maps(kb_id=kb_id, file_id=file_id, maps=normalized)


async def persist_document_nodes(
    *,
    repo: IngestRepo,
    kb_id: str,
    file_id: str,
    nodes: Sequence[Dict[str, Any]],
    title: Optional[str] = None,
    source_name: Optional[str] = None,
    meta_data: Optional[Dict[str, Any]] = None,
) -> tuple[DocumentModel, List[NodeModel]]:
    """
    [职责] 创建 Document 并批量写入 Nodes（常用组合步骤）。
    [边界] 不提交事务；不处理向量映射。
    [上游关系] ingest/pipeline.py 调用。
    [下游关系] NodeModel 为后续 embedding/检索提供证据。
    """
    doc = await create_document(  # docstring: 创建文档记录
        repo=repo,
        kb_id=kb_id,
        file_id=file_id,
        title=title,
        source_name=source_name,
        meta_data=meta_data,
    )
    nodes_out = await persist_nodes(repo=repo, document_id=doc.id, nodes=nodes)  # docstring: 批量写入节点
    return doc, nodes_out


async def mark_file_ingested(
    *,
    repo: IngestRepo,
    file_id: str,
    node_count: int,
    last_ingested_at: Optional[datetime] = None,
) -> bool:
    """
    [职责] 将文件标记为 ingest success。
    [边界] 不提交事务；不更新向量映射。
    [上游关系] ingest/pipeline.py 成功路径调用。
    [下游关系] ingest_gate 与 UI 依赖 ingest_status。
    """
    return await repo.mark_file_ingested(
        file_id,
        status="success",  # docstring: 成功状态
        node_count=int(node_count),  # docstring: 节点数量统计
        last_ingested_at=last_ingested_at or datetime.now(timezone.utc),  # docstring: 完成时间
    )


async def mark_file_failed(
    *,
    repo: IngestRepo,
    file_id: str,
    node_count: int = 0,
    last_ingested_at: Optional[datetime] = None,
) -> bool:
    """
    [职责] 将文件标记为 ingest failed。
    [边界] 不提交事务；不清理已写入节点。
    [上游关系] ingest/pipeline.py 异常路径调用。
    [下游关系] gate tests / UI 用于失败可观察性。
    """
    return await repo.mark_file_ingested(
        file_id,
        status="failed",  # docstring: 失败状态
        node_count=int(node_count),  # docstring: 失败时的节点统计
        last_ingested_at=last_ingested_at or datetime.now(timezone.utc),  # docstring: 失败时间
    )
