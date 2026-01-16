# src/uae_law_rag/backend/pipelines/retrieval/vector.py

"""
[职责] vector recall：基于 Milvus 向量检索召回语义相关证据，并映射为统一 Candidate 结构。
[边界] 仅执行向量检索与分数归一化；不做 keyword/fusion/rerank；不负责落库与编排。
[上游关系] retrieval pipeline 传入 query_vector/kb_scope 与 MilvusRepo；依赖 kb/schema 的 scope 表达式。
[下游关系] fusion/rerank/persist 消费 Candidate 列表用于排序与审计落库。
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Sequence, cast

from uae_law_rag.backend.kb.repo import MilvusRepo
from uae_law_rag.backend.kb.schema import (
    ARTICLE_ID_FIELD,
    DOCUMENT_ID_FIELD,
    FILE_ID_FIELD,
    KB_ID_FIELD,
    NODE_ID_FIELD,
    PAGE_FIELD,
    SECTION_PATH_FIELD,
    build_expr_for_scope,
)
from uae_law_rag.backend.pipelines.retrieval.types import Candidate, _coerce_int


MetricType = Literal["IP", "L2", "COSINE"]  # docstring: 向量度量类型（与 Milvus metric 对齐）

DEFAULT_OUTPUT_FIELDS = [
    NODE_ID_FIELD,
    KB_ID_FIELD,
    FILE_ID_FIELD,
    DOCUMENT_ID_FIELD,
    PAGE_FIELD,
    ARTICLE_ID_FIELD,
    SECTION_PATH_FIELD,
]  # docstring: 默认 payload 字段（证据与过滤需要）


def _normalize_metric_type(metric_type: Optional[str]) -> MetricType:
    """
    [职责] 归一化 metric_type（统一为大写）。
    [边界] 仅支持 IP/L2/COSINE；未知值回退为 COSINE。
    [上游关系] vector_recall 调用。
    [下游关系] _normalize_vector_score 与 score_details。
    """
    mt = str(metric_type or "COSINE").strip().upper()  # docstring: 统一大写
    if mt in {"IP", "L2", "COSINE"}:
        return cast(MetricType, mt)  # docstring: 支持的 metric
    return cast(MetricType, "COSINE")  # docstring: 未知 metric 回退


def _normalize_output_fields(output_fields: Optional[Sequence[str]]) -> List[str]:
    """
    [职责] 归一化 output_fields，确保包含 node_id。
    [边界] 不校验字段是否存在于 collection；只保证字段集合稳定。
    [上游关系] vector_recall 调用。
    [下游关系] milvus_repo.search 的 output_fields。
    """
    fields = list(output_fields) if output_fields else list(DEFAULT_OUTPUT_FIELDS)  # docstring: 兜底字段
    if NODE_ID_FIELD not in fields:
        fields.insert(0, NODE_ID_FIELD)  # docstring: 强制 node_id 回查字段
    # de-dup while preserving order
    seen: set[str] = set()  # docstring: 去重缓存
    ordered: List[str] = []
    for f in fields:
        key = str(f).strip()
        if not key or key in seen:
            continue
        seen.add(key)
        ordered.append(key)
    return ordered


def _normalize_scope_ids(kb_scope: Dict[str, Any]) -> Dict[str, Optional[str]]:
    """
    [职责] 规范化 kb_scope（kb_id/file_id/document_id）。
    [边界] 必须包含 kb_id；file_id/document_id 可为空。
    [上游关系] vector_recall 调用。
    [下游关系] build_expr_for_scope 构造过滤表达式。
    """
    kb_id = str(kb_scope.get("kb_id") or "").strip()  # docstring: KB 作用域
    if not kb_id:
        raise ValueError("kb_id is required for vector recall")  # docstring: 强制 KB 过滤
    file_id = str(kb_scope.get("file_id") or "").strip() or None  # docstring: 文件作用域（可选）
    document_id = str(kb_scope.get("document_id") or "").strip() or None  # docstring: 文档作用域（可选）
    return {"kb_id": kb_id, "file_id": file_id, "document_id": document_id}


def _resolve_collection(collection: Optional[str], kb_scope: Dict[str, Any]) -> str:
    """
    [职责] 解析 Milvus collection 名称（显式参数优先）。
    [边界] 若无法解析则抛错；不做环境默认值推断。
    [上游关系] vector_recall 调用。
    [下游关系] milvus_repo.search 的 collection 参数。
    """
    if collection:
        return str(collection).strip()  # docstring: 显式 collection 优先
    fallback = kb_scope.get("collection") or kb_scope.get("milvus_collection")  # docstring: 兼容字段
    name = str(fallback or "").strip()
    if not name:
        raise ValueError("milvus collection is required for vector recall")  # docstring: collection 必填
    return name


def _normalize_page(value: Any) -> Optional[int]:
    """
    [职责] 归一化页码为 int。
    [边界] 仅处理 int/float/数字字符串；其他返回 None。
    [上游关系] _hit_to_candidate 调用。
    [下游关系] Candidate.page。
    """
    if value is None:
        return None
    if isinstance(value, int):
        return None if value == 0 else value  # docstring: 0 作为未知页码 sentinel，统一归一为 None
    if isinstance(value, float):
        v = int(value)
        return None if v == 0 else v
    if isinstance(value, str) and value.strip().isdigit():
        v = int(value.strip())
        return None if v == 0 else v
    return None


def _normalize_vector_score(raw_score: float, metric_type: MetricType) -> float:
    """
    [职责] 将向量距离/相似度统一为“越大越好”的分数。
    [边界] 仅做单调转换；不跨候选归一化。
    [上游关系] _hit_to_candidate 调用。
    [下游关系] Candidate.score 用于后续 fusion/rerank。
    """
    if metric_type == "L2":
        dist = float(raw_score) if raw_score >= 0 else 0.0  # docstring: L2 距离非负
        return 1.0 / (1.0 + dist)  # docstring: 距离 -> 相似度
    return float(raw_score)  # docstring: IP/COSINE 直接使用


def _hit_to_candidate(
    hit: Dict[str, Any],
    *,
    metric_type: MetricType,
) -> Optional[Candidate]:
    """
    [职责] 将 Milvus hit 映射为统一 Candidate 结构。
    [边界] 若缺失 node_id 则丢弃；excerpt 不可用时为 None。
    [上游关系] vector_recall 调用。
    [下游关系] 返回 Candidate 供 fusion/rerank/persist 使用。
    """
    payload = hit.get("payload") or {}  # docstring: Milvus payload
    node_id = payload.get(NODE_ID_FIELD) or payload.get("node_id")  # docstring: 证据节点ID
    if not node_id:
        return None  # docstring: 无 node_id 无法回查证据

    vector_id = str(hit.get("vector_id") or "")  # docstring: 向量主键
    _raw = hit.get("score")  # docstring: 原始向量分数（可能为空）
    raw_score = float(_raw) if _raw is not None else None
    norm_score = (
        _normalize_vector_score(float(raw_score), metric_type) if raw_score is not None else 0.0
    )  # docstring: 缺失分数降级

    meta = dict(payload)  # docstring: 透传 payload 元数据
    if vector_id:
        meta["vector_id"] = vector_id  # docstring: 附加 vector_id 便于审计

    score_details = {
        "raw_score": raw_score,
        "score_norm": norm_score,
        "metric_type": metric_type,
        "vector_id": vector_id,
    }  # docstring: 可解释分数细节

    return Candidate(
        node_id=str(node_id),  # docstring: 节点ID
        stage="vector",  # docstring: 标记 vector 阶段
        score=norm_score,  # docstring: 归一化分数
        score_details=score_details,  # docstring: 分数细节快照
        excerpt=None,  # docstring: 向量召回无 snippet
        page=_normalize_page(payload.get(PAGE_FIELD)),  # docstring: 页码快照
        start_offset=_coerce_int(payload.get("start_offset")),  # docstring: 起始偏移（如有）
        end_offset=_coerce_int(payload.get("end_offset")),  # docstring: 结束偏移（如有）
        meta=meta,  # docstring: 透传 meta
    )


async def vector_recall(
    *,
    milvus_repo: MilvusRepo,
    kb_scope: Dict[str, Any],
    query_vector: List[float],
    top_k: int,
    output_fields: Optional[List[str]] = None,
    metric_type: Optional[str] = None,
    collection: Optional[str] = None,
) -> List[Candidate]:
    """
    [职责] vector_recall：执行 Milvus 向量检索并产出 Candidate 列表。
    [边界] 不做 keyword/fusion/rerank；不落库；仅依赖 MilvusRepo。
    [上游关系] retrieval pipeline 传入 query_vector/kb_scope/top_k。
    [下游关系] fusion/rerank/persist 使用返回候选。
    """
    if not query_vector or int(top_k) <= 0:
        return []  # docstring: 空向量或非法 top_k 直接返回空

    scope = _normalize_scope_ids(kb_scope)  # docstring: 规范化 KB scope
    expr = build_expr_for_scope(  # docstring: 构造 Milvus 过滤表达式
        kb_id=scope["kb_id"] or "",
        file_id=scope["file_id"],
        document_id=scope["document_id"],
    )
    fields = _normalize_output_fields(output_fields)  # docstring: 规范化输出字段
    mt = _normalize_metric_type(metric_type)  # docstring: 规范化 metric_type
    collection_name = _resolve_collection(collection, kb_scope)  # docstring: collection 解析

    results = await milvus_repo.search(
        collection=collection_name,
        query_vectors=[query_vector],
        top_k=int(top_k),
        expr=expr,
        output_fields=fields,
        metric_type=str(mt),
    )  # docstring: 向量检索

    hits = results[0] if results else []  # docstring: 单 query 的命中列表
    candidates: List[Candidate] = []
    for h in hits:
        cand = _hit_to_candidate(h, metric_type=mt)  # docstring: 映射为 Candidate
        if cand:
            candidates.append(cand)  # docstring: 仅保留合法候选
    return candidates
