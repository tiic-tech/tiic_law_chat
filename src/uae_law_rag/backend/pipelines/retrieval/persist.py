# src/uae_law_rag/backend/pipelines/retrieval/persist.py

"""
[职责] persist：将 RetrievalRecord 与 RetrievalHit 写入 DB，形成可回放审计单元。
[边界] 不执行检索算法；不做事务提交；仅负责 Candidate → DB 结构映射。
[上游关系] retrieval pipeline 产出 record_params 与 hits；依赖 RetrievalRepo 提供写入接口。
[下游关系] DB retrieval_record/retrieval_hit 作为 generation/evaluator 的证据入口。
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from uae_law_rag.backend.db.repo.retrieval_repo import RetrievalRepo
from uae_law_rag.backend.pipelines.retrieval.types import Candidate


_STAGE_TO_SOURCE = {
    "keyword": "keyword",
    "vector": "vector",
    "fusion": "fused",
    "rerank": "reranked",
}  # docstring: Candidate.stage → RetrievalHit.source 映射


def _coerce_int(value: Any) -> Optional[int]:
    """
    [职责] 尝试将值转换为 int（用于 page/offset 兜底）。
    [边界] 仅支持 int/float/数字字符串；否则返回 None。
    [上游关系] _candidate_to_hit 调用。
    [下游关系] RetrievalHit.page/start_offset/end_offset。
    """
    if value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return None


def _resolve_source(stage: str) -> str:
    """
    [职责] 将 Candidate.stage 映射为 DB hit.source。
    [边界] 未知 stage 回退为 fused。
    [上游关系] _candidate_to_hit 调用。
    [下游关系] RetrievalHit.source 写入。
    """
    key = str(stage or "").strip().lower()
    return _STAGE_TO_SOURCE.get(key, "fused")  # docstring: 未知 stage 兜底为 fused


def _candidate_to_hit(candidate: Candidate, *, rank: int) -> Dict[str, Any]:
    """
    [职责] 将 Candidate 映射为 RetrievalHitModel 写入 payload。
    [边界] 不读取 NodeModel；page/offset 优先 Candidate，其次 meta。
    [上游关系] persist_retrieval 调用。
    [下游关系] RetrievalRepo.bulk_create_hits 写入 DB。
    """
    meta = candidate.meta or {}  # docstring: 透传 meta 作为兜底
    page = candidate.page if candidate.page is not None else _coerce_int(meta.get("page"))  # docstring: 页码兜底
    start_offset = (
        candidate.start_offset if candidate.start_offset is not None else _coerce_int(meta.get("start_offset"))
    )  # docstring: 起始偏移兜底
    end_offset = (
        candidate.end_offset if candidate.end_offset is not None else _coerce_int(meta.get("end_offset"))
    )  # docstring: 结束偏移兜底

    return {
        "node_id": str(candidate.node_id),  # docstring: 证据节点ID
        "source": _resolve_source(candidate.stage),  # docstring: hit 来源阶段
        "rank": int(rank),  # docstring: 命中排名（1-based）
        "score": float(candidate.score),  # docstring: 主分数
        "score_details": dict(candidate.score_details or {}),  # docstring: 分数细节快照
        "excerpt": candidate.excerpt,  # docstring: 片段摘要（可选）
        "page": page,  # docstring: 页码快照
        "start_offset": start_offset,  # docstring: 起始偏移快照
        "end_offset": end_offset,  # docstring: 结束偏移快照
    }


def _normalize_record_params(record_params: Mapping[str, Any]) -> Dict[str, Any]:
    """
    [职责] 规范化 RetrievalRecord 入参（必填字段校验 + 类型转换）。
    [边界] 仅做最小校验；业务策略由 pipeline 保证。
    [上游关系] persist_retrieval 调用。
    [下游关系] RetrievalRepo.create_record。
    """
    required = [
        "message_id",
        "kb_id",
        "query_text",
        "keyword_top_k",
        "vector_top_k",
        "fusion_top_k",
        "rerank_top_k",
        "fusion_strategy",
        "rerank_strategy",
    ]
    missing = [k for k in required if k not in record_params]
    if missing:
        raise ValueError(f"record_params missing: {', '.join(missing)}")  # docstring: 必填字段缺失

    params: Dict[str, Any] = {
        "message_id": str(record_params["message_id"]),  # docstring: 归属 message
        "kb_id": str(record_params["kb_id"]),  # docstring: 归属 KB
        "query_text": str(record_params["query_text"]),  # docstring: 检索 query
        "keyword_top_k": int(record_params["keyword_top_k"]),  # docstring: keyword top_k
        "vector_top_k": int(record_params["vector_top_k"]),  # docstring: vector top_k
        "fusion_top_k": int(record_params["fusion_top_k"]),  # docstring: fusion top_k
        "rerank_top_k": int(record_params["rerank_top_k"]),  # docstring: rerank top_k
        "fusion_strategy": str(record_params["fusion_strategy"]),  # docstring: 融合策略
        "rerank_strategy": str(record_params["rerank_strategy"]),  # docstring: rerank 策略
        "provider_snapshot": record_params.get("provider_snapshot") or {},  # docstring: provider 快照
        "timing_ms": record_params.get("timing_ms") or {},  # docstring: timing 快照
    }
    return params


async def persist_retrieval(
    *,
    retrieval_repo: RetrievalRepo,
    record_params: Mapping[str, Any],
    hits: Sequence[Candidate],
) -> Tuple[str, int]:
    """
    [职责] persist_retrieval：写入 RetrievalRecord 与 hits，返回 record_id 与 hit_count。
    [边界] 不提交事务；不重排 hits；仅按输入顺序写入 rank。
    [上游关系] retrieval pipeline 产出 record_params 与 hits。
    [下游关系] RetrievalRepo 写入 DB；generation/evaluator 消费记录与命中。
    """
    params = _normalize_record_params(record_params)  # docstring: 规范化 record 入参
    record = await retrieval_repo.create_record(**params)  # docstring: 先落 RetrievalRecord

    hit_payloads = [_candidate_to_hit(c, rank=i) for i, c in enumerate(hits, start=1)]  # docstring: 命中映射
    if hit_payloads:
        await retrieval_repo.bulk_create_hits(
            retrieval_record_id=record.id,
            hits=hit_payloads,
        )  # docstring: 批量落库 hits

    return record.id, len(hit_payloads)
