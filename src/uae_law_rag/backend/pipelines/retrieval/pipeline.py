# src/uae_law_rag/backend/pipelines/retrieval/pipeline.py

"""
[职责] retrieval pipeline：编排 keyword/vector/fusion/rerank/persist 全链路检索，产出可审计的 RetrievalBundle。
[边界] 不创建 Message/KB；不提交事务；仅调度检索与落库组件并记录 timing/provider 快照。
[上游关系] services/chat_service 或脚本调用；输入 query_text/query_vector 与配置；依赖 PipelineContext。
[下游关系] generation/evaluator 消费 RetrievalBundle；DB 记录用于回放审计与 gate tests。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, cast

from sqlalchemy.ext.asyncio import AsyncSession

from uae_law_rag.backend.db.repo.retrieval_repo import RetrievalRepo
from uae_law_rag.backend.kb.repo import MilvusRepo
from uae_law_rag.backend.pipelines.base.context import PipelineContext
from uae_law_rag.backend.schemas.ids import KnowledgeBaseId, MessageId, NodeId, RetrievalRecordId, UUIDStr
from uae_law_rag.backend.schemas.retrieval import (
    FusionStrategy,
    HitSource,
    RetrievalBundle,
    RetrievalHit,
    RetrievalRecord,
    RerankStrategy,
)

from . import fusion as fusion_mod
from . import keyword as keyword_mod
from . import persist as persist_mod
from . import rerank as rerank_mod
from . import vector as vector_mod
from .types import Candidate


_STAGE_TO_SOURCE: Dict[str, HitSource] = {
    "keyword": "keyword",
    "vector": "vector",
    "fusion": "fused",
    "rerank": "reranked",
}  # docstring: Candidate.stage → RetrievalHit.source


@dataclass(frozen=True)
class _RetrievalConfig:
    """Normalized retrieval config."""  # docstring: 内部使用的配置快照

    keyword_top_k: int
    vector_top_k: int
    fusion_top_k: int
    rerank_top_k: int
    fusion_strategy: str
    rerank_strategy: str
    rerank_model: Optional[str]
    rerank_config: Dict[str, Any]
    keyword_allow_fallback: bool
    metric_type: Optional[str]
    output_fields: Optional[List[str]]
    file_id: Optional[str]
    document_id: Optional[str]
    collection: Optional[str]


def _normalize_config(config: Optional[Mapping[str, Any]]) -> _RetrievalConfig:
    """
    [职责] 归一化 retrieval config（补齐默认值并转换类型）。
    [边界] 不做业务策略校验；仅处理缺省与类型。
    [上游关系] run_retrieval_pipeline 调用。
    [下游关系] keyword/vector/fusion/rerank/persist 使用。
    """
    cfg = dict(config or {})  # docstring: 复制配置

    def _as_int(key: str, default: int) -> int:
        v = cfg.get(key, default)
        return int(default if v is None else v)  # docstring: int 兜底

    def _as_str(key: str, default: str) -> str:
        v = cfg.get(key, default)
        return str(default if v is None else v)  # docstring: str 兜底

    def _as_opt_str(key: str) -> Optional[str]:
        v = cfg.get(key)
        if v is None:
            return None
        s = str(v).strip()
        return s or None  # docstring: 空字符串视为 None

    return _RetrievalConfig(
        keyword_top_k=_as_int("keyword_top_k", 200),
        vector_top_k=_as_int("vector_top_k", 50),
        fusion_top_k=_as_int("fusion_top_k", 50),
        rerank_top_k=_as_int("rerank_top_k", 10),
        fusion_strategy=_as_str("fusion_strategy", "union"),
        rerank_strategy=_as_str("rerank_strategy", "none"),
        rerank_model=_as_opt_str("rerank_model") or _as_opt_str("model_name"),
        rerank_config=dict(cfg.get("rerank_config") or {}),
        keyword_allow_fallback=bool(cfg.get("keyword_allow_fallback", True)),
        metric_type=_as_opt_str("metric_type") or _as_opt_str("vector_metric_type"),
        output_fields=cfg.get("output_fields") or cfg.get("vector_output_fields"),
        file_id=_as_opt_str("file_id"),
        document_id=_as_opt_str("document_id"),
        collection=_as_opt_str("milvus_collection") or _as_opt_str("collection"),
    )


def _build_kb_scope(*, kb_id: str, cfg: _RetrievalConfig) -> Dict[str, Optional[str]]:
    """
    [职责] 构建向量检索 scope（kb_id/file_id/document_id）。
    [边界] kb_id 必填；file_id/document_id 可为空。
    [上游关系] run_retrieval_pipeline 调用。
    [下游关系] vector_recall 构造 expr 过滤。
    """
    return {
        "kb_id": str(kb_id),  # docstring: KB 作用域
        "file_id": cfg.file_id,  # docstring: 文件作用域
        "document_id": cfg.document_id,  # docstring: 文档作用域
    }


def _build_provider_snapshot(
    *,
    base_snapshot: Dict[str, Any],
    cfg: _RetrievalConfig,
    kb_id: str,
    errors: Dict[str, str],
) -> Dict[str, Any]:
    """
    [职责] 构建 provider_snapshot（检索配置与错误快照）。
    [边界] 不校验 provider 语义；仅做聚合与透传。
    [上游关系] run_retrieval_pipeline 调用。
    [下游关系] RetrievalRecord.provider_snapshot 写入 DB。
    """
    snapshot = dict(base_snapshot or {})  # docstring: 基础快照复制
    retrieval_snapshot = {
        "kb_id": kb_id,
        "keyword": {
            "top_k": cfg.keyword_top_k,
            "allow_fallback": cfg.keyword_allow_fallback,
        },
        "vector": {
            "top_k": cfg.vector_top_k,
            "metric_type": cfg.metric_type,
            "output_fields": cfg.output_fields,
            "collection": cfg.collection,
            "file_id": cfg.file_id,
            "document_id": cfg.document_id,
        },
        "fusion": {
            "strategy": cfg.fusion_strategy,
            "top_k": cfg.fusion_top_k,
        },
        "rerank": {
            "strategy": cfg.rerank_strategy,
            "top_k": cfg.rerank_top_k,
            "model": cfg.rerank_model,
        },
    }  # docstring: retrieval 快照
    if errors:
        retrieval_snapshot["errors"] = dict(errors)  # docstring: 错误快照
    snapshot["retrieval"] = retrieval_snapshot  # docstring: 写入 retrieval 维度快照
    return snapshot


def _candidate_to_schema_hit(candidate: Candidate, *, retrieval_record_id: str, rank: int) -> RetrievalHit:
    """
    [职责] Candidate → RetrievalHit schema 映射（用于 bundle 返回）。
    [边界] 不读取 DB；仅使用 Candidate 快照字段。
    [上游关系] run_retrieval_pipeline 调用。
    [下游关系] RetrievalBundle.hits。
    """
    source = _STAGE_TO_SOURCE.get(candidate.stage, "fused")  # docstring: stage→source 映射
    return RetrievalHit(
        retrieval_record_id=cast(RetrievalRecordId, UUIDStr(str(retrieval_record_id))),  # docstring: 归属检索记录
        node_id=cast(NodeId, UUIDStr(str(candidate.node_id))),  # docstring: 证据节点ID
        source=cast(HitSource, source),  # docstring: 命中来源
        rank=int(rank),  # docstring: 排名
        score=float(candidate.score),  # docstring: 分数
        score_details=dict(candidate.score_details or {}),  # docstring: 分数细节
        excerpt=candidate.excerpt,  # docstring: 片段摘要
        page=candidate.page,  # docstring: 页码快照
        start_offset=candidate.start_offset,  # docstring: 起始偏移
        end_offset=candidate.end_offset,  # docstring: 结束偏移
    )


def _timing_snapshot(ctx: PipelineContext) -> Dict[str, float]:
    """
    [职责] 导出 retrieval timing 快照（key 统一为 total）。
    [边界] 不做字段裁剪；直接导出 TimingCollector dict。
    [上游关系] run_retrieval_pipeline 调用。
    [下游关系] RetrievalRecord.timing_ms。
    """
    return ctx.timing.to_dict(include_total=True, total_key="total")  # docstring: total 使用一致 key


async def run_retrieval_pipeline(
    *,
    session: AsyncSession,
    milvus_repo: MilvusRepo,
    retrieval_repo: RetrievalRepo,
    message_id: str,
    kb_id: str,
    query_text: str,
    query_vector: Optional[List[float]],
    config: Optional[Mapping[str, Any]] = None,
    ctx: Optional[PipelineContext] = None,
) -> RetrievalBundle:
    """
    [职责] run_retrieval_pipeline：执行检索全链路并落库，返回 RetrievalBundle。
    [边界] 不提交事务；不创建 message/KB；依赖上游提供 query_vector。
    [上游关系] services/chat_service 或脚本调用；上游准备 query_text/query_vector。
    [下游关系] generation pipeline 使用 RetrievalBundle 构建上下文与引用。
    """
    ctx = ctx or PipelineContext.from_session(session)  # docstring: 统一 ctx 装配
    cfg = _normalize_config(config)  # docstring: 归一化配置

    ctx.timing.reset()  # docstring: 清理上次 timing
    errors: Dict[str, str] = {}

    keyword_hits: List[Candidate] = []
    vector_hits: List[Candidate] = []
    fused_hits: List[Candidate] = []
    final_hits: List[Candidate] = []

    if str(query_text or "").strip():
        with ctx.timing.stage("keyword"):
            try:
                keyword_hits = await keyword_mod.keyword_recall(
                    session=session,
                    kb_id=kb_id,
                    query=str(query_text),
                    top_k=cfg.keyword_top_k,
                    file_id=cfg.file_id,
                    allow_fallback=cfg.keyword_allow_fallback,
                )  # docstring: keyword 召回
            except Exception as exc:  # pragma: no cover - 运行时依赖/SQL异常
                errors["keyword_error"] = f"{exc.__class__.__name__}: {exc}"  # docstring: 记录错误
                keyword_hits = []
    else:
        ctx.timing.add_ms("keyword", 0.0, accumulate=False)  # docstring: 空 query 记 0ms

    if query_vector:
        with ctx.timing.stage("vector"):
            try:
                kb_scope = _build_kb_scope(kb_id=kb_id, cfg=cfg)  # docstring: 构建 KB scope
                vector_hits = await vector_mod.vector_recall(
                    milvus_repo=milvus_repo,
                    kb_scope=kb_scope,
                    query_vector=query_vector,
                    top_k=cfg.vector_top_k,
                    output_fields=cfg.output_fields,
                    metric_type=cfg.metric_type,
                    collection=cfg.collection,
                )  # docstring: vector 召回
            except Exception as exc:  # pragma: no cover - Milvus 依赖异常
                errors["vector_error"] = f"{exc.__class__.__name__}: {exc}"  # docstring: 记录错误
                vector_hits = []
    else:
        ctx.timing.add_ms("vector", 0.0, accumulate=False)  # docstring: 无 query_vector 记 0ms

    with ctx.timing.stage("fusion"):
        try:
            fused_hits = fusion_mod.fuse_candidates(
                keyword=keyword_hits,
                vector=vector_hits,
                strategy=cfg.fusion_strategy,
                top_k=cfg.fusion_top_k,
            )  # docstring: 融合去重
        except Exception as exc:  # pragma: no cover - 逻辑兜底
            errors["fusion_error"] = f"{exc.__class__.__name__}: {exc}"  # docstring: 记录错误
            fused_hits = list(keyword_hits) + list(vector_hits)  # docstring: 回退拼接
            fused_hits = fused_hits[: cfg.fusion_top_k]  # docstring: 回退时仍遵守 top_k

    if cfg.rerank_strategy.strip().lower() != "none" and fused_hits:
        with ctx.timing.stage("rerank"):
            try:
                final_hits = await rerank_mod.rerank(
                    query=str(query_text),
                    candidates=fused_hits,
                    strategy=cfg.rerank_strategy,
                    top_k=cfg.rerank_top_k,
                    model=cfg.rerank_model,
                    rerank_config=cfg.rerank_config,
                )  # docstring: rerank 精排
            except Exception as exc:  # pragma: no cover - 依赖异常
                errors["rerank_error"] = f"{exc.__class__.__name__}: {exc}"  # docstring: 记录错误
                final_hits = list(fused_hits)  # docstring: rerank 失败回退
    else:
        ctx.timing.add_ms("rerank", 0.0, accumulate=False)  # docstring: 跳过 rerank 记 0ms
        final_hits = list(fused_hits)[: cfg.rerank_top_k]  # docstring: 无 rerank 时按 top_k 截断

    provider_snapshot = _build_provider_snapshot(
        base_snapshot=dict(ctx.provider_snapshot),
        cfg=cfg,
        kb_id=kb_id,
        errors=errors,
    )  # docstring: provider 快照

    record_params = {
        "message_id": message_id,
        "kb_id": kb_id,
        "query_text": str(query_text),
        "keyword_top_k": cfg.keyword_top_k,
        "vector_top_k": cfg.vector_top_k,
        "fusion_top_k": cfg.fusion_top_k,
        "rerank_top_k": cfg.rerank_top_k,
        "fusion_strategy": cfg.fusion_strategy,
        "rerank_strategy": cfg.rerank_strategy,
        "provider_snapshot": provider_snapshot,
        "timing_ms": _timing_snapshot(ctx),
    }  # docstring: RetrievalRecord 参数快照

    retrieval_record_id, _hit_count = await persist_mod.persist_retrieval(
        retrieval_repo=retrieval_repo,
        record_params=record_params,
        hits=final_hits,
    )  # docstring: 落库 record + hits

    record = RetrievalRecord(
        id=cast(RetrievalRecordId, UUIDStr(str(retrieval_record_id))),
        message_id=cast(MessageId, UUIDStr(str(message_id))),
        kb_id=cast(KnowledgeBaseId, UUIDStr(str(kb_id))),
        query_text=str(query_text),
        keyword_top_k=cfg.keyword_top_k,
        vector_top_k=cfg.vector_top_k,
        fusion_top_k=cfg.fusion_top_k,
        rerank_top_k=cfg.rerank_top_k,
        fusion_strategy=cast(FusionStrategy, cfg.fusion_strategy),
        rerank_strategy=cast(RerankStrategy, cfg.rerank_strategy),
        provider_snapshot=provider_snapshot,
        timing_ms=record_params["timing_ms"],
    )  # docstring: 构造 RetrievalRecord schema

    hits_schema = [
        _candidate_to_schema_hit(h, retrieval_record_id=retrieval_record_id, rank=i)
        for i, h in enumerate(final_hits, start=1)
    ]  # docstring: 构造 RetrievalHit schema 列表

    return RetrievalBundle(record=record, hits=hits_schema)
