# src/uae_law_rag/backend/pipelines/retrieval/keyword.py

"""
[职责] keyword recall：基于 DB FTS5/BM25 完成关键词全量召回，并映射为统一 Candidate 结构。
[边界] 仅执行关键词检索与分数归一化；不做向量召回/融合/重排；不负责落库与编排。
[上游关系] retrieval pipeline 传入 query/kb_id；依赖 db/fts.search_nodes 返回 KeywordHit。
[下游关系] fusion/rerank/persist 消费 Candidate 列表用于排序与审计落库。
"""

from __future__ import annotations

import re
from typing import Any, List, Literal, Optional, Sequence

from sqlalchemy.ext.asyncio import AsyncSession

from uae_law_rag.backend.db.fts import KeywordHit, search_nodes
from uae_law_rag.backend.pipelines.retrieval.types import Candidate

CandidateStage = Literal["keyword", "vector", "fusion", "rerank"]  # docstring: 检索阶段枚举


def _normalize_query(query: str) -> str:
    """
    [职责] 归一化 keyword query（去除多余空白）。
    [边界] 不做语义改写；不引入停用词策略。
    [上游关系] keyword_recall 调用。
    [下游关系] _tokenize_query 与 FTS 查询字符串构造。
    """
    raw = str(query or "")  # docstring: 兜底为字符串
    return " ".join(raw.strip().split())  # docstring: 合并多余空白


def _tokenize_query(query: str) -> List[str]:
    """
    [职责] 将 query 拆分为 FTS 友好 token 列表。
    [边界] 仅做基础拆分；不做 stemming/拼写纠错。
    [上游关系] keyword_recall 调用。
    [下游关系] _build_fts_query 构造 FTS 查询表达式。
    """
    tokens = re.findall(r"[\w]+", query, flags=re.UNICODE)  # docstring: 提取 Unicode 词元
    return [t for t in tokens if t.strip()]  # docstring: 剔除空 token


def _build_fts_query(tokens: Sequence[str], *, mode: Literal["and", "or"]) -> str:
    """
    [职责] 将 token 列表构造为 FTS 查询表达式。
    [边界] 仅拼接 AND/OR；不注入高级 FTS 语法。
    [上游关系] keyword_recall 调用。
    [下游关系] db/fts.search_nodes 使用该表达式检索。
    """
    if not tokens:
        return ""  # docstring: 无 token 直接返回空
    sep = " OR " if mode == "or" else " "  # docstring: OR 提升召回；AND 作为主路径
    return sep.join(tokens)  # docstring: 构造 FTS 查询字符串


def _bm25_to_score(bm25: float) -> float:
    """
    [职责] 将 bm25（越小越好）转换为“越大越好”的统一分数。
    [边界] 仅做单调转换；不跨候选归一化。
    [上游关系] _hit_to_candidate 调用。
    [下游关系] Candidate.score 用于后续 fusion/rerank。
    """
    if bm25 is None:
        return 0.0  # docstring: 缺失分数不得被判定为强相关
    raw = float(bm25)  # docstring: bm25 兜底为 float
    if raw <= 0:
        return 1.0  # docstring: 负值/零视为强相关
    return 1.0 / (1.0 + raw)  # docstring: 映射到 (0,1]


def _hit_to_candidate(
    hit: KeywordHit,
    *,
    fts_query: str,
    mode: Literal["and", "or"],
) -> Candidate:
    """
    [职责] 将 KeywordHit 映射为统一 Candidate 结构。
    [边界] 不读取 NodeModel 原文；excerpt 仅使用 FTS snippet。
    [上游关系] keyword_recall 调用。
    [下游关系] 返回 Candidate 供 fusion/rerank/persist 使用。
    """
    meta = dict(hit.meta or {})  # docstring: 透传元信息（kb/doc/file/page/article）
    raw_score = hit.score  # docstring: bm25 原始分数（可能为空）
    norm_score = _bm25_to_score(raw_score)  # docstring: 统一分数（越大越好）
    score_details = {
        "bm25": float(raw_score) if raw_score is not None else None,
        "bm25_norm": norm_score,
        "fts_query": fts_query,
        "keyword_mode": mode,
        "keyword_strategy": "fts5",
    }  # docstring: 可解释分数细节
    excerpt = str(hit.snippet or "") or None  # docstring: 命中片段（可空）

    def _coerce_int(v: Any) -> Optional[int]:
        if v is None:
            return None
        try:
            return int(v)
        except (TypeError, ValueError):
            return None

    return Candidate(
        node_id=str(hit.node_id),  # docstring: 节点ID
        stage="keyword",  # docstring: 标记 keyword 阶段
        score=norm_score,  # docstring: 归一化分数
        score_details=score_details,  # docstring: 分数细节快照
        excerpt=excerpt,  # docstring: snippet 作为摘要
        page=_coerce_int(meta.get("page")),  # docstring: 页码快照
        start_offset=_coerce_int(meta.get("start_offset")),  # docstring: 起始偏移（可能为空）
        end_offset=_coerce_int(meta.get("end_offset")),  # docstring: 结束偏移（可能为空）
        meta=meta,  # docstring: 透传 meta
    )


async def keyword_recall(
    *,
    session: AsyncSession,
    kb_id: str,
    query: str,
    top_k: int,
    file_id: Optional[str] = None,
    allow_fallback: bool = True,
) -> List[Candidate]:
    """
    [职责] keyword_recall：执行 FTS5 关键词召回并产出 Candidate 列表。
    [边界] 不做向量召回/融合/重排；不落库；仅依赖 DB FTS。
    [上游关系] retrieval pipeline 传入 query/kb_id/top_k。
    [下游关系] fusion/rerank/persist 使用返回候选。
    """
    normalized = _normalize_query(query)  # docstring: 归一化输入 query
    if not normalized or int(top_k) <= 0:
        return []  # docstring: 空 query 或非法 top_k 直接返回空

    tokens = _tokenize_query(normalized)  # docstring: 拆分为 token 列表
    if not tokens:
        return []  # docstring: 无有效 token 时返回空

    fts_query = _build_fts_query(tokens, mode="and")  # docstring: 主路径使用 AND
    hits = await search_nodes(
        session,
        kb_id=kb_id,
        query=fts_query,
        top_k=int(top_k),
        file_id=file_id,
    )  # docstring: DB FTS 检索候选
    mode: Literal["and", "or"] = "and"

    if allow_fallback and not hits and len(tokens) > 1:
        fts_query = _build_fts_query(tokens, mode="or")  # docstring: 召回不足时回退 OR
        hits = await search_nodes(
            session,
            kb_id=kb_id,
            query=fts_query,
            top_k=int(top_k),
            file_id=file_id,
        )  # docstring: OR 模式提升召回
        mode = "or"

    return [_hit_to_candidate(h, fts_query=fts_query, mode=mode) for h in hits]  # docstring: 统一候选输出
