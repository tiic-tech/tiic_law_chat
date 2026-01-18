# src/uae_law_rag/backend/pipelines/evaluator/query_plan.py

"""
[职责] query_plan：将用户 raw_query 转为可解释、可序列化的 QueryPlan（规则版 v1）。
[边界] v1 不调用 LLM；不依赖 DB；不改动 retrieval/generation 主链路；仅提供纯函数。
[上游关系] /api/evaluator/query_plan（只读）；/api/chat debug.query_plan（只读透传）。
[下游关系] keyword_recall evaluator（可用 keywords_list 做批量评估）；后续 M2 扩展 enhanced_queries。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence


# docstring: 最小 stopwords 集合。后续可按 locale/domain 扩展，但 v1 保持极简、确定性。
_DEFAULT_STOPWORDS = {
    "a",
    "an",
    "the",
    "and",
    "or",
    "of",
    "to",
    "for",
    "in",
    "on",
    "at",
    "by",
    "with",
    "from",
    "as",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "this",
    "that",
    "these",
    "those",
    "it",
    "its",
    "into",
    "over",
    "under",
    "about",
    "within",
    "without",
    "between",
    "among",
}


_TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:[-_][A-Za-z0-9]+)*")


@dataclass(frozen=True)
class QueryPlan:
    """
    [职责] QueryPlan：pipeline 内部结构。对外可映射到 QueryAnalysisView。
    [边界] 仅包含最小字段；meta 用于审计规则策略版本。
    """

    raw_query: str
    keywords_list: List[str]
    enhanced_queries: List[str]
    meta: Dict[str, object]


def _tokenize(raw_query: str) -> List[str]:
    # docstring: 使用可预测 regex 抽取 token，避免 locale/分词器引入的不确定性。
    return _TOKEN_RE.findall(raw_query)


def _normalize_token(tok: str) -> str:
    return tok.strip().lower()


def _is_useless(tok: str) -> bool:
    if not tok:
        return True
    if tok in _DEFAULT_STOPWORDS:
        return True
    # docstring: 过滤单字符与纯数字（v1 规则；后续如需保留数字，可在 meta 标注策略变更）
    if len(tok) <= 1:
        return True
    if tok.isdigit():
        return True
    return False


def _dedupe_preserve_order(items: Sequence[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _make_phrases(tokens: Sequence[str], max_phrases: int = 3) -> List[str]:
    """
    [职责] 简单 2-gram phrase 生成，用于补充关键词短语（可解释、确定性）。
    [边界] v1 只做相邻组合；不做统计权重；不依赖语料。
    """
    phrases: List[str] = []
    for i in range(len(tokens) - 1):
        phrases.append(f"{tokens[i]} {tokens[i+1]}")
    return phrases[: max_phrases if max_phrases >= 0 else 0]


def build_query_plan(
    raw_query: str,
    *,
    kb_id: Optional[str] = None,
    max_keywords: int = 8,
    include_phrases: bool = True,
    max_phrases: int = 3,
    strategy_version: str = "rule_v1",
) -> QueryPlan:
    """
    [职责] 构造 QueryPlan（规则版 v1）。
    [入参]
      - raw_query: 用户原始 query（必须非空）
      - kb_id: 预留（便于未来按 kb/domain/locale 定制 stopwords/短语策略）
      - max_keywords: keywords_list 的上限，避免 payload 爆炸
      - include_phrases/max_phrases: 是否合成少量 2-gram phrase
      - strategy_version: meta 标记，用于审计与回归测试
    [输出]
      - QueryPlan：raw_query / keywords_list / enhanced_queries / meta
    """
    q = (raw_query or "").strip()
    if not q:
        # docstring: 上游 http schema 已会拦截空字符串；这里仍兜底保证纯函数安全。
        return QueryPlan(
            raw_query="",
            keywords_list=[],
            enhanced_queries=[],
            meta={
                "strategy": strategy_version,
                "kb_id": kb_id,
                "tokens_n": 0,
                "keywords_n": 0,
                "note": "empty_query",
            },
        )

    raw_tokens = _tokenize(q)
    norm_tokens = [_normalize_token(t) for t in raw_tokens]
    filtered = [t for t in norm_tokens if not _is_useless(t)]

    # docstring: 去重并保序，保持确定性。
    keywords = _dedupe_preserve_order(filtered)[: max_keywords if max_keywords > 0 else 0]

    # docstring: 可选添加少量短语（2-gram），用于覆盖 “public companies” 这种场景。
    if include_phrases and len(keywords) >= 2 and max_phrases > 0:
        phrases = _make_phrases(keywords, max_phrases=max_phrases)
        keywords = _dedupe_preserve_order(list(keywords) + phrases)[:max_keywords]

    # docstring: v1 不做 LLM rewrite；enhanced_queries 暂为空（或未来按模板生成少量）。
    enhanced_queries: List[str] = []

    meta: Dict[str, object] = {
        "strategy": strategy_version,
        "kb_id": kb_id,
        "tokens_n": len(raw_tokens),
        "keywords_n": len(keywords),
        "include_phrases": bool(include_phrases),
        "max_keywords": int(max_keywords),
        "max_phrases": int(max_phrases),
    }

    return QueryPlan(
        raw_query=q,
        keywords_list=keywords,
        enhanced_queries=enhanced_queries,
        meta=meta,
    )
