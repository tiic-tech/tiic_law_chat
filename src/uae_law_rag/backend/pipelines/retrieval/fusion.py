# src/uae_law_rag/backend/pipelines/retrieval/fusion.py

"""
[职责] fusion：融合 keyword/vector 候选集合，完成去重与稳定排序，产出统一 Candidate 列表。
[边界] 仅依赖“score 越大越好”的单调性与 rank；不假设 score 同一量纲；不落库。
[上游关系] keyword_recall/vector_recall 输出候选；pipeline 传入 fusion_strategy/top_k。
[下游关系] rerank/persist 消费融合结果用于精排与审计落库。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Literal, Optional, Sequence, Tuple, TypeVar, cast

from uae_law_rag.backend.pipelines.retrieval.types import Candidate


FusionStrategy = Literal["union", "rrf", "weighted", "interleave"]  # docstring: fusion 策略枚举
T = TypeVar("T")  # docstring: 字段值类型占位


@dataclass(frozen=True)
class _RankedCandidate:
    """Ranked candidate with rank and source."""  # docstring: 内部使用的 rank 包装结构

    candidate: Candidate
    rank: int


def _normalize_strategy(strategy: str) -> Tuple[FusionStrategy, bool]:
    """
    [职责] 归一化 fusion 策略名称。
    [边界] 不校验参数合法性；未知策略回退为 union。
    [上游关系] fuse_candidates 调用。
    [下游关系] 后续策略分支选择。
    """
    s = str(strategy or "").strip().lower()  # docstring: 统一小写
    if s in {"union", "rrf", "weighted", "interleave"}:
        return cast(FusionStrategy, s), False  # docstring: 合法策略
    return "union", True  # docstring: 未知策略回退


def _rank_candidates(candidates: Sequence[Candidate]) -> Dict[str, _RankedCandidate]:
    """
    [职责] 按单调 score 排序并生成 node_id -> ranked candidate 映射。
    [边界] 仅依赖 score 单调性；不做跨列表归一化。
    [上游关系] fuse_candidates 调用。
    [下游关系] 融合策略按 rank 计算分数。
    """
    ordered = sorted(candidates, key=lambda c: (-float(c.score), str(c.node_id)))  # docstring: 单列表内按 score 降序
    out: Dict[str, _RankedCandidate] = {}
    for idx, cand in enumerate(ordered, start=1):
        node_id = str(cand.node_id)  # docstring: 统一 node_id 字符串
        if node_id in out:
            continue  # docstring: 去重保留最高 rank
        out[node_id] = _RankedCandidate(candidate=cand, rank=idx)  # docstring: 记录 rank
    return out


def _rank_score(rank: Optional[int]) -> float:
    """
    [职责] 将 rank 转为单调分数（越靠前越高）。
    [边界] 仅依赖 rank；缺失 rank 返回 0；rank=1 映射为 1.0。
    [上游关系] union/weighted 融合使用。
    [下游关系] fused score 计算。
    """
    if not rank:
        return 0.0  # docstring: 无 rank 视为 0
    return 1.0 / float(rank)  # docstring: rank -> (0,1] 单调分数


def _rrf_score(rank: Optional[int], rrf_k: int) -> float:
    """
    [职责] Reciprocal Rank Fusion 分数计算。
    [边界] 仅依赖 rank；缺失 rank 返回 0。
    [上游关系] rrf/weighted 融合使用。
    [下游关系] fused score 计算。
    """
    if not rank:
        return 0.0  # docstring: 无 rank 视为 0
    return 1.0 / (float(rrf_k) + float(rank))  # docstring: RRF 公式


def _merge_meta(
    keyword: Optional[Candidate],
    vector: Optional[Candidate],
) -> Dict[str, object]:
    """
    [职责] 合并 keyword/vector meta（优先保留 keyword）。
    [边界] 仅做浅合并；冲突字段保留 keyword 值。
    [上游关系] 融合候选构造调用。
    [下游关系] Candidate.meta 透传。
    """
    meta: Dict[str, object] = {}
    if keyword:
        meta.update(keyword.meta or {})  # docstring: keyword meta 优先
    if vector:
        for k, v in (vector.meta or {}).items():
            meta.setdefault(k, v)  # docstring: 仅补充缺失字段
    return meta


def _choose_field(keyword_val: Optional[T], vector_val: Optional[T]) -> Optional[T]:
    """
    [职责] 选择字段值（优先 keyword，缺失则 fallback）。
    [边界] 仅做空值判断；不做类型转换。
    [上游关系] 融合候选构造调用。
    [下游关系] Candidate.excerpt/page/offset。
    """
    # docstring: 对字符串字段，空/纯空白视为缺失，允许 fallback（用于 excerpt 等）。
    if keyword_val is not None:
        if isinstance(keyword_val, str) and not keyword_val.strip():
            pass  # docstring: 空字符串按缺失处理，继续 fallback
        else:
            return keyword_val  # docstring: keyword 优先

    if vector_val is not None:
        if isinstance(vector_val, str) and not vector_val.strip():
            return None  # docstring: 两侧都是空字符串则视为缺失
        return vector_val  # docstring: vector 兜底
    return None


def _build_score_details(
    *,
    strategy: FusionStrategy,
    keyword: Optional[_RankedCandidate],
    vector: Optional[_RankedCandidate],
    fused_score: float,
    rrf_k: int,
    weights: Dict[str, float],
    fallback: bool,
) -> Dict[str, object]:
    """
    [职责] 构造 score_details（包含双路分数与 rank）。
    [边界] 不做深层拷贝；仅聚合必要字段。
    [上游关系] 融合候选构造调用。
    [下游关系] Candidate.score_details 写入。
    """

    def _pack(rc: Optional[_RankedCandidate]) -> Optional[Dict[str, object]]:
        if not rc:
            return None
        return {
            "score": float(rc.candidate.score),
            "rank": int(rc.rank),
            "rank_score": _rank_score(rc.rank),
            "score_details": dict(rc.candidate.score_details or {}),
        }  # docstring: 单路 score 快照

    return {
        "fusion_strategy": strategy,  # docstring: 融合策略
        "strategy_fallback": fallback,  # docstring: 是否发生策略回退
        "fusion_score": float(fused_score),  # docstring: 融合分数
        "rrf_k": int(rrf_k),  # docstring: RRF 常量
        "weights": dict(weights),  # docstring: 权重配置
        "keyword": _pack(keyword),  # docstring: keyword 侧细节
        "vector": _pack(vector),  # docstring: vector 侧细节
    }


def _interleave(
    *,
    keyword: Sequence[Candidate],
    vector: Sequence[Candidate],
    top_k: int,
) -> List[Candidate]:
    """
    [职责] interleave 融合：按 keyword/vector 交替输出并去重。
    [边界] 仅依赖单列表排序；不比较跨列表 raw score。
    [上游关系] fuse_candidates 传入候选列表。
    [下游关系] 返回 fusion candidates。
    """
    kw_ranked = list(_rank_candidates(keyword).values())  # docstring: keyword 排名
    vec_ranked = list(_rank_candidates(vector).values())  # docstring: vector 排名

    fused: List[Candidate] = []
    seen: set[str] = set()  # docstring: 去重集合
    i = 0
    while len(fused) < int(top_k) and (i < len(kw_ranked) or i < len(vec_ranked)):
        for src in (kw_ranked, vec_ranked):
            if i >= len(src):
                continue
            rc = src[i]
            node_id = str(rc.candidate.node_id)
            if node_id in seen:
                continue  # docstring: 跳过重复
            seen.add(node_id)  # docstring: 记录已输出节点

            position = len(fused) + 1  # docstring: 位置即 rank
            fused_score = _rank_score(position)  # docstring: 位置 -> 融合分数
            score_details = _build_score_details(
                strategy="interleave",
                keyword=rc if rc.candidate.stage == "keyword" else None,
                vector=rc if rc.candidate.stage == "vector" else None,
                fused_score=fused_score,
                rrf_k=60,
                weights={"keyword": 0.5, "vector": 0.5},
                fallback=False,
            )
            fused.append(
                Candidate(
                    node_id=node_id,
                    stage="fusion",
                    score=fused_score,
                    score_details=score_details,
                    excerpt=_choose_field(rc.candidate.excerpt, None),
                    page=_choose_field(rc.candidate.page, None),
                    start_offset=_choose_field(rc.candidate.start_offset, None),
                    end_offset=_choose_field(rc.candidate.end_offset, None),
                    meta=dict(rc.candidate.meta or {}),
                )
            )  # docstring: 追加 interleave 候选
            if len(fused) >= int(top_k):
                break
        i += 1
    return fused


def fuse_candidates(
    *,
    keyword: Sequence[Candidate],
    vector: Sequence[Candidate],
    strategy: str,
    top_k: int,
    rrf_k: int = 60,
    keyword_weight: float = 0.5,
    vector_weight: float = 0.5,
) -> List[Candidate]:
    """
    [职责] fuse_candidates：融合 keyword/vector 候选，去重并输出 top_k。
    [边界] 仅依赖单调 score 与 rank；不直接比较跨列表 score 数值。
    [上游关系] keyword/vector recall 输出候选；pipeline 传入 fusion 策略与 top_k。
    [下游关系] rerank/persist 消费融合结果。
    """
    if int(top_k) <= 0:
        return []  # docstring: 非法 top_k 直接返回空

    fusion_strategy, fallback = _normalize_strategy(strategy)  # docstring: 归一化策略
    if fusion_strategy == "interleave":
        return _interleave(keyword=keyword, vector=vector, top_k=top_k)  # docstring: 交替融合

    kw_ranked = _rank_candidates(keyword)  # docstring: keyword 排名映射
    vec_ranked = _rank_candidates(vector)  # docstring: vector 排名映射

    weights = {
        "keyword": float(keyword_weight),
        "vector": float(vector_weight),
    }  # docstring: 权重快照

    node_ids = set(kw_ranked.keys()) | set(vec_ranked.keys())  # docstring: 融合去重集合
    fused: List[Tuple[Candidate, float, int, bool]] = []

    for node_id in node_ids:
        kw = kw_ranked.get(node_id)  # docstring: keyword 命中
        vec = vec_ranked.get(node_id)  # docstring: vector 命中

        kw_rank = kw.rank if kw else None
        vec_rank = vec.rank if vec else None

        if fusion_strategy == "rrf":
            fused_score = _rrf_score(kw_rank, rrf_k) + _rrf_score(vec_rank, rrf_k)  # docstring: RRF 融合
        elif fusion_strategy == "weighted":
            fused_score = weights["keyword"] * _rrf_score(kw_rank, rrf_k) + weights["vector"] * _rrf_score(
                vec_rank, rrf_k
            )  # docstring: 权重融合（基于 rank）
        else:
            fused_score = max(_rank_score(kw_rank), _rank_score(vec_rank))  # docstring: union 取最佳 rank

        keyword_cand = kw.candidate if kw else None
        vector_cand = vec.candidate if vec else None
        score_details = _build_score_details(
            strategy=fusion_strategy,
            keyword=kw,
            vector=vec,
            fused_score=fused_score,
            rrf_k=rrf_k,
            weights=weights,
            fallback=fallback,
        )  # docstring: 分数细节快照

        meta = _merge_meta(keyword_cand, vector_cand)  # docstring: 合并 meta

        excerpt = _choose_field(
            keyword_cand.excerpt if keyword_cand else None,
            vector_cand.excerpt if vector_cand else None,
        )
        if isinstance(excerpt, str) and not excerpt.strip():
            excerpt = None  # docstring: 统一将空白 excerpt 视为缺失

        fused_candidate = Candidate(
            node_id=node_id,  # docstring: 节点ID
            stage="fusion",  # docstring: 标记 fusion 阶段
            score=fused_score,  # docstring: 融合分数
            score_details=score_details,  # docstring: 分数细节
            excerpt=excerpt,
            page=_choose_field(
                keyword_cand.page if keyword_cand else None,
                vector_cand.page if vector_cand else None,
            ),
            start_offset=_choose_field(
                keyword_cand.start_offset if keyword_cand else None,
                vector_cand.start_offset if vector_cand else None,
            ),
            end_offset=_choose_field(
                keyword_cand.end_offset if keyword_cand else None,
                vector_cand.end_offset if vector_cand else None,
            ),
            meta=meta,  # docstring: 透传 meta
        )

        min_rank = min(kw_rank or 10**9, vec_rank or 10**9)  # docstring: 用于稳定排序的 rank
        has_keyword = bool(kw)  # docstring: keyword 优先级
        fused.append((fused_candidate, fused_score, min_rank, has_keyword))

    fused_sorted = sorted(
        fused,
        key=lambda x: (-float(x[1]), -int(x[3]), int(x[2]), str(x[0].node_id)),
    )  # docstring: 稳定排序（分数>keyword>rank>id）

    return [item[0] for item in fused_sorted[: int(top_k)]]  # docstring: 截断 top_k
