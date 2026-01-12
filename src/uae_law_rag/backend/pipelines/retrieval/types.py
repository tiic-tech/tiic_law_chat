# src/uae_law_rag/backend/pipelines/retrieval/types.py
"""
[职责] Retrieval types：提供 retrieval 各阶段共享的最小公共类型（无 DB/外部依赖）。
[边界] 仅定义数据结构与类型；不包含任何检索逻辑。
[上游关系] keyword/vector/fusion/rerank/persist 等模块 import 使用。
[下游关系] 统一 Candidate / stage 语义，确保落库与审计结构一致。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional


CandidateStage = Literal["keyword", "vector", "fusion", "rerank"]  # docstring: 检索阶段枚举


@dataclass(frozen=True)
class Candidate:
    """
    [职责] Candidate：检索阶段统一候选结构（keyword/vector/fusion/rerank 共享）。
    [边界] 只携带必要证据指针与可解释分数；不包含全文与 DB 事务信息。
    [上游关系] keyword_recall/vector_recall 等阶段产出。
    [下游关系] fusion/rerank 继续处理；persist 写入 RetrievalHitModel。
    """

    node_id: str
    stage: CandidateStage
    score: float
    score_details: Dict[str, Any]
    excerpt: Optional[str]
    page: Optional[int]
    start_offset: Optional[int]
    end_offset: Optional[int]
    meta: Dict[str, Any]
