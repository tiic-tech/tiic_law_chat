# src/uae_law_rag/backend/db/repo/__init__.py

"""
[职责] db.repo 聚合导出：集中暴露仓储（Repo）对象/函数，供 service 层调用。
[边界] 仅做导入与 __all__ 暴露；不包含业务编排（pipeline orchestration）。
[上游关系] 依赖各 repo 模块（user/conversation/message/ingest/retrieval/generation）。
[下游关系] services 层通过本模块统一导入仓储能力；测试用例可直接引用以做 gate tests。
"""

from __future__ import annotations

from .user_repo import UserRepo
from .conversation_repo import ConversationRepo
from .message_repo import MessageRepo
from .ingest_repo import IngestRepo
from .retrieval_repo import RetrievalRepo
from .generation_repo import GenerationRepo
from .evaluator_repo import EvaluatorRepo
from .node_repo import NodeRepo  # noqa: F401
from .document_repo import DocumentRepo

__all__ = [
    "UserRepo",
    "ConversationRepo",
    "MessageRepo",
    "IngestRepo",
    "RetrievalRepo",
    "GenerationRepo",
    "EvaluatorRepo",
    "NodeRepo",
    "DocumentRepo",
]
