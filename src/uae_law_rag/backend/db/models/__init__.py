# src/uae_law_rag/backend/db/models/__init__.py

"""
[职责] db.models 聚合导出：集中声明 ORM Models，供 Alembic 自动发现与应用层统一导入。
[边界] 仅做导入与 __all__ 暴露；不包含任何业务逻辑。
[上游关系] 依赖各模型文件（user/conversation/message/doc/retrieval/generation）。
[下游关系] backend.db.engine / alembic env.py / repo 层 / service 层 会导入本模块以加载元数据。
"""

from __future__ import annotations

from ..base import Base
from .user import UserModel
from .conversation import ConversationModel
from .message import MessageModel
from .doc import KnowledgeBaseModel, KnowledgeFileModel, DocumentModel, NodeModel, NodeVectorMapModel
from .retrieval import RetrievalRecordModel, RetrievalHitModel
from .generation import GenerationRecordModel
from .evaluator import EvaluationRecordModel
from .run_config import RunConfigModel

__all__ = [
    # base
    "Base",
    # core chat
    "UserModel",
    "ConversationModel",
    "MessageModel",
    # kb + docs
    "KnowledgeBaseModel",
    "KnowledgeFileModel",
    "DocumentModel",
    "NodeModel",
    "NodeVectorMapModel",
    # records
    "RetrievalRecordModel",
    "RetrievalHitModel",
    "GenerationRecordModel",
    # evaluator
    "EvaluationRecordModel",
    # run config
    "RunConfigModel",
]
