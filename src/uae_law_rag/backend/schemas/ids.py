# src/uae_law_rag/backend/schemas/ids.py

"""
[职责] ID 契约层：统一系统内各类实体 ID 的类型别名、生成策略与基础校验（UUID v4 string）。
[边界] 不依赖数据库 ORM；不包含业务字段；仅提供类型/工具函数/轻量校验以便 schemas 与 pipelines/services 复用。
[上游关系] 无（纯工具/契约层）。
[下游关系] backend/schemas/*（retrieval/generation/chat/audit）与 pipelines/services 在创建/传递实体引用时使用。
"""

from __future__ import annotations

from typing import Annotated, NewType
from uuid import UUID, uuid4

from pydantic import BaseModel, ConfigDict, Field, field_validator


# --- Type aliases (contract-first) ---

UUIDStr = NewType("UUIDStr", str)  # docstring: 统一 UUID 字符串类型（运行时仍为 str）

UserId = UUIDStr  # docstring: user.id（SQL UserModel.id）
ConversationId = UUIDStr  # docstring: conversation.id（SQL ConversationModel.id）
MessageId = UUIDStr  # docstring: message.id（SQL MessageModel.id）

KnowledgeBaseId = UUIDStr  # docstring: knowledge_base.id（SQL KnowledgeBaseModel.id）
KnowledgeFileId = UUIDStr  # docstring: knowledge_file.id（SQL KnowledgeFileModel.id）
DocumentId = UUIDStr  # docstring: document.id（SQL DocumentModel.id）
NodeId = UUIDStr  # docstring: node.id（SQL NodeModel.id）
VectorId = UUIDStr  # docstring: milvus vector primary key（NodeVectorMapModel.vector_id / Milvus PK）

RetrievalRecordId = UUIDStr  # docstring: retrieval_record.id（SQL RetrievalRecordModel.id）
RetrievalHitId = UUIDStr  # docstring: retrieval_hit.id（SQL RetrievalHitModel.id）
GenerationRecordId = UUIDStr  # docstring: generation_record.id（SQL GenerationRecordModel.id）


def new_uuid() -> UUIDStr:
    """Generate UUID v4 as string."""  # docstring: 系统内唯一 ID 的默认生成策略
    return UUIDStr(str(uuid4()))


def is_uuid_str(value: str) -> bool:
    """Return True if value parses as UUID string."""  # docstring: 轻量校验工具（不抛异常）
    try:
        UUID(str(value))
        return True
    except Exception:
        return False


# --- Reusable schema mixins ---

IdField = Annotated[str, Field(min_length=36, max_length=36)]  # docstring: UUID v4 string 的字段约束（最小契约）


class HasId(BaseModel):
    """
    [职责] 通用 ID mixin：为需要 id 的 schema 提供统一字段与校验。
    [边界] 仅定义 id 字段；不包含 timestamps/ownership 等。
    [上游关系] 调用方在创建实体时注入或由 new_uuid() 生成。
    [下游关系] retrieval/generation 等 schema 可继承以统一 id 规范。
    """

    model_config = ConfigDict(extra="forbid")

    id: IdField = Field(...)  # docstring: 实体唯一 ID（UUID v4 string）

    @field_validator("id")
    @classmethod
    def _validate_id_uuid(cls, v: str) -> str:
        """Ensure id is UUID string."""  # docstring: schema 层的最小约束，防止漂移到非 UUID
        if not is_uuid_str(v):
            raise ValueError("id must be a UUID string")
        return v
