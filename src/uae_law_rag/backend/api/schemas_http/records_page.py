# src/uae_law_rag/backend/api/schemas_http/records_page.py

from __future__ import annotations

from typing import Any, Dict, Optional

from pydantic import BaseModel, ConfigDict, Field

from ._common import DocumentId, KnowledgeBaseId, KnowledgeFileId  # type: ignore


class PageRecordView(BaseModel):
    """
    [职责] PageRecordView：page 级回放视图（用于 PagePreview / 高亮）。
    [边界] v0 返回 markdown 文本；不返回 PDF 渲染；不提供 bbox。
    """

    model_config = ConfigDict(extra="forbid")

    kb_id: KnowledgeBaseId = Field(...)
    document_id: DocumentId = Field(...)
    file_id: KnowledgeFileId = Field(...)

    page: int = Field(..., ge=1)
    pages_total: Optional[int] = Field(default=None, ge=1)

    content: str = Field(default="")
    content_len: int = Field(default=0, ge=0)

    meta: Dict[str, Any] = Field(default_factory=dict)
