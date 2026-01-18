# src/uae_law_rag/backend/db/repo/document_repo.py

from __future__ import annotations

from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from uae_law_rag.backend.db.models.doc import DocumentModel, KnowledgeFileModel


class DocumentRepo:
    """
    [职责] DocumentRepo：只读回放文档/文件元信息。
    [边界] 不做 ingest 写入；不做检索重算。
    """

    def __init__(self, session: AsyncSession):
        self._session = session

    async def get_document(self, document_id: str) -> Optional[DocumentModel]:
        did = str(document_id or "").strip()
        if not did:
            return None
        stmt = select(DocumentModel).where(DocumentModel.id == did)
        return (await self._session.execute(stmt)).scalars().first()

    async def get_file(self, file_id: str) -> Optional[KnowledgeFileModel]:
        fid = str(file_id or "").strip()
        if not fid:
            return None
        stmt = select(KnowledgeFileModel).where(KnowledgeFileModel.id == fid)
        return (await self._session.execute(stmt)).scalars().first()
