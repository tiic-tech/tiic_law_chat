# src/uae_law_rag/backend/pipelines/ingest/persist_milvus.py

"""
[职责] persist_milvus：将 embedding 结果写入 Milvus（可检索/可过滤/可审计）。
[边界] 不负责 embedding 生成；不写入 SQL；不管理 collection/index 生命周期。
[上游关系] ingest/pipeline.py 构造 entities 后调用；或由服务层传入 embedding 产物。
[下游关系] retrieval/vector.py 依赖向量实体；ingest_gate 依赖写入成功。
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

from uae_law_rag.backend.kb.schema import (
    ARTICLE_ID_FIELD,
    DOCUMENT_ID_FIELD,
    EMBEDDING_FIELD,
    FILE_ID_FIELD,
    KB_ID_FIELD,
    NODE_ID_FIELD,
    PAGE_FIELD,
    SECTION_PATH_FIELD,
    VECTOR_ID_FIELD,
)


REQUIRED_FIELDS = [
    VECTOR_ID_FIELD,
    EMBEDDING_FIELD,
    NODE_ID_FIELD,
    KB_ID_FIELD,
    FILE_ID_FIELD,
    DOCUMENT_ID_FIELD,
    ARTICLE_ID_FIELD,
    SECTION_PATH_FIELD,
]  # docstring: Milvus payload 必填字段（与 kb/schema 对齐）


def _normalize_entity(entity: Dict[str, Any], *, embed_dim: Optional[int] = None) -> Dict[str, Any]:
    """
    [职责] 归一化 Milvus entity（字段齐全、类型稳定）。
    [边界] 不校验 Milvus 连接；不做字段映射以外的业务逻辑。
    [上游关系] upsert 调用。
    [下游关系] milvus_repo.upsert_embeddings 输入。
    """
    for key in REQUIRED_FIELDS:
        if key not in entity:
            raise ValueError(f"missing field: {key}")  # docstring: 必填字段缺失即失败

    vector_id = str(entity.get(VECTOR_ID_FIELD) or "")  # docstring: vector 主键
    node_id = str(entity.get(NODE_ID_FIELD) or "")  # docstring: node_id 引用
    kb_id = str(entity.get(KB_ID_FIELD) or "")  # docstring: kb 作用域
    file_id = str(entity.get(FILE_ID_FIELD) or "")  # docstring: file 作用域
    document_id = str(entity.get(DOCUMENT_ID_FIELD) or "")  # docstring: document 作用域

    if not vector_id or not node_id or not kb_id or not file_id or not document_id:
        raise ValueError("id fields must be non-empty strings")  # docstring: 确保关键引用有效

    embedding = entity.get(EMBEDDING_FIELD)
    if not isinstance(embedding, list) or not embedding:
        raise ValueError("embedding must be a non-empty list")  # docstring: 向量不能为空
    vector = [float(x) for x in embedding]  # docstring: 统一向量类型为 float

    if embed_dim is not None and len(vector) != int(embed_dim):
        raise ValueError(f"embedding dim mismatch: {len(vector)} != {embed_dim}")  # docstring: 维度一致性校验

    page = entity.get(PAGE_FIELD)
    if page is None:
        page = 0  # docstring: page 缺失时用 0 表示未知
    page_val = int(page)  # docstring: 统一页码类型

    article_id = str(entity.get(ARTICLE_ID_FIELD) or "")  # docstring: 法条编号（允许空）
    section_path = str(entity.get(SECTION_PATH_FIELD) or "")  # docstring: 结构路径（允许空）

    out = {
        VECTOR_ID_FIELD: vector_id,
        EMBEDDING_FIELD: vector,
        NODE_ID_FIELD: node_id,
        KB_ID_FIELD: kb_id,
        FILE_ID_FIELD: file_id,
        DOCUMENT_ID_FIELD: document_id,
        PAGE_FIELD: page_val,
        ARTICLE_ID_FIELD: article_id,
        SECTION_PATH_FIELD: section_path,
    }
    # 透传额外字段（用于过滤/审计），但不得覆盖固定字段
    for k, v in entity.items():
        if k not in out:
            out[k] = v

    return out


def _normalize_entities(entities: Sequence[Dict[str, Any]], *, embed_dim: Optional[int] = None) -> List[Dict[str, Any]]:
    """
    [职责] 批量归一化 entity 列表。
    [边界] 不做去重；不做排序。
    [上游关系] upsert 调用。
    [下游关系] milvus_repo.upsert_embeddings。
    """
    return [_normalize_entity(e, embed_dim=embed_dim) for e in entities]  # docstring: 逐条规范化


async def upsert(
    *,
    milvus_repo: Any,
    collection: str,
    entities: Sequence[Dict[str, Any]],
    embed_dim: Optional[int] = None,
) -> None:
    """
    [职责] 将规范化后的 entities 写入 Milvus collection。
    [边界] 不创建 collection；不提交 SQL；不处理重试策略。
    [上游关系] ingest/pipeline.py 调用（persist_milvus 阶段）。
    [下游关系] Milvus 向量检索与一致性校验。
    """
    if not entities:
        return  # docstring: 空输入直接跳过

    normalized = _normalize_entities(entities, embed_dim=embed_dim)  # docstring: 对齐字段与类型
    if not hasattr(milvus_repo, "upsert_embeddings"):
        raise AttributeError("milvus_repo.upsert_embeddings is required")  # docstring: 强制 repo 接口

    await milvus_repo.upsert_embeddings(collection=collection, entities=normalized)  # docstring: 写入 Milvus
