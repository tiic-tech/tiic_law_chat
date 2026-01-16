# src/uae_law_rag/backend/kb/repo.py

"""
[职责] Milvus 数据访问仓储：封装向量实体的 upsert/search/delete，作为向量侧的唯一数据接口。
[边界] 不负责 SQL 写入（由 ingest_repo 负责）；不负责 fusion/rerank；不做 LlamaIndex 依赖。
[上游关系] kb/client.py 提供 MilvusClient；kb/schema.py 定义字段契约与默认 search/output_fields。
[下游关系] pipelines/ingest/persist_milvus.py 使用 upsert；pipelines/retrieval/vector.py 使用 search；
         pipelines/retrieval/record.py 会消费 search 返回结果写入 RetrievalHitModel。
"""

from __future__ import annotations

import inspect
from typing import Any, Dict, List, Optional

from .client import MilvusClient


class MilvusRepo:
    """
    Repository for Milvus vector storage operations.
    """

    def __init__(self, client: MilvusClient) -> None:
        self._client = client  # docstring: MilvusClient（连接与 collection 管理封装）

    async def upsert_embeddings(self, *, collection: str, entities: List[Dict[str, Any]]) -> None:
        """
        Upsert vector entities into collection.

        Args:
          collection: collection name
          entities: list of dict, keys must match schema fields (vector_id, embedding, node_id, ...)
        """  # docstring: ingest 阶段写入 Milvus 的最小接口
        col = await self._client.get_collection(collection)  # docstring: collection 句柄

        # pymilvus insert expects column-based dict or list-of-dict depending on version.
        # We normalize to list-of-dict and rely on pymilvus to handle it; if needed later, convert to columns.
        call = col.insert(entities)  # docstring: 写入实体
        await self._maybe_await(call)

        # flush to guarantee visibility in subsequent search in tests
        await self._maybe_await(col.flush())

    async def search(
        self,
        *,
        collection: str,
        query_vectors: List[List[float]],
        top_k: int,
        expr: Optional[str] = None,
        output_fields: Optional[List[str]] = None,
        metric_type: Optional[str] = None,
        search_params: Optional[Dict[str, Any]] = None,
    ) -> List[List[Dict[str, Any]]]:
        """
        Vector search.

        Returns:
          list per query vector; each item is list of dict:
            {"vector_id": str, "score": float, "payload": dict}
        """  # docstring: retrieval/vector.py 的最小接口；与 gate test 结构对齐
        col = await self._client.get_collection(collection)  # docstring: collection 句柄

        # pymilvus search signature: data, anns_field, param, limit, expr, output_fields
        fields = list(output_fields or [])
        if "node_id" not in fields:
            fields.insert(0, "node_id")

        mt = str(metric_type or "COSINE").strip().upper()
        params = dict(search_params or {"ef": 128, "nprobe": 16})
        call = col.search(
            data=query_vectors,  # docstring: query vectors
            anns_field="embedding",  # docstring: 向量字段名（与 schema 常量一致）
            param={"metric_type": mt, "params": params},  # docstring: 可接受外部传参
            limit=int(top_k),  # docstring: top_k
            expr=expr,  # docstring: 过滤表达式（kb/file/doc scope）
            output_fields=fields,  # docstring: 返回的 payload 字段
        )
        raw = await self._maybe_await(call)

        # Normalize results
        out: List[List[Dict[str, Any]]] = []
        for hits in raw:
            q_res: List[Dict[str, Any]] = []
            for h in hits:
                # hit.id is PK (vector_id) if primary key is VARCHAR
                vector_id = getattr(h, "id", None)  # docstring: 主键
                score = getattr(h, "score", None)  # docstring: 相似度/距离（由 Milvus 返回）
                entity = getattr(h, "entity", None)  # docstring: payload 容器
                payload: Dict[str, Any] = {}
                if entity is not None:
                    # entity can behave like a dict with get()
                    for f in fields:
                        try:
                            payload[f] = entity.get(f)
                        except Exception:
                            # fallback to attribute
                            payload[f] = getattr(entity, f, None)
                q_res.append(
                    {
                        "vector_id": str(vector_id) if vector_id is not None else "",
                        "score": float(score) if score is not None else 0.0,
                        "payload": payload,
                    }
                )
            out.append(q_res)
        return out

    async def delete_by_expr(self, *, collection: str, expr: str) -> None:
        """
        Delete entities by boolean expression.

        Typical use:
          - delete all vectors for kb_id / file_id scope
        """  # docstring: ingest 重建/文件删除的最小接口
        col = await self._client.get_collection(collection)  # docstring: collection 句柄
        call = col.delete(expr)  # docstring: 按表达式删除
        await self._maybe_await(call)
        await self._maybe_await(col.flush())

    @staticmethod
    async def _maybe_await(value: Any) -> Any:
        """
        Normalize pymilvus calls across versions: some return plain values, others return awaitables.
        """  # docstring: 兼容不同 pymilvus sync/async 形态
        if inspect.isawaitable(value):
            return await value
        return value
