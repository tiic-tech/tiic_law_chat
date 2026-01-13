# playground/milvus_gate/test_milvus_integration_gate.py

"""
[职责] Milvus integration gate：验证 client/index/repo 的最小对外接口与行为（需 Milvus 可用）。
[边界] 只做 “创建 collection → 建索引/加载 → upsert → search → delete” 最小闭环；
       不引入 fusion/rerank；不依赖 LlamaIndex。
[上游关系] 依赖 kb/schema.py 的 collection spec；依赖 Milvus 运行环境。
[下游关系] ingest/persist_milvus 与 retrieval/vector 将复用 repo API；record 层将消费 search 输出。
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import List

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]  # docstring: 定位项目根目录
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))  # docstring: 优先使用本地源码而非 site-packages

from uae_law_rag.backend.kb.schema import build_collection_spec, build_expr_for_scope


pytestmark = pytest.mark.milvus_gate


def _milvus_env() -> dict:
    """
    Minimal env contract for tests.

    Supported patterns:
      - MILVUS_URI (recommended): e.g. http://localhost:19530
      - MILVUS_URL (alias for MILVUS_URI)
      - or MILVUS_HOST + MILVUS_PORT
    """  # docstring: 让 gate tests 可在 CI/本地可控启用
    uri = os.getenv("MILVUS_URI", "").strip()
    url = os.getenv("MILVUS_URL", "").strip()  # docstring: 兼容 infra/compose 常用变量
    if not uri and url:
        uri = url  # docstring: 将 URL 视为 URI
        os.environ["MILVUS_URI"] = uri  # docstring: 兼容 MilvusClient.from_env
    host = os.getenv("MILVUS_HOST", "").strip()
    port = os.getenv("MILVUS_PORT", "").strip()
    return {"uri": uri, "host": host, "port": port}


def _should_skip() -> bool:
    env = _milvus_env()
    if env["uri"]:
        return False
    if env["host"] and env["port"]:
        return False
    return True


@pytest.mark.asyncio
async def test_milvus_gate_collection_lifecycle() -> None:
    """
    Contract test (expected APIs):

    kb.client:
      - get_client(...)
      - healthcheck()
      - create_collection(spec, drop_if_exists=True)
      - drop_collection(name)

    kb.index:
      - ensure_index(spec)
      - load_collection(name)
      - release_collection(name)

    This test drives those APIs.
    """  # docstring: 反推 client/index.py 的最小接口
    if _should_skip():
        pytest.skip("Milvus env not configured. Set MILVUS_URI/MILVUS_URL or MILVUS_HOST/MILVUS_PORT.")

    # ---- expected imports (to be implemented next) ----
    from uae_law_rag.backend.kb.client import MilvusClient  # type: ignore
    from uae_law_rag.backend.kb.index import MilvusIndexManager  # type: ignore

    spec = build_collection_spec(
        name=f"kb_gate_{int(time.time())}",
        embed_dim=4,
        metric_type="COSINE",
        index_type="HNSW",
        default_top_k=5,
    )

    client = MilvusClient.from_env(force_reconnect=True)  # docstring: 从环境变量初始化连接
    idx = MilvusIndexManager(client)
    try:
        await client.healthcheck()  # docstring: 必须可用，否则 gate fail

        # create/drop idempotent
        await client.create_collection(spec, drop_if_exists=True)  # docstring: 创建契约一致的 collection
        await idx.ensure_index(spec)  # docstring: 为 embedding 字段建索引
        await idx.load_collection(spec.name)  # docstring: load 后才可 search
    finally:
        try:
            await idx.release_collection(spec.name)  # docstring: 生命周期可控
        except Exception:
            pass
        try:
            await client.drop_collection(spec.name)  # docstring: 清理资源
        except Exception:
            pass
        client.disconnect()  # docstring: 释放 Milvus alias


@pytest.mark.asyncio
async def test_milvus_gate_upsert_search_delete_roundtrip() -> None:
    """
    Contract test (expected APIs):

    kb.repo:
      - upsert_embeddings(collection, entities)
      - search(collection, query_vectors, top_k, expr=None, output_fields=None)
      - delete_by_expr(collection, expr)

    This test drives those APIs.
    """  # docstring: 反推 repo.py 的最小接口与返回结构
    if _should_skip():
        pytest.skip("Milvus env not configured. Set MILVUS_URI/MILVUS_URL or MILVUS_HOST/MILVUS_PORT.")

    from uae_law_rag.backend.kb.client import MilvusClient  # type: ignore
    from uae_law_rag.backend.kb.index import MilvusIndexManager  # type: ignore
    from uae_law_rag.backend.kb.repo import MilvusRepo  # type: ignore

    spec = build_collection_spec(
        name=f"kb_repo_gate_{int(time.time())}",
        embed_dim=4,
        metric_type="COSINE",
        index_type="HNSW",
        default_top_k=3,
    )

    client = MilvusClient.from_env(force_reconnect=True)
    idx = MilvusIndexManager(client)
    try:
        await client.create_collection(spec, drop_if_exists=True)

        await idx.ensure_index(spec)
        await idx.load_collection(spec.name)

        repo = MilvusRepo(client)

        # --- upsert: entities must match schema payload keys ---
        entities: List[dict] = [
            {
                "vector_id": "v1",
                "embedding": [1.0, 0.0, 0.0, 0.0],
                "node_id": "n1",
                "kb_id": "KB1",
                "file_id": "F1",
                "document_id": "D1",
                "page": 1,
                "article_id": "Article 1",
                "section_path": "Chapter 1",
            },
            {
                "vector_id": "v2",
                "embedding": [0.0, 1.0, 0.0, 0.0],
                "node_id": "n2",
                "kb_id": "KB1",
                "file_id": "F1",
                "document_id": "D1",
                "page": 2,
                "article_id": "Article 2",
                "section_path": "Chapter 1",
            },
        ]
        await repo.upsert_embeddings(collection=spec.name, entities=entities)  # docstring: 插入/更新向量实体

        # --- search within KB scope ---
        expr = build_expr_for_scope(kb_id="KB1")
        res = await repo.search(
            collection=spec.name,
            query_vectors=[[1.0, 0.0, 0.0, 0.0]],
            top_k=2,
            expr=expr,
            output_fields=spec.search.output_fields,
        )

        # expected return:
        # list[ list[ {"vector_id":..., "score":..., "payload":{...}} ] ]
        assert isinstance(res, list)
        assert len(res) == 1
        assert len(res[0]) >= 1
        top = res[0][0]
        assert "vector_id" in top
        assert "score" in top
        assert "payload" in top
        assert top["payload"]["kb_id"] == "KB1"

        # --- delete by expr and verify empty ---
        await repo.delete_by_expr(collection=spec.name, expr=expr)  # docstring: 按作用域清理（file/kb 重建用）

        res2 = await repo.search(
            collection=spec.name,
            query_vectors=[[1.0, 0.0, 0.0, 0.0]],
            top_k=2,
            expr=expr,
            output_fields=spec.search.output_fields,
        )
        assert len(res2[0]) == 0
    finally:
        try:
            await client.drop_collection(spec.name)
        except Exception:
            pass
        client.disconnect()  # docstring: 释放 Milvus alias
