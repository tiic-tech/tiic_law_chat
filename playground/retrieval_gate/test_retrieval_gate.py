#!/usr/bin/env python3
# playground/retrieval_gate/test_retrieval_gate.py

"""
[职责] retrieval_gate：验证检索全链路（keyword/vector/fusion/rerank/persist）的最小可信门槛。
[边界] 只做最小可验证闭环；不覆盖 LLM 生成；依赖 Milvus 环境可用。
[上游关系] 依赖 db/fts + repo + Milvus client/index/repo + retrieval pipeline 实现。
[下游关系] 保障 generation/evaluator 在可信 evidence 基础上运行。
"""

from __future__ import annotations

import os
import time
from typing import Any, Dict, List

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from uae_law_rag.backend.db.fts import ensure_sqlite_fts
from uae_law_rag.backend.db.repo import ConversationRepo, IngestRepo, MessageRepo, RetrievalRepo, UserRepo
from uae_law_rag.backend.kb.client import MilvusClient
from uae_law_rag.backend.kb.index import MilvusIndexManager
from uae_law_rag.backend.kb.repo import MilvusRepo
from uae_law_rag.backend.kb.schema import build_collection_spec
from uae_law_rag.backend.pipelines.base.context import PipelineContext
from uae_law_rag.backend.pipelines.retrieval import fusion as fusion_mod
from uae_law_rag.backend.pipelines.retrieval import keyword as keyword_mod
from uae_law_rag.backend.pipelines.retrieval import pipeline as pipeline_mod
from uae_law_rag.backend.pipelines.retrieval import vector as vector_mod


pytestmark = pytest.mark.retrieval_gate


def _milvus_env() -> Dict[str, str]:
    """
    [职责] 读取 Milvus 连接环境变量。
    [边界] 仅解析 env；不校验连通性。
    [上游关系] gate test 调用。
    [下游关系] _should_skip 判定。
    """
    uri = os.getenv("MILVUS_URI", "").strip()
    url = os.getenv("MILVUS_URL", "").strip()  # docstring: 兼容 infra/compose 常用变量
    if not uri and url:
        uri = url  # docstring: 将 URL 视为 URI
        os.environ["MILVUS_URI"] = uri  # docstring: 兼容 MilvusClient.from_env
    host = os.getenv("MILVUS_HOST", "").strip()
    port = os.getenv("MILVUS_PORT", "").strip()
    return {"uri": uri, "host": host, "port": port}


def _should_skip() -> bool:
    """
    [职责] 判断是否跳过 Milvus 相关 gate。
    [边界] 仅依据 env；不做网络探测。
    [上游关系] gate test 调用。
    [下游关系] pytest.skip。
    """
    env = _milvus_env()
    if env["uri"]:
        return False
    if env["host"] and env["port"]:
        return False
    return True


@pytest.mark.asyncio
async def test_retrieval_gate_end_to_end(session: AsyncSession) -> None:
    """
    [职责] 验证 keyword/vector/fusion/persist 的最小可用闭环。
    [边界] 不测试 rerank 模型效果；仅验证可调用与输出结构。
    [上游关系] 依赖 SQLite FTS 与 Milvus。
    [下游关系] 为 generation/evaluator 提供可信 evidence 底座。
    """
    if _should_skip():
        pytest.skip(
            "Milvus env not configured. Set MILVUS_URI/MILVUS_URL or MILVUS_HOST/MILVUS_PORT."
        )  # docstring: 环境缺失

    await ensure_sqlite_fts(session)  # docstring: 确保 FTS 虚表与触发器

    user_repo = UserRepo(session)
    ingest_repo = IngestRepo(session)
    conv_repo = ConversationRepo(session)
    msg_repo = MessageRepo(session)
    retrieval_repo = RetrievalRepo(session)

    u = await user_repo.create(username="retrieval_u")  # docstring: 创建用户
    kb = await ingest_repo.create_kb(
        user_id=u.id,
        kb_name="retrieval_kb",
        milvus_collection=f"kb_retrieval_{int(time.time())}",
        embed_model="bge-m3",
        embed_dim=4,
    )  # docstring: 创建 KB
    conv = await conv_repo.create(
        user_id=u.id,
        chat_type="chat",
        default_kb_id=kb.id,
        name="retrieval_conv",
        settings={},
    )  # docstring: 创建会话
    msg = await msg_repo.create_user_message(
        conversation_id=conv.id,
        chat_type="chat",
        query="Scope of application",
    )  # docstring: 创建 message

    f = await ingest_repo.create_file(
        kb_id=kb.id,
        file_name="retrieval.pdf",
        file_ext="pdf",
        sha256="c" * 64,
        source_uri="file://retrieval.pdf",
        file_version=1,
        file_mtime=0.0,
        file_size=10,
        pages=1,
        ingest_profile={"parser": "pymupdf4llm"},
    )  # docstring: 创建 file
    doc = await ingest_repo.create_document(
        kb_id=kb.id,
        file_id=f.id,
        title="Retrieval Gate Doc",
        source_name="retrieval.pdf",
        meta_data={},
    )  # docstring: 创建 document
    nodes = await ingest_repo.bulk_create_nodes(
        document_id=doc.id,
        nodes=[
            {
                "node_index": 0,
                "text": "Article 2: Scope of application for real beneficiary procedures.",
                "page": 1,
                "start_offset": 0,
                "end_offset": 70,
                "article_id": "Article 2",
                "section_path": "Chapter 1",
                "meta_data": {"text": "Article 2: Scope of application for real beneficiary procedures."},
            },
            {
                "node_index": 1,
                "text": "Article 3: Competent authority and obligations.",
                "page": 1,
                "start_offset": 71,
                "end_offset": 120,
                "article_id": "Article 3",
                "section_path": "Chapter 1",
                "meta_data": {"text": "Article 3: Competent authority and obligations."},
            },
        ],
    )  # docstring: 创建节点
    await session.commit()  # docstring: 确保 FTS 触发器落库

    spec = build_collection_spec(
        name=kb.milvus_collection,
        embed_dim=4,
        metric_type="COSINE",
        index_type="HNSW",
        default_top_k=3,
    )  # docstring: 构建 Milvus collection 规范

    client = MilvusClient.from_env(force_reconnect=True)  # docstring: 初始化 Milvus client
    try:
        await client.create_collection(spec, drop_if_exists=True)  # docstring: 创建 collection
        idx = MilvusIndexManager(client)
        await idx.ensure_index(spec)  # docstring: 创建索引
        await idx.load_collection(spec.name)  # docstring: load collection

        repo = MilvusRepo(client)
        entities: List[Dict[str, Any]] = [
            {
                "vector_id": "v1",
                "embedding": [1.0, 0.0, 0.0, 0.0],
                "node_id": nodes[0].id,
                "kb_id": kb.id,
                "file_id": f.id,
                "document_id": doc.id,
                "page": 1,
                "article_id": "Article 2",
                "section_path": "Chapter 1",
            },
            {
                "vector_id": "v2",
                "embedding": [0.0, 1.0, 0.0, 0.0],
                "node_id": nodes[1].id,
                "kb_id": kb.id,
                "file_id": f.id,
                "document_id": doc.id,
                "page": 1,
                "article_id": "Article 3",
                "section_path": "Chapter 1",
            },
        ]  # docstring: Milvus payload
        await repo.upsert_embeddings(collection=spec.name, entities=entities)  # docstring: 写入向量
        # docstring: ensure newly upserted data is queryable (avoid eventual consistency flakes)
        await idx.load_collection(spec.name)

        kw_hits = await keyword_mod.keyword_recall(
            session=session,
            kb_id=kb.id,
            query="Scope",
            top_k=10,
        )  # docstring: keyword 召回
        assert len(kw_hits) >= 1
        assert any(h.node_id == nodes[0].id for h in kw_hits)

        vec_hits = await vector_mod.vector_recall(
            milvus_repo=repo,
            kb_scope={"kb_id": kb.id, "file_id": f.id},
            query_vector=[1.0, 0.0, 0.0, 0.0],
            top_k=2,
            output_fields=spec.search.output_fields,
            metric_type=spec.search.metric_type,
            collection=spec.name,
        )  # docstring: vector 召回
        assert len(vec_hits) >= 1
        assert any(h.node_id == nodes[0].id for h in vec_hits)
        assert all(h.meta.get("kb_id") == kb.id for h in vec_hits)  # docstring: scope 过滤可回放

        fused = fusion_mod.fuse_candidates(
            keyword=kw_hits,
            vector=vec_hits,
            strategy="union",
            top_k=2,
        )  # docstring: 融合去重
        assert len(fused) >= 1
        assert len({f.node_id for f in fused}) == len(fused)

        ctx = PipelineContext.from_session(session)  # docstring: 构造 pipeline ctx
        config = {
            "keyword_top_k": 10,
            "vector_top_k": 2,
            "fusion_top_k": 2,
            "rerank_top_k": 2,
            "fusion_strategy": "union",
            "rerank_strategy": "none",
            "milvus_collection": spec.name,
            "output_fields": spec.search.output_fields,
            "metric_type": spec.search.metric_type,
        }  # docstring: retrieval config

        bundle = await pipeline_mod.run_retrieval_pipeline(
            session=session,
            milvus_repo=repo,
            retrieval_repo=retrieval_repo,
            message_id=msg.id,
            kb_id=kb.id,
            query_text="Scope of application",
            query_vector=[1.0, 0.0, 0.0, 0.0],
            config=config,
            ctx=ctx,
        )  # docstring: 执行检索 pipeline

        rec = await retrieval_repo.get_record_by_message(msg.id)  # docstring: 回查 RetrievalRecord
        assert rec is not None
        assert rec.id == bundle.record.id
        assert rec.message_id == msg.id
        assert isinstance(rec.timing_ms, dict)
        assert {"keyword", "vector", "fusion", "total"}.issubset(set(rec.timing_ms.keys()))

        hits = await retrieval_repo.list_hits(rec.id)  # docstring: 回查 hits 落库
        assert len(hits) == len(bundle.hits)
        assert len({h.node_id for h in hits}) == len(hits)
        assert all(h.source in {"keyword", "vector", "fused", "reranked"} for h in hits)

        assert bundle.hits  # docstring: bundle hits 非空
        details = bundle.hits[0].score_details or {}
        assert "keyword" in details or "vector" in details  # docstring: 分数细节可解释

        # docstring: ensure fused hit exists and is explainable
        fused_hit = next((h for h in bundle.hits if h.source == "fused"), None)
        assert fused_hit is not None
        fused_details = fused_hit.score_details or {}
        assert fused_details.get("fusion_strategy") == "union"

        # docstring: stronger explainability assertions for keyword/vector
        kw0 = kw_hits[0].score_details or {}
        assert {"bm25", "fts_query", "keyword_mode"}.issubset(set(kw0.keys()))
        vec0 = vec_hits[0].score_details or {}
        assert {"raw_score", "metric_type"}.issubset(set(vec0.keys()))

        # docstring: persist assertions: rank must be contiguous starting at 1
        assert [h.rank for h in hits] == list(range(1, len(hits) + 1))
    finally:
        try:
            await client.drop_collection(spec.name)  # docstring: 清理 Milvus 资源
        except Exception:
            pass
        client.disconnect()  # docstring: 释放 Milvus alias
