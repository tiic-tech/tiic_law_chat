#!/usr/bin/env python3
# playground/generation_gate/test_generation_gate.py

"""
[职责] generation_gate：验证生成链路（prompt/generator/postprocess/persist）的最小可信门槛。
[边界] 仅做离线可验证闭环；不依赖外部 LLM；不评估答案质量。
[上游关系] 依赖 DB repos + generation pipeline；使用 mock provider 执行生成。
[下游关系] 保障 evaluator/chat 在可信 citations 基础上运行。
"""

from __future__ import annotations

import time
from typing import List, Set, cast

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from uae_law_rag.backend.db.repo import (
    ConversationRepo,
    GenerationRepo,
    IngestRepo,
    MessageRepo,
    RetrievalRepo,
    UserRepo,
)
from uae_law_rag.backend.pipelines.base.context import PipelineContext
from uae_law_rag.backend.pipelines.generation import pipeline as pipeline_mod
from uae_law_rag.backend.schemas.ids import KnowledgeBaseId, MessageId, NodeId, RetrievalRecordId, UUIDStr
from uae_law_rag.backend.schemas.retrieval import (
    RetrievalBundle,
    RetrievalHit,
    RetrievalRecord,
    RetrievalTimingMs,
)


pytestmark = pytest.mark.generation_gate


def _hit_nodes(hits: List[RetrievalHit]) -> Set[str]:
    """
    [职责] 提取命中列表中的 node_id 集合。
    [边界] 仅做集合转换；不校验 UUID 合法性。
    [上游关系] gate 断言调用。
    [下游关系] citations 子集断言。
    """
    return {str(h.node_id) for h in hits}  # docstring: node_id 去重集合


@pytest.mark.asyncio
async def test_generation_gate_end_to_end(session: AsyncSession) -> None:
    """
    [职责] 验证生成链路的最小可用闭环（prompt→LLM→postprocess→persist）。
    [边界] 不依赖外部 LLM；使用 mock provider；不覆盖 evaluator。
    [上游关系] 依赖 repos 与 generation pipeline。
    [下游关系] 为 generation_gate 提供行为锁定。
    """
    user_repo = UserRepo(session)
    ingest_repo = IngestRepo(session)
    conv_repo = ConversationRepo(session)
    msg_repo = MessageRepo(session)
    retrieval_repo = RetrievalRepo(session)
    generation_repo = GenerationRepo(session)

    u = await user_repo.create(username="generation_u")  # docstring: 创建用户
    kb = await ingest_repo.create_kb(
        user_id=u.id,
        kb_name="generation_kb",
        milvus_collection=f"kb_generation_{int(time.time())}",
        embed_model="bge-m3",
        embed_dim=4,
    )  # docstring: 创建 KB
    conv = await conv_repo.create(
        user_id=u.id,
        chat_type="chat",
        default_kb_id=kb.id,
        name="generation_conv",
        settings={},
    )  # docstring: 创建会话
    msg = await msg_repo.create_user_message(
        conversation_id=conv.id,
        chat_type="chat",
        query="What is the scope of application?",
    )  # docstring: 创建 message

    f = await ingest_repo.create_file(
        kb_id=kb.id,
        file_name="generation.pdf",
        file_ext="pdf",
        sha256="d" * 64,
        source_uri="file://generation.pdf",
        file_version=1,
        file_mtime=0.0,
        file_size=10,
        pages=1,
        ingest_profile={"parser": "pymupdf4llm"},
    )  # docstring: 创建 file
    doc = await ingest_repo.create_document(
        kb_id=kb.id,
        file_id=f.id,
        title="Generation Gate Doc",
        source_name="generation.pdf",
        meta_data={},
    )  # docstring: 创建 document
    nodes = await ingest_repo.bulk_create_nodes(
        document_id=doc.id,
        nodes=[
            {
                "node_index": 0,
                "text": "Article 2: Scope of application for the regulation.",
                "page": 1,
                "start_offset": 0,
                "end_offset": 55,
                "article_id": "Article 2",
                "section_path": "Chapter 1",
                "meta_data": {"text": "Article 2: Scope of application for the regulation."},
            }
        ],
    )  # docstring: 创建节点

    record = await retrieval_repo.create_record(
        message_id=msg.id,
        kb_id=kb.id,
        query_text="What is the scope of application?",
        keyword_top_k=10,
        vector_top_k=5,
        fusion_top_k=5,
        rerank_top_k=3,
        fusion_strategy="union",
        rerank_strategy="none",
        provider_snapshot={"retrieval": {"kb_id": kb.id}},
        timing_ms={"total": 1.0},
    )  # docstring: 创建 retrieval_record
    await retrieval_repo.bulk_create_hits(
        retrieval_record_id=record.id,
        hits=[
            {
                "node_id": nodes[0].id,
                "source": "fused",
                "rank": 1,
                "score": 0.9,
                "score_details": {"mock_score": 0.9},
                "excerpt": "Article 2: Scope of application for the regulation.",
                "page": 1,
                "start_offset": 0,
                "end_offset": 55,
            }
        ],
    )  # docstring: 创建 retrieval_hit

    retrieval_record = RetrievalRecord(
        id=cast(RetrievalRecordId, UUIDStr(str(record.id))),  # docstring: retrieval_record_id
        message_id=cast(MessageId, UUIDStr(str(msg.id))),  # docstring: message_id
        kb_id=cast(KnowledgeBaseId, UUIDStr(str(kb.id))),  # docstring: kb_id
        query_text="What is the scope of application?",
        keyword_top_k=10,
        vector_top_k=5,
        fusion_top_k=5,
        rerank_top_k=3,
        fusion_strategy="union",
        rerank_strategy="none",
        provider_snapshot={"retrieval": {"kb_id": kb.id}},
        timing_ms=RetrievalTimingMs(total=1.0),
    )  # docstring: 构造 RetrievalRecord schema
    hits = [
        RetrievalHit(
            retrieval_record_id=cast(RetrievalRecordId, UUIDStr(str(record.id))),  # docstring: 归属检索记录
            node_id=cast(NodeId, UUIDStr(str(nodes[0].id))),  # docstring: 证据节点ID
            source="fused",
            rank=1,
            score=0.9,
            score_details={"mock_score": 0.9},
            excerpt="Article 2: Scope of application for the regulation.",
            page=1,
            start_offset=0,
            end_offset=55,
        )
    ]  # docstring: 构造 RetrievalHit schema 列表
    retrieval_bundle = RetrievalBundle(record=retrieval_record, hits=hits)  # docstring: 组装 RetrievalBundle

    ctx = PipelineContext.from_session(session)  # docstring: 构造 pipeline ctx
    config = {
        "prompt_name": "uae_law_default",
        "prompt_version": "v1",
        "model_provider": "mock",
        "model_name": "mock",
        "generation_config": {"temperature": 0.0},
        "postprocess_config": {"require_citations": True, "strict_json": True},
    }  # docstring: generation config

    bundle = await pipeline_mod.run_generation_pipeline(
        session=session,
        generation_repo=generation_repo,
        message_id=msg.id,
        retrieval_bundle=retrieval_bundle,
        config=config,
        ctx=ctx,
    )  # docstring: 执行 generation pipeline

    rec = await generation_repo.get_record_by_message(msg.id)  # docstring: 读取 generation_record
    assert rec is not None
    assert rec.id == bundle.record.id  # docstring: record_id 对齐
    assert rec.output_raw and rec.output_raw.strip()  # docstring: output_raw 非空
    assert rec.status in {"success", "partial", "failed"}  # docstring: status 合法
    assert rec.retrieval_record_id == record.id  # docstring: retrieval_record_id 对齐

    hit_nodes = _hit_nodes(hits)  # docstring: 命中 node_id 集合
    citations_payload = rec.citations or {}  # docstring: citations payload
    cited_nodes = set(citations_payload.get("nodes") or [])  # docstring: 引用 nodes
    assert cited_nodes  # docstring: citations 数量 >= 1
    assert cited_nodes.issubset(hit_nodes)  # docstring: citations ⊆ hits

    structured = rec.output_structured or {}  # docstring: 结构化输出
    assert isinstance(structured, dict)  # docstring: output_structured 应为 dict
    assert "answer" in structured  # docstring: answer 字段存在

    snapshot = rec.messages_snapshot or {}  # docstring: messages_snapshot
    timing_ms = snapshot.get("timing_ms") or {}  # docstring: timing 快照
    assert "llm" in timing_ms  # docstring: timing 包含 llm
    assert "postprocess" in timing_ms  # docstring: timing 包含 postprocess
    assert "total" in timing_ms  # docstring: timing 包含 total
    provider_snapshot = snapshot.get("provider_snapshot") or {}  # docstring: provider 快照
    generation_snapshot = provider_snapshot.get("generation") or {}  # docstring: generation 快照
    assert generation_snapshot.get("model_provider") == "mock"  # docstring: provider 记录
    assert generation_snapshot.get("model_name") == "mock"  # docstring: model 记录
