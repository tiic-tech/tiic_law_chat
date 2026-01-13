# playground/evaluator_gate/test_evaluator_gate.py

"""
[职责] evaluator_gate：验证评估链路（checks/scores/persist）的最小可信门槛。
[边界] 不依赖外部 LLM；不评估模型质量；仅验证评估输出与落库一致性。
[上游关系] 依赖 DB repos + evaluator pipeline；使用 retrieval/generation 快照作为输入。
[下游关系] 保障 API/服务层按评估结果做可信展示与拒答。
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict, cast

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

from uae_law_rag.backend.db.repo import (
    ConversationRepo,
    EvaluatorRepo,
    GenerationRepo,
    IngestRepo,
    MessageRepo,
    RetrievalRepo,
    UserRepo,
)
from uae_law_rag.backend.pipelines.base.context import PipelineContext
from uae_law_rag.backend.pipelines.evaluator import pipeline as pipeline_mod
from uae_law_rag.backend.schemas.evaluator import EvaluatorConfig
from uae_law_rag.backend.schemas.generation import CitationsPayload, GenerationRecord
from uae_law_rag.backend.schemas.ids import (
    GenerationRecordId,
    KnowledgeBaseId,
    MessageId,
    NodeId,
    RetrievalRecordId,
    UUIDStr,
)
from uae_law_rag.backend.schemas.retrieval import RetrievalHit, RetrievalRecord, RetrievalTimingMs


pytestmark = pytest.mark.evaluator_gate


async def _seed_base(
    *,
    user_repo: UserRepo,
    ingest_repo: IngestRepo,
    conv_repo: ConversationRepo,
    username: str,
) -> Dict[str, Any]:
    """
    [职责] 创建 evaluator gate 所需的基础实体（user/kb/conversation/file/doc/nodes）。
    [边界] 不提交事务；仅做最小可复用快照创建。
    [上游关系] gate test 调用。
    [下游关系] _create_eval_case 复用 base 数据。
    """
    u = await user_repo.create(username=username)  # docstring: 创建用户
    kb = await ingest_repo.create_kb(
        user_id=u.id,
        kb_name="evaluator_kb",
        milvus_collection=f"kb_evaluator_{int(time.time())}",
        embed_model="bge-m3",
        embed_dim=4,
    )  # docstring: 创建 KB
    conv = await conv_repo.create(
        user_id=u.id,
        chat_type="chat",
        default_kb_id=kb.id,
        name="evaluator_conv",
        settings={},
    )  # docstring: 创建会话
    f = await ingest_repo.create_file(
        kb_id=kb.id,
        file_name="evaluator.pdf",
        file_ext="pdf",
        sha256="e" * 64,
        source_uri="file://evaluator.pdf",
        file_version=1,
        file_mtime=0.0,
        file_size=10,
        pages=1,
        ingest_profile={"parser": "pymupdf4llm"},
    )  # docstring: 创建 file
    doc = await ingest_repo.create_document(
        kb_id=kb.id,
        file_id=f.id,
        title="Evaluator Gate Doc",
        source_name="evaluator.pdf",
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

    return {"user": u, "kb": kb, "conv": conv, "file": f, "doc": doc, "nodes": nodes}  # docstring: 返回基础快照


async def _create_eval_case(
    *,
    msg_repo: MessageRepo,
    retrieval_repo: RetrievalRepo,
    generation_repo: GenerationRepo,
    base: Dict[str, Any],
    query: str,
    answer: str,
    with_citations: bool,
) -> Dict[str, Any]:
    """
    [职责] 基于 base 快照创建 message/retrieval/generation 并构建 schema 输入。
    [边界] 仅构造最小可评估快照；不触发 pipeline。
    [上游关系] gate test 调用。
    [下游关系] evaluator pipeline 输入使用。
    """
    msg = await msg_repo.create_user_message(
        conversation_id=base["conv"].id,
        chat_type="chat",
        query=query,
    )  # docstring: 创建 message

    record = await retrieval_repo.create_record(
        message_id=msg.id,
        kb_id=base["kb"].id,
        query_text=query,
        keyword_top_k=10,
        vector_top_k=5,
        fusion_top_k=5,
        rerank_top_k=3,
        fusion_strategy="union",
        rerank_strategy="none",
        provider_snapshot={"retrieval": {"kb_id": base["kb"].id}},
        timing_ms={"total": 1.0},
    )  # docstring: 创建 retrieval_record
    await retrieval_repo.bulk_create_hits(
        retrieval_record_id=record.id,
        hits=[
            {
                "node_id": base["nodes"][0].id,
                "source": "fused",
                "rank": 1,
                "score": 0.9,
                # docstring: 增强 - 注入 keyword/vector 信号，避免未来启用 require_* checks 时不稳定
                "score_details": {
                    "mock_score": 0.9,
                    "keyword_score": 0.9,
                    "vector_score": 0.9,
                },
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
        kb_id=cast(KnowledgeBaseId, UUIDStr(str(base["kb"].id))),  # docstring: kb_id
        query_text=str(query),
        keyword_top_k=10,
        vector_top_k=5,
        fusion_top_k=5,
        rerank_top_k=3,
        fusion_strategy="union",
        rerank_strategy="none",
        provider_snapshot={"retrieval": {"kb_id": base["kb"].id}},
        timing_ms=RetrievalTimingMs(total=1.0),
    )  # docstring: 构造 RetrievalRecord schema
    hits = [
        RetrievalHit(
            retrieval_record_id=cast(RetrievalRecordId, UUIDStr(str(record.id))),  # docstring: 归属检索记录
            node_id=cast(NodeId, UUIDStr(str(base["nodes"][0].id))),  # docstring: 证据节点ID
            source="fused",
            rank=1,
            score=0.9,
            # docstring: 增强 - 注入 keyword/vector 信号（与 DB hit 一致）
            score_details={
                "mock_score": 0.9,
                "keyword_score": 0.9,
                "vector_score": 0.9,
            },
            excerpt="Article 2: Scope of application for the regulation.",
            page=1,
            start_offset=0,
            end_offset=55,
        )
    ]  # docstring: 构造 RetrievalHit schema 列表

    node_id = str(base["nodes"][0].id)  # docstring: node_id 字符串
    citation_items = (
        [{"node_id": node_id, "rank": 1, "quote": "Scope of application."}] if with_citations else []
    )  # docstring: citations items
    citation_nodes = [node_id] if with_citations else []  # docstring: citations nodes

    output_structured = {
        "answer": answer,
        "citations": citation_items,
    }  # docstring: 结构化输出
    raw_payload = {
        "answer": answer,
        "citations": citation_items,
    }  # docstring: 输出 payload
    output_raw = json.dumps(raw_payload, ensure_ascii=True)  # docstring: output_raw JSON

    # docstring: 增强 - 使用标准 citations payload 结构，提升 schema 演进兼容性
    citations_payload = {
        "version": "v1",
        "nodes": citation_nodes,
        "items": citation_items,
        "meta": {},
    }  # docstring: citations payload dict
    gen = await generation_repo.create_record(
        message_id=msg.id,
        retrieval_record_id=record.id,
        prompt_name="uae_law_default",
        prompt_version="v1",
        model_provider="mock",
        model_name="mock",
        messages_snapshot={"system": "You are a UAE law assistant", "user": query},
        output_raw=output_raw,
        output_structured=output_structured,
        citations=citations_payload,
        status="success",
    )  # docstring: 创建 generation_record

    citations_schema = CitationsPayload(
        version="v1",
        nodes=[cast(NodeId, UUIDStr(str(node_id)))] if with_citations else [],
        items=list(citation_items),
        meta={},
    )  # docstring: CitationsPayload schema
    generation_record = GenerationRecord(
        id=cast(GenerationRecordId, UUIDStr(str(gen.id))),  # docstring: generation_record_id
        message_id=cast(MessageId, UUIDStr(str(msg.id))),  # docstring: message_id
        retrieval_record_id=cast(RetrievalRecordId, UUIDStr(str(record.id))),  # docstring: retrieval_record_id
        prompt_name="uae_law_default",
        prompt_version="v1",
        model_provider="mock",
        model_name="mock",
        messages_snapshot={"system": "You are a UAE law assistant", "user": query},
        output_raw=output_raw,
        output_structured=output_structured,
        citations=citations_schema,
        status="success",
        error_message=None,
    )  # docstring: 构造 GenerationRecord schema

    return {
        "message": msg,
        "retrieval_record": retrieval_record,
        "retrieval_hits": hits,
        "generation_record": generation_record,
        "retrieval_record_id": record.id,
        "generation_record_id": gen.id,
    }  # docstring: 返回评估案例快照


def _assert_checks_payload(checks_payload: Dict[str, Any]) -> None:
    """
    [职责] 断言 checks payload 结构合法性。
    [边界] 仅检查 name/status 基础字段；不校验业务含义。
    [上游关系] evaluator_gate 调用。
    [下游关系] gate 断言通过后继续检查状态。
    """
    items = checks_payload.get("items") or []  # docstring: checks.items
    assert isinstance(items, list)  # docstring: items 必须是 list
    for item in items:
        assert isinstance(item, dict)  # docstring: gate: checks.items 每项必须为 dict（避免落库异常被吞）
        assert str(item.get("name") or "").strip()  # docstring: name 非空
        assert item.get("status") in {"pass", "fail", "warn", "skipped"}  # docstring: status 合法


@pytest.mark.asyncio
async def test_evaluator_gate_end_to_end(session: AsyncSession) -> None:
    """
    [职责] 验证 evaluator pipeline 的最小可用闭环（checks→scores→persist）。
    [边界] 不测试模型质量；仅验证评估结果结构与落库一致性。
    [上游关系] 依赖 repos 与 evaluator pipeline。
    [下游关系] 为 evaluator_gate 提供行为锁定。
    """
    user_repo = UserRepo(session)
    ingest_repo = IngestRepo(session)
    conv_repo = ConversationRepo(session)
    msg_repo = MessageRepo(session)
    retrieval_repo = RetrievalRepo(session)
    generation_repo = GenerationRepo(session)
    evaluator_repo = EvaluatorRepo(session)

    base = await _seed_base(
        user_repo=user_repo,
        ingest_repo=ingest_repo,
        conv_repo=conv_repo,
        username="evaluator_u_pass",
    )  # docstring: 构建基础快照
    case = await _create_eval_case(
        msg_repo=msg_repo,
        retrieval_repo=retrieval_repo,
        generation_repo=generation_repo,
        base=base,
        query="What is the scope of application?",
        answer="Article 2 defines the scope of application in the regulation.",
        with_citations=True,
    )  # docstring: 构建评估案例

    ctx = PipelineContext.from_session(session)  # docstring: 构造 pipeline ctx
    config = EvaluatorConfig(
        rule_version="v0",
        require_citations=True,
        min_citations=1,
        min_answer_chars=10,
        citation_coverage_threshold=0.5,
    )  # docstring: evaluator config

    result = await pipeline_mod.run_evaluator_pipeline(
        session=session,
        evaluator_repo=evaluator_repo,
        input={
            "conversation_id": base["conv"].id,
            "message_id": case["message"].id,
            "retrieval_record": case["retrieval_record"],
            "retrieval_hits": case["retrieval_hits"],
            "generation_record": case["generation_record"],
            "config": config,
        },
        ctx=ctx,
    )  # docstring: 执行 evaluator pipeline

    record = await evaluator_repo.get_by_message_id(case["message"].id)  # docstring: 读取 evaluation_record
    assert record is not None
    assert record.id == result.id  # docstring: record_id 对齐
    assert record.message_id == case["message"].id  # docstring: message_id 对齐
    assert record.retrieval_record_id == case["retrieval_record_id"]  # docstring: retrieval_record_id 对齐
    assert record.generation_record_id == case["generation_record_id"]  # docstring: generation_record_id 对齐
    assert record.status in {"pass", "fail", "partial", "skipped"}  # docstring: status 合法
    assert record.rule_version == config.rule_version  # docstring: rule_version 稳定

    assert result.status == "pass"  # docstring: 通过完整评估
    assert result.config.rule_version == config.rule_version  # docstring: config 版本一致
    assert result.checks  # docstring: checks 非空
    for chk in result.checks:
        assert chk.name  # docstring: check.name 非空
        assert chk.status in {"pass", "fail", "warn", "skipped"}  # docstring: check.status 合法

    coverage = result.scores.overall.get("citation_coverage")  # docstring: coverage 指标
    assert coverage is not None
    assert 0.0 <= float(coverage) <= 1.0  # docstring: coverage 在区间内

    checks_payload = record.checks or {}  # docstring: checks payload
    items = checks_payload.get("items") or []  # docstring: checks.items
    assert items  # docstring: checks 非空（非 skipped）
    _assert_checks_payload(checks_payload)  # docstring: 校验 checks payload

    scores_payload = record.scores or {}  # docstring: scores payload
    db_coverage = (scores_payload.get("overall") or {}).get("citation_coverage")  # docstring: DB coverage
    assert db_coverage is not None
    assert 0.0 <= float(db_coverage) <= 1.0  # docstring: coverage 在区间内
    assert float(db_coverage) == float(coverage)  # docstring: DB 与 result coverage 一致

    # docstring: 增强 - 基础 meta/trace/timing 字段存在性（可回放审计最小合同）
    meta_payload = record.meta or {}  # docstring: meta payload
    assert isinstance(meta_payload, dict)  # docstring: meta 必须为 dict
    assert str(meta_payload.get("trace_id") or "").strip()  # docstring: trace_id 必填
    assert str(meta_payload.get("request_id") or "").strip()  # docstring: request_id 必填
    timing_ms = meta_payload.get("timing_ms") or {}  # docstring: timing_ms
    assert isinstance(timing_ms, dict)  # docstring: timing_ms 必须为 dict
    assert "total" in timing_ms  # docstring: timing_ms.total 必须存在

    # docstring: 增强 - checks 名称稳定性（至少锁定关键检查项存在）
    check_names = {c.name for c in result.checks}  # docstring: checks 名称集合
    assert "require_citations" in check_names  # docstring: require_citations 必须存在
    assert "citation_coverage" in check_names  # docstring: citation_coverage 必须存在


@pytest.mark.asyncio
async def test_evaluator_gate_require_citations_fail(session: AsyncSession) -> None:
    """
    [职责] 验证 require_citations 失败时整体状态为 fail。
    [边界] 只验证门禁联动；不测试其他规则组合。
    [上游关系] 依赖 repos 与 evaluator pipeline。
    [下游关系] 锁定 require_citations → overall fail 的强制逻辑。
    """
    user_repo = UserRepo(session)
    ingest_repo = IngestRepo(session)
    conv_repo = ConversationRepo(session)
    msg_repo = MessageRepo(session)
    retrieval_repo = RetrievalRepo(session)
    generation_repo = GenerationRepo(session)
    evaluator_repo = EvaluatorRepo(session)

    base = await _seed_base(
        user_repo=user_repo,
        ingest_repo=ingest_repo,
        conv_repo=conv_repo,
        username="evaluator_u_fail",
    )  # docstring: 构建基础快照
    case = await _create_eval_case(
        msg_repo=msg_repo,
        retrieval_repo=retrieval_repo,
        generation_repo=generation_repo,
        base=base,
        query="What is the competent authority?",
        answer="The competent authority is defined in Article 3.",
        with_citations=False,
    )  # docstring: 构建无 citations 案例

    ctx = PipelineContext.from_session(session)  # docstring: 构造 pipeline ctx
    config = EvaluatorConfig(
        rule_version="v0",
        require_citations=True,
        min_citations=1,
        min_answer_chars=5,
    )  # docstring: evaluator config

    result = await pipeline_mod.run_evaluator_pipeline(
        session=session,
        evaluator_repo=evaluator_repo,
        input={
            "conversation_id": base["conv"].id,
            "message_id": case["message"].id,
            "retrieval_record": case["retrieval_record"],
            "retrieval_hits": case["retrieval_hits"],
            "generation_record": case["generation_record"],
            "config": config,
        },
        ctx=ctx,
    )  # docstring: 执行 evaluator pipeline

    record = await evaluator_repo.get_by_message_id(case["message"].id)  # docstring: 读取 evaluation_record
    assert record is not None
    assert record.status == "fail"  # docstring: require_citations 失败 -> overall fail

    assert result.status == "fail"  # docstring: overall fail
    assert result.checks  # docstring: checks 非空
    require_check = next((c for c in result.checks if c.name == "require_citations"), None)  # docstring: 查找检查项
    assert require_check is not None
    assert require_check.status == "fail"  # docstring: require_citations 失败

    checks_payload = record.checks or {}  # docstring: checks payload
    items = checks_payload.get("items") or []  # docstring: checks.items
    assert items  # docstring: checks 非空（非 skipped）
    _assert_checks_payload(checks_payload)  # docstring: 校验 checks payload

    # docstring: 增强 - fail case 仍需落库基本 meta/timing 字段
    meta_payload = record.meta or {}  # docstring: meta payload
    assert str(meta_payload.get("trace_id") or "").strip()  # docstring: trace_id 必填
    timing_ms = meta_payload.get("timing_ms") or {}  # docstring: timing_ms
    assert isinstance(timing_ms, dict)  # docstring: timing_ms 必须为 dict
    assert "total" in timing_ms  # docstring: timing_ms.total 必须存在
