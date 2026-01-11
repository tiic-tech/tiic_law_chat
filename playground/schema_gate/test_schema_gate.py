# playground/schema_gate/test_schema_gate.py

"""
[职责] Schema gate：对 backend/schemas 的“合同”进行综合断言，防止字段漂移、extra 策略漂移、默认值漂移。
[边界] 不触发 DB/Milvus；不测试 pipelines；只测试 Pydantic schema 的结构与约束行为。
[上游关系] backend/schemas/{ids,retrieval,generation,evaluator,chat,audit}。
[下游关系] pipelines/services/db 落库结构将依赖这些合同；本测试用于在进入实现前锁死接口。
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from uae_law_rag.backend.schemas.audit import ProviderSnapshot, TimingSnapshot, TraceContext
from uae_law_rag.backend.schemas.chat import ChatContextConfig, ChatRequest, ChatResponse
from uae_law_rag.backend.schemas.evaluator import (
    EvaluationCheck,
    EvaluationResult,
    EvaluatorConfig,
)
from uae_law_rag.backend.schemas.generation import Citation, GenerationRecord
from uae_law_rag.backend.schemas.ids import is_uuid_str, new_uuid
from uae_law_rag.backend.schemas.retrieval import (
    RetrievalBundle,
    RetrievalHit,
    RetrievalRecord,
    RetrievalTiming,
)


pytestmark = pytest.mark.schema_gate


# -----------------------------
# ids.py
# -----------------------------


def test_ids_new_uuid_is_valid_uuid_str() -> None:
    """new_uuid() must return UUID v4 string-like value."""  # docstring: 基础ID合同
    u = new_uuid()
    assert isinstance(u, str)
    assert len(u) == 36
    assert is_uuid_str(u) is True


def test_ids_is_uuid_str_rejects_non_uuid() -> None:
    """is_uuid_str should reject non-uuid strings."""  # docstring: 基础校验工具行为
    assert is_uuid_str("not-a-uuid") is False
    assert is_uuid_str("") is False


# -----------------------------
# retrieval.py
# -----------------------------


def test_retrieval_timing_allows_extra() -> None:
    """RetrievalTiming is extra=allow so it can accept future breakdown fields."""  # docstring: 允许扩展
    t = RetrievalTiming.model_validate({"total_ms": 1.0, "keyword_ms": 0.2, "custom_stage_ms": 0.3})
    assert t.total_ms == 1.0
    assert getattr(t, "custom_stage_ms") == 0.3


def test_retrieval_record_forbids_extra() -> None:
    """RetrievalRecord must be strict (extra=forbid)."""  # docstring: 锁死合同防漂移
    record = RetrievalRecord(
        id=new_uuid(),
        message_id=new_uuid(),
        kb_id=new_uuid(),
        query_text="Q?",
    )
    assert record.keyword_top_k >= 1
    assert record.fusion_strategy in ("union", "interleave", "weighted")

    with pytest.raises(ValidationError):
        RetrievalRecord(
            id=new_uuid(),
            message_id=new_uuid(),
            kb_id=new_uuid(),
            query_text="Q?",
            extra_field_should_fail=1,  # type: ignore[arg-type]
        )


def test_retrieval_hit_forbids_extra_and_has_required_fields() -> None:
    """RetrievalHit must be strict and contain evidence pointers."""  # docstring: 命中条目最小合同
    hit = RetrievalHit(
        retrieval_record_id=new_uuid(),
        rank=0,
        stage="vector",
        node_id=new_uuid(),
        vector_id=new_uuid(),
        score=0.9,
        snippet="",
        meta={"page": 1},
    )
    assert hit.rank == 0
    assert hit.stage in ("keyword", "vector", "fusion", "rerank")
    assert "page" in hit.meta

    with pytest.raises(ValidationError):
        RetrievalHit(
            retrieval_record_id=new_uuid(),
            rank=0,
            stage="vector",
            node_id=new_uuid(),
            foo="bar",  # type: ignore[arg-type]
        )


def test_retrieval_bundle_contract() -> None:
    """RetrievalBundle must pack record + hits for internal pipeline handoff."""  # docstring: pipeline 内部传输合同
    record = RetrievalRecord(
        id=new_uuid(),
        message_id=new_uuid(),
        kb_id=new_uuid(),
        query_text="What is X?",
    )
    hit = RetrievalHit(
        retrieval_record_id=record.id,
        rank=0,
        stage="keyword",
        node_id=new_uuid(),
        score=1.0,
    )
    bundle = RetrievalBundle(record=record, hits=[hit])
    assert bundle.record.id == record.id
    assert bundle.hits[0].retrieval_record_id == record.id


# -----------------------------
# generation.py
# -----------------------------


def test_citation_allows_extra() -> None:
    """Citation is extra=allow to support future locators (page/article/section)."""  # docstring: 引用定位信息可扩展
    c = Citation.model_validate(
        {
            "node_id": new_uuid(),
            "rank": 0,
            "quote": "short",
            "locator": {"page": 2},
            "custom_locator": "x",
        }
    )
    assert c.node_id
    assert c.locator["page"] == 2
    assert getattr(c, "custom_locator") == "x"


def test_generation_record_forbids_extra() -> None:
    """GenerationRecord must be strict (extra=forbid)."""  # docstring: 锁死生成记录合同
    gr = GenerationRecord(
        id=new_uuid(),
        message_id=new_uuid(),
        retrieval_record_id=new_uuid(),
        prompt_name="p",
        model_provider="ollama",
        model_name="llama3",
        output_raw="A",
        citations=[Citation(node_id=new_uuid())],
    )
    assert gr.status in ("success", "failed", "partial")
    assert len(gr.citations) == 1

    with pytest.raises(ValidationError):
        GenerationRecord(
            id=new_uuid(),
            message_id=new_uuid(),
            retrieval_record_id=new_uuid(),
            prompt_name="p",
            model_provider="ollama",
            model_name="llama3",
            output_raw="A",
            extra_field_should_fail=123,  # type: ignore[arg-type]
        )


# -----------------------------
# evaluator.py
# -----------------------------


def test_evaluator_config_defaults_and_ranges() -> None:
    """EvaluatorConfig default gates must be stable and within sane ranges."""  # docstring: 规则默认值锁定
    cfg = EvaluatorConfig()
    assert cfg.rule_version == "v0"
    assert cfg.retrieval_topk >= 1
    assert 0.0 <= cfg.citation_coverage_threshold <= 1.0

    with pytest.raises(ValidationError):
        EvaluatorConfig(citation_coverage_threshold=1.5)  # type: ignore[arg-type]


def test_evaluation_check_allows_extra() -> None:
    """EvaluationCheck extra=allow for future structured details."""  # docstring: check detail 可扩展
    chk = EvaluationCheck.model_validate(
        {"name": "require_citations", "status": "pass", "detail": {"min": 1}, "foo": "bar"}
    )
    assert chk.name == "require_citations"
    assert getattr(chk, "foo") == "bar"


def test_evaluation_result_forbids_extra_and_has_refs() -> None:
    """EvaluationResult must be strict and reference message/retrieval/generation IDs."""  # docstring: 评估结果合同
    res = EvaluationResult(
        message_id=new_uuid(),
        retrieval_record_id=new_uuid(),
        generation_record_id=new_uuid(),
        status="pass",
        checks=[EvaluationCheck(name="min_answer_chars", status="pass")],
    )
    assert res.message_id
    assert res.retrieval_record_id
    assert res.config.rule_version

    with pytest.raises(ValidationError):
        EvaluationResult(
            message_id=new_uuid(),
            retrieval_record_id=new_uuid(),
            foo="bar",  # type: ignore[arg-type]
        )


# -----------------------------
# chat.py
# -----------------------------


def test_chat_request_forbids_extra_and_basic_fields() -> None:
    """ChatRequest is strict; query required; context is optional with defaults."""  # docstring: HTTP 输入合同
    req = ChatRequest(query="Q?")
    assert req.chat_type in ("chat", "agent_chat")
    assert req.context.keyword_top_k is None

    with pytest.raises(ValidationError):
        ChatRequest(query="Q?", extra_should_fail=True)  # type: ignore[arg-type]


def test_chat_context_config_allows_extra() -> None:
    """ChatContextConfig extra=allow so UI can send future controls without breaking."""  # docstring: 前端扩展不致崩
    cfg = ChatContextConfig.model_validate({"keyword_top_k": 10, "foo": "bar"})
    assert cfg.keyword_top_k == 10
    assert getattr(cfg, "foo") == "bar"


def test_chat_response_forbids_extra_and_requires_ids() -> None:
    """ChatResponse must be strict and require conversation_id/message_id/kb_id."""  # docstring: HTTP 输出合同
    resp = ChatResponse(
        conversation_id=new_uuid(),
        message_id=new_uuid(),
        kb_id=new_uuid(),
        answer="A",
    )
    assert resp.answer == "A"
    assert resp.citations == []

    with pytest.raises(ValidationError):
        ChatResponse(
            conversation_id=new_uuid(),
            message_id=new_uuid(),
            kb_id=new_uuid(),
            extra_should_fail=1,  # type: ignore[arg-type]
        )


# -----------------------------
# audit.py
# -----------------------------


def test_trace_context_defaults() -> None:
    """TraceContext must auto-generate trace_id/request_id by default."""  # docstring: 可观测性基础合同
    tc = TraceContext()
    assert is_uuid_str(tc.trace_id)
    assert is_uuid_str(tc.request_id)


def test_provider_snapshot_requires_core_fields_and_allows_extra() -> None:
    """ProviderSnapshot must require kind/provider/name and allow extra params."""  # docstring: provider 快照合同
    ps = ProviderSnapshot.model_validate(
        {
            "kind": "llm",
            "provider": "ollama",
            "name": "llama3",
            "params": {"t": 0.1},
            "foo": "bar",
        }
    )
    assert ps.kind == "llm"
    assert ps.params["t"] == 0.1
    assert getattr(ps, "foo") == "bar"

    with pytest.raises(ValidationError):
        ProviderSnapshot(kind="llm", provider="ollama")  # type: ignore[call-arg]


def test_timing_snapshot_allows_extra() -> None:
    """TimingSnapshot is extensible for future breakdown keys."""  # docstring: timing 扩展合同
    ts = TimingSnapshot.model_validate({"total_ms": 10.0, "breakdown": {"retrieval": 3.0}, "custom": 1})
    assert ts.total_ms == 10.0
    assert ts.breakdown["retrieval"] == 3.0
    assert getattr(ts, "custom") == 1
