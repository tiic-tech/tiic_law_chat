# playground/fastapi_gate/services/test_chat_service_gate.py

"""
[职责] chat_service gate：验证 retrieval/generation/evaluator 全链路的状态机、落库与裁决契约。
[边界] 不测试 HTTP router；不评估 prompt/回答质量；仅验证 Gate 合同与审计锚点。
[上游关系] 依赖 ingest pipeline、SQLite FTS、Milvus 环境与 chat_service。
[下游关系] 保障前端与审计系统可依赖 message.status 与 evaluator 裁决。
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

_REPO_ROOT = Path(__file__).resolve().parents[3]  # docstring: 定位项目根目录
_SRC_ROOT = _REPO_ROOT / "src"  # docstring: 源码目录
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))  # docstring: 优先使用本地源码而非 site-packages

from uae_law_rag.backend.db.fts import ensure_sqlite_fts
from uae_law_rag.backend.db.repo import (
    EvaluatorRepo,
    GenerationRepo,
    IngestRepo,
    MessageRepo,
    RetrievalRepo,
    UserRepo,
)
from uae_law_rag.backend.kb.client import MilvusClient
from uae_law_rag.backend.kb.index import MilvusIndexManager
from uae_law_rag.backend.kb.repo import MilvusRepo
from uae_law_rag.backend.kb.schema import build_collection_spec
from uae_law_rag.backend.pipelines.generation import generator as generator_mod
from uae_law_rag.backend.pipelines.ingest.pipeline import run_ingest_pdf
from uae_law_rag.backend.schemas.audit import TraceContext
from uae_law_rag.backend.services.chat_service import chat
from uae_law_rag.backend.utils.constants import DEBUG_KEY, TIMING_MS_KEY, TIMING_TOTAL_MS_KEY, TRACE_ID_KEY


pytestmark = pytest.mark.fastapi_gate


_ENV_BOOTSTRAPPED = False  # docstring: 防止重复加载 .env


def _ensure_mock_llm_available() -> None:
    """
    [职责] 确认 MockLLM 可用（generation gate 依赖）。
    [边界] 仅做 import 探测；不可用则跳过 gate。
    [上游关系] gate test 调用。
    [下游关系] 保障 generation/evaluator 可执行。
    """
    try:
        from llama_index.core.llms import MockLLM  # type: ignore  # docstring: MockLLM 探测

        _ = MockLLM  # docstring: 显式引用避免 lint 报警
        return
    except Exception:
        try:
            from llama_index.core.llms.mock import MockLLM  # type: ignore  # docstring: 兼容路径探测

            _ = MockLLM  # docstring: 显式引用避免 lint 报警
            return
        except Exception:
            pytest.skip(
                "MockLLM not available; skip chat_service gate (generation/evaluator)."
            )  # docstring: 依赖缺失跳过


def _load_dotenv(path: Path) -> None:
    """
    [职责] 轻量加载 .env 到进程环境变量（无第三方依赖）。
    [边界] 不覆盖已存在的 env；不解析复杂格式。
    [上游关系] _ensure_env_loaded 调用。
    [下游关系] _milvus_env 读取 MILVUS_* 变量。
    """
    if not path.exists():
        return  # docstring: 无 .env 直接跳过
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue  # docstring: 跳过空行与注释
        if line.startswith("export "):
            line = line[len("export ") :].strip()  # docstring: 兼容 export 写法
        if "=" not in line:
            continue  # docstring: 非键值行忽略
        key, value = line.split("=", 1)
        key = key.strip()
        if not key or key in os.environ:
            continue  # docstring: 保留现有 env
        val = value.strip().strip("'").strip('"')  # docstring: 去除包裹引号
        os.environ[key] = val  # docstring: 写入环境变量


def _ensure_env_loaded() -> None:
    """
    [职责] 确保 .env 已加载到进程环境变量。
    [边界] 仅加载一次；不覆盖已有 env。
    [上游关系] _milvus_env 调用。
    [下游关系] gate tests 读取 Milvus 连接参数。
    """
    global _ENV_BOOTSTRAPPED
    if _ENV_BOOTSTRAPPED:
        return  # docstring: 避免重复加载
    _load_dotenv(_REPO_ROOT / ".env")  # docstring: 加载项目根目录 .env
    _ENV_BOOTSTRAPPED = True  # docstring: 标记已加载


def _milvus_env() -> Dict[str, str]:
    """
    [职责] 读取 Milvus 连接环境变量。
    [边界] 仅解析 env；不校验连通性。
    [上游关系] gate test 调用。
    [下游关系] _should_skip 判定。
    """
    _ensure_env_loaded()  # docstring: 先加载 .env
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


def _pdf_path() -> Path:
    """
    [职责] 返回测试用 PDF 路径。
    [边界] 不校验文件存在性。
    [上游关系] gate test 调用。
    [下游关系] ingest pipeline 使用该路径。
    """
    return (
        _REPO_ROOT
        / "src/uae_law_rag/raw_data/Cabinet Resolution No. (109) of 2023 Regulating the Real Beneficiary Procedures.pdf"
    )


async def _setup_milvus_collection(
    *,
    collection_name: str,
    embed_dim: int,
) -> Tuple[MilvusClient, MilvusIndexManager, MilvusRepo, Any]:
    """
    [职责] 创建并加载 Milvus collection（供 retrieval 使用）。
    [边界] 不处理异常恢复；调用方负责清理。
    [上游关系] gate test 调用。
    [下游关系] run_ingest_pdf / chat_service 使用。
    """
    client = MilvusClient.from_env(force_reconnect=True)  # docstring: Milvus 连接
    await client.healthcheck()  # docstring: 必须可用
    spec = build_collection_spec(
        name=collection_name,
        embed_dim=int(embed_dim),
        metric_type="COSINE",
        index_type="HNSW",
        default_top_k=5,
    )  # docstring: collection 契约
    await client.create_collection(spec, drop_if_exists=True)  # docstring: 创建 collection
    idx = MilvusIndexManager(client)  # docstring: index 管理器
    await idx.ensure_index(spec)  # docstring: 建索引
    await idx.load_collection(spec.name)  # docstring: load collection
    repo = MilvusRepo(client)  # docstring: Milvus repo 封装
    return client, idx, repo, spec


def _patch_generation_run() -> Any:
    """
    [职责] 替换 generation.run_generation 以输出确定性 JSON（避免依赖外部 LLM 实现差异）。
    [边界] 仅用于 gate 测试；必须在测试结束恢复原函数。
    [上游关系] test_chat_service_gate 调用。
    [下游关系] generation pipeline 使用该 mock 输出。
    """
    original = generator_mod.run_generation  # docstring: 保存原始 run_generation

    async def _mock_run_generation(
        *,
        messages_snapshot: Mapping[str, Any],
        model_provider: str,
        model_name: str,
        generation_config: Any = None,
    ) -> Dict[str, Any]:
        """
        [职责] 生成 deterministic raw_text（引用 evidence 构造 citations）。
        [边界] 不访问网络；不解析 JSON；仅返回 raw_text。
        [上游关系] generation pipeline 调用。
        [下游关系] postprocess 解析 raw_text 并生成 citations。
        """
        response_text = generator_mod._build_mock_response(messages_snapshot or {})  # docstring: 构造 mock JSON
        return {
            "raw_text": response_text,
            "provider": str(model_provider or "mock"),
            "model": str(model_name or "mock"),
            "usage": None,
        }  # docstring: mock generation 输出

    generator_mod.run_generation = _mock_run_generation  # docstring: 安装 mock run_generation
    return original  # docstring: 返回原函数用于恢复


def _assert_debug_blocked(debug_payload: Dict[str, Any]) -> None:
    """
    [职责] 断言 blocked 路径的 debug 合同。
    [边界] 仅检查关键字段存在性与 gate 语义；不验证内容细节。
    [上游关系] test_chat_service_gate 调用。
    [下游关系] pytest 断言 gate 合规。
    """
    assert debug_payload.get("retrieval_record_id")  # docstring: blocked 仍需 retrieval_record_id
    assert not debug_payload.get("generation_record_id")  # docstring: blocked 不允许 generation_record_id
    assert not debug_payload.get("evaluation_record_id")  # docstring: blocked 不允许 evaluation_record_id
    gate = debug_payload.get("gate") or {}  # docstring: gate 摘要
    retrieval_gate = gate.get("retrieval") or {}  # docstring: retrieval gate 摘要
    assert retrieval_gate.get("passed") is False  # docstring: retrieval gate 必须失败
    provider_snapshot = debug_payload.get("provider_snapshot") or {}  # docstring: provider_snapshot 摘要
    assert isinstance(provider_snapshot, dict)  # docstring: provider_snapshot 必须为 dict
    assert "embed" in provider_snapshot  # docstring: blocked 仍需 embed 快照
    timing_ms = debug_payload.get("timing_ms") or {}  # docstring: timing_ms 摘要
    assert isinstance(timing_ms, dict)  # docstring: timing_ms 必须为 dict
    assert "retrieval" in timing_ms  # docstring: blocked 至少包含 retrieval timing


def _assert_debug_full(debug_payload: Dict[str, Any]) -> None:
    """
    [职责] 断言非 blocked 路径的 debug 合同。
    [边界] 仅检查 record_id/gate/provider_snapshot 基本结构；不验证内容正确性。
    [上游关系] test_chat_service_gate 调用。
    [下游关系] pytest 断言 gate 合规。
    """
    assert debug_payload.get("retrieval_record_id")  # docstring: retrieval_record_id 必须存在
    assert debug_payload.get("generation_record_id")  # docstring: generation_record_id 必须存在
    assert debug_payload.get("evaluation_record_id")  # docstring: evaluation_record_id 必须存在
    gate = debug_payload.get("gate") or {}  # docstring: gate 摘要
    assert "retrieval" in gate  # docstring: retrieval gate 必须存在
    assert "generation" in gate  # docstring: generation gate 必须存在
    assert "evaluator" in gate  # docstring: evaluator gate 必须存在
    provider_snapshot = debug_payload.get("provider_snapshot") or {}  # docstring: provider_snapshot 摘要
    assert isinstance(provider_snapshot, dict)  # docstring: provider_snapshot 必须为 dict
    assert "embed" in provider_snapshot  # docstring: embed 快照必须存在
    assert "llm" in provider_snapshot  # docstring: llm 快照必须存在
    timing_ms = debug_payload.get("timing_ms") or {}  # docstring: timing_ms 摘要
    assert isinstance(timing_ms, dict)  # docstring: timing_ms 必须为 dict
    assert "retrieval" in timing_ms  # docstring: full chain 必须包含 retrieval timing
    assert "generation" in timing_ms  # docstring: full chain 必须包含 generation timing
    assert "evaluator" in timing_ms  # docstring: full chain 必须包含 evaluator timing


@pytest.mark.asyncio
async def test_chat_service_gate(session: AsyncSession) -> None:
    """
    [职责] 验证 chat_service Phase1/2/3 的最小可信闭环与裁决合同。
    [边界] 不验证 HTTP 层；不验证 prompt 细节；仅验证状态机与审计锚点。
    [上游关系] 依赖 SQLite FTS + Milvus + ingest pipeline。
    [下游关系] 确保 Gate 裁决与 message.status 映射一致。
    """
    if _should_skip():
        pytest.skip(
            "Milvus env not configured. Set MILVUS_URI/MILVUS_URL or MILVUS_HOST/MILVUS_PORT."
        )  # docstring: 环境不可用直接跳过
    _ensure_mock_llm_available()  # docstring: generation/evaluator 依赖检查

    pdf_file = _pdf_path()  # docstring: PDF fixture 路径
    if not pdf_file.exists():
        pytest.skip("PDF fixture not found for chat service gate.")  # docstring: 缺少测试文件直接跳过

    await ensure_sqlite_fts(session)  # docstring: 确保 FTS 触发器可用

    user_repo = UserRepo(session)  # docstring: 用户仓储
    ingest_repo = IngestRepo(session)  # docstring: ingest 仓储
    msg_repo = MessageRepo(session)  # docstring: message 仓储
    retrieval_repo = RetrievalRepo(session)  # docstring: retrieval 仓储
    generation_repo = GenerationRepo(session)  # docstring: generation 仓储
    evaluator_repo = EvaluatorRepo(session)  # docstring: evaluator 仓储

    uniq = time.time_ns()
    user = await user_repo.create(username=f"chat_service_gate_{uniq}")  # docstring: 创建测试用户
    collection_name = f"chat_service_gate_{uniq}"  # docstring: 唯一 collection 名称
    kb = await ingest_repo.create_kb(
        user_id=user.id,
        kb_name=f"chat_service_kb_{uniq}",
        milvus_collection=collection_name,
        embed_model="hash",
        embed_dim=32,
        embed_provider="hash",
        chunking_config={"enable_sentence_window": True, "window_size": 2},
    )  # docstring: 创建 KB 配置

    client = None
    idx = None
    spec = None
    original_run_generation = None  # docstring: 保存原始 generation.run_generation
    try:
        client, idx, repo, spec = await _setup_milvus_collection(
            collection_name=collection_name,
            embed_dim=kb.embed_dim,
        )  # docstring: 初始化 Milvus

        ingest_result = await run_ingest_pdf(
            session=session,
            kb_id=kb.id,
            pdf_path=str(pdf_file),
            milvus_repo=repo,
            milvus_collection=collection_name,
        )  # docstring: 导入 PDF 以生成检索证据
        assert ingest_result.node_count > 0  # docstring: ingest 必须产出节点
        await session.commit()  # docstring: 提交导入结果

        original_run_generation = _patch_generation_run()  # docstring: 注入 mock generation

        base_context = {
            "embed_provider": "hash",
            "embed_model": "hash",
            "embed_dim": 32,
            "model_provider": "mock",
            "model_name": "mock",
            "keyword_top_k": 50,
            "vector_top_k": 5,
        }  # docstring: 通用上下文配置

        response_blocked = await chat(
            session=session,
            milvus_repo=repo,
            query="__no_hit_query__",
            conversation_id=None,
            user_id=user.id,
            kb_id=kb.id,
            chat_type="chat",
            context={**base_context, "vector_top_k": 0},
            trace_context=TraceContext(),
            debug=True,
        )  # docstring: Case A BLOCKED

        assert response_blocked.get("status") == "blocked"  # docstring: blocked 状态
        assert response_blocked.get("answer") == ""  # docstring: blocked answer 为空
        assert response_blocked.get("citations") == []  # docstring: blocked citations 为空
        assert response_blocked.get(TRACE_ID_KEY)  # docstring: trace_id 必须存在
        timing = response_blocked.get(TIMING_MS_KEY, {})
        assert TIMING_TOTAL_MS_KEY in timing  # docstring: total_ms 必须存在

        blocked_msg = await msg_repo.get_by_id(response_blocked["message_id"])  # docstring: 回查 message
        assert blocked_msg is not None
        assert blocked_msg.status == "blocked"  # docstring: message.status blocked

        blocked_debug = response_blocked.get(DEBUG_KEY) or {}
        _assert_debug_blocked(blocked_debug)  # docstring: blocked debug 断言

        blocked_record_id = blocked_debug.get("retrieval_record_id")
        assert isinstance(blocked_record_id, str) and blocked_record_id  # docstring: retrieval_record_id 必须为 str
        retrieval_record = await retrieval_repo.get_record(blocked_record_id)  # docstring: 回查 retrieval record
        assert retrieval_record is not None
        blocked_hits = await retrieval_repo.list_hits(retrieval_record.id)  # docstring: blocked hits
        assert len(blocked_hits) == 0  # docstring: blocked 不允许 hits

        blocked_gen = await generation_repo.get_record_by_message(
            response_blocked["message_id"]
        )  # docstring: generation 记录检查
        assert blocked_gen is None  # docstring: blocked 不创建 generation record
        blocked_eval = await evaluator_repo.get_by_message_id(
            response_blocked["message_id"]
        )  # docstring: evaluation 记录检查
        assert blocked_eval is None  # docstring: blocked 不创建 evaluation record

        trace_context = TraceContext()  # docstring: trace 上下文
        response_success = await chat(
            session=session,
            milvus_repo=repo,
            query="Article",
            conversation_id=None,
            user_id=user.id,
            kb_id=kb.id,
            chat_type="chat",
            context=base_context,
            trace_context=trace_context,
            debug=True,
        )  # docstring: Case B SUCCESS

        assert response_success.get("status") == "success"  # docstring: success 状态
        assert response_success.get("answer")  # docstring: success answer 非空
        assert len(response_success.get("citations") or []) >= 1  # docstring: success citations >= 1
        assert response_success.get(TRACE_ID_KEY) == str(trace_context.trace_id)  # docstring: trace_id 透传

        msg_row = await msg_repo.get_by_id(response_success["message_id"])  # docstring: 回查 message
        assert msg_row is not None
        assert msg_row.status == "success"  # docstring: message.status success

        debug_payload = response_success.get(DEBUG_KEY) or {}
        _assert_debug_full(debug_payload)  # docstring: success debug 断言
        gate = debug_payload.get("gate") or {}  # docstring: gate 摘要
        assert gate.get("retrieval", {}).get("passed") is True  # docstring: retrieval gate 必须通过
        assert gate.get("evaluator", {}).get("status") == "pass"  # docstring: evaluator gate pass

        retrieval_record_id = debug_payload.get("retrieval_record_id")
        assert isinstance(retrieval_record_id, str) and retrieval_record_id  # docstring: retrieval_record_id 必须为 str
        retrieval_record = await retrieval_repo.get_record(retrieval_record_id)  # docstring: 回查 retrieval record
        assert retrieval_record is not None
        hits = await retrieval_repo.list_hits(retrieval_record.id)  # docstring: 回查 hits
        assert len(hits) > 0  # docstring: success 必须有命中

        gen_record = await generation_repo.get_record_by_message(
            response_success["message_id"]
        )  # docstring: 回查 generation record
        assert gen_record is not None
        assert gen_record.status in {"success", "partial", "failed"}  # docstring: generation 状态合法

        eval_record = await evaluator_repo.get_by_message_id(
            response_success["message_id"]
        )  # docstring: 回查 evaluation record
        assert eval_record is not None
        assert eval_record.status == "pass"  # docstring: evaluator pass
        evaluator_summary = response_success.get("evaluator") or {}
        assert evaluator_summary.get("status") == "pass"  # docstring: evaluator summary pass

        response_partial = await chat(
            session=session,
            milvus_repo=repo,
            query="Article",
            conversation_id=response_success["conversation_id"],
            user_id=user.id,
            kb_id=kb.id,
            chat_type="chat",
            context={**base_context, "evaluator_config": {"min_answer_chars": 200}},
            trace_context=TraceContext(),
            debug=True,
        )  # docstring: Case C PARTIAL

        assert response_partial.get("status") == "partial"  # docstring: partial 状态
        assert response_partial.get("answer")  # docstring: partial answer 允许非空
        assert len(response_partial.get("citations") or []) >= 1  # docstring: partial citations 允许存在
        evaluator_summary = response_partial.get("evaluator") or {}
        assert evaluator_summary.get("status") == "partial"  # docstring: evaluator partial
        assert len(evaluator_summary.get("warnings") or []) >= 1  # docstring: partial warnings 必须存在
        partial_debug = response_partial.get(DEBUG_KEY) or {}
        _assert_debug_full(partial_debug)  # docstring: partial debug 断言
        assert (
            partial_debug.get("gate", {}).get("evaluator", {}).get("status") == "partial"
        )  # docstring: evaluator gate partial

        partial_msg = await msg_repo.get_by_id(response_partial["message_id"])  # docstring: 回查 partial message
        assert partial_msg is not None
        assert partial_msg.status == "partial"  # docstring: message.status partial

        partial_eval = await evaluator_repo.get_by_message_id(
            response_partial["message_id"]
        )  # docstring: 回查 partial evaluation
        assert partial_eval is not None
        assert partial_eval.status == "partial"  # docstring: evaluation.status partial

        response_failed = await chat(
            session=session,
            milvus_repo=repo,
            query="Article",
            conversation_id=response_success["conversation_id"],
            user_id=user.id,
            kb_id=kb.id,
            chat_type="chat",
            context={**base_context, "evaluator_config": {"min_citations": 3}},
            trace_context=TraceContext(),
            debug=True,
        )  # docstring: Case D FAILED

        assert response_failed.get("status") == "failed"  # docstring: failed 状态
        assert response_failed.get("answer") == ""  # docstring: failed answer 必须为空
        assert response_failed.get("citations") == []  # docstring: failed citations 必须为空
        failed_debug = response_failed.get(DEBUG_KEY) or {}
        _assert_debug_full(failed_debug)  # docstring: failed debug 断言
        assert failed_debug.get("gate", {}).get("evaluator", {}).get("status") in {
            "fail",
            "skipped",
        }  # docstring: evaluator gate fail/skipped

        failed_msg = await msg_repo.get_by_id(response_failed["message_id"])  # docstring: 回查 failed message
        assert failed_msg is not None
        assert failed_msg.status == "failed"  # docstring: message.status failed
        assert failed_msg.error_message  # docstring: failed 必须有 error_message

        failed_eval = await evaluator_repo.get_by_message_id(
            response_failed["message_id"]
        )  # docstring: 回查 failed evaluation
        assert failed_eval is not None
        assert failed_eval.status in {"fail", "skipped"}  # docstring: evaluation.status fail/skipped

    finally:
        if original_run_generation is not None:
            generator_mod.run_generation = original_run_generation  # docstring: 恢复 run_generation
        if idx is not None and spec is not None:
            try:
                await idx.release_collection(spec.name)  # docstring: 释放 collection 资源
            except Exception:
                pass
        if client is not None and spec is not None:
            try:
                await client.drop_collection(spec.name)  # docstring: 清理测试 collection
            except Exception:
                pass
            client.disconnect()  # docstring: 释放 Milvus alias
