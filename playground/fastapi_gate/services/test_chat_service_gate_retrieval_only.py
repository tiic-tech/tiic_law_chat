# playground/fastapi_gate/services/test_chat_service_gate_retrieval_only.py

"""
[职责] chat_service gate：验证 retrieval-only 阶段的状态机、检索落库与 BLOCK 裁决。
[边界] 仅覆盖 Message + Retrieval；不涉及 generation/evaluator。
[上游关系] 依赖 ingest pipeline、SQLite FTS、Milvus 环境与 chat_service。
[下游关系] 保障后续 generation/evaluator 在可信 evidence 基础上运行。
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Tuple

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

_REPO_ROOT = Path(__file__).resolve().parents[3]  # docstring: 定位项目根目录
_SRC_ROOT = _REPO_ROOT / "src"  # docstring: 源码目录
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))  # docstring: 优先使用本地源码而非 site-packages

from uae_law_rag.backend.db.fts import ensure_sqlite_fts
from uae_law_rag.backend.db.repo import IngestRepo, MessageRepo, RetrievalRepo, UserRepo
from uae_law_rag.backend.kb.client import MilvusClient
from uae_law_rag.backend.kb.index import MilvusIndexManager
from uae_law_rag.backend.kb.repo import MilvusRepo
from uae_law_rag.backend.kb.schema import build_collection_spec
from uae_law_rag.backend.pipelines.ingest.pipeline import run_ingest_pdf
from uae_law_rag.backend.schemas.audit import TraceContext
from uae_law_rag.backend.services.chat_service import chat
from uae_law_rag.backend.utils.constants import DEBUG_KEY, TIMING_MS_KEY, TIMING_TOTAL_MS_KEY, TRACE_ID_KEY


pytestmark = pytest.mark.fastapi_gate


_ENV_BOOTSTRAPPED = False  # docstring: 防止重复加载 .env


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


@pytest.mark.asyncio
async def test_chat_service_gate_retrieval_only(session: AsyncSession) -> None:
    """
    [职责] 验证 chat_service retrieval-only 阶段的最小可信闭环。
    [边界] 不验证 generation/evaluator；仅验证 Message + Retrieval。
    [上游关系] 依赖 SQLite FTS + Milvus + ingest pipeline。
    [下游关系] 确保 service gate 可阻断无证据请求。
    """
    if _should_skip():
        pytest.skip(
            "Milvus env not configured. Set MILVUS_URI/MILVUS_URL or MILVUS_HOST/MILVUS_PORT."
        )  # docstring: 环境不可用直接跳过

    pdf_file = _pdf_path()  # docstring: PDF fixture 路径
    if not pdf_file.exists():
        pytest.skip("PDF fixture not found for chat service gate.")  # docstring: 缺少测试文件直接跳过

    await ensure_sqlite_fts(session)  # docstring: 确保 FTS 触发器可用

    user_repo = UserRepo(session)  # docstring: 用户仓储
    ingest_repo = IngestRepo(session)  # docstring: ingest 仓储
    msg_repo = MessageRepo(session)  # docstring: message 仓储
    retrieval_repo = RetrievalRepo(session)  # docstring: retrieval 仓储

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

        trace_context = TraceContext()  # docstring: trace 上下文
        response = await chat(
            session=session,
            milvus_repo=repo,
            query="Article",
            conversation_id=None,
            user_id=user.id,
            kb_id=kb.id,
            chat_type="chat",
            context={"embed_provider": "hash", "embed_model": "hash", "embed_dim": 32, "vector_top_k": 5},
            trace_context=trace_context,
            debug=True,
        )  # docstring: 调用 chat_service (ready)

        assert response.get("status") == "ready"  # docstring: retrieval ready
        assert response.get("conversation_id")  # docstring: conversation_id 必须存在
        assert response.get("message_id")  # docstring: message_id 必须存在
        assert response.get("kb_id") == kb.id  # docstring: KB 作用域一致
        assert response.get(TRACE_ID_KEY) == str(trace_context.trace_id)  # docstring: trace_id 透传

        timing = response.get(TIMING_MS_KEY, {})
        assert TIMING_TOTAL_MS_KEY in timing  # docstring: total_ms 必须存在

        debug_payload = response.get(DEBUG_KEY) or {}
        retrieval_record_id = debug_payload.get("retrieval_record_id")
        assert retrieval_record_id  # docstring: retrieval_record_id 必须存在
        assert isinstance(debug_payload.get("provider_snapshot"), dict)  # docstring: provider_snapshot 必须返回
        assert isinstance(debug_payload.get("timing_ms"), dict)  # docstring: timing_ms 必须返回

        msg_row = await msg_repo.get_by_id(response["message_id"])  # docstring: 回查 message
        assert msg_row is not None
        assert msg_row.status == "ready"  # docstring: message.status 必须 ready

        record = await retrieval_repo.get_record(retrieval_record_id)  # docstring: 回查 retrieval record
        assert record is not None
        snapshot = record.provider_snapshot or {}
        embed_snapshot = snapshot.get("embed") or {}
        assert embed_snapshot.get("provider") == "hash"  # docstring: embed provider 覆盖生效
        assert embed_snapshot.get("model") == "hash"  # docstring: embed model 覆盖生效
        assert embed_snapshot.get("dim") == 32  # docstring: embed dim 覆盖生效

        hits = await retrieval_repo.list_hits(record.id)  # docstring: 回查 hits
        assert len(hits) > 0  # docstring: ready 必须有命中

        response_blocked = await chat(
            session=session,
            milvus_repo=repo,
            query="__no_hit_query__",
            conversation_id=response["conversation_id"],
            user_id=user.id,
            kb_id=kb.id,
            chat_type="chat",
            context={"embed_provider": "hash", "embed_model": "hash", "embed_dim": 32, "vector_top_k": 0},
            trace_context=TraceContext(),
            debug=True,
        )  # docstring: 调用 chat_service (blocked)

        assert response_blocked.get("status") == "blocked"  # docstring: 必须 blocked
        blocked_msg = await msg_repo.get_by_id(response_blocked["message_id"])  # docstring: 回查 blocked message
        assert blocked_msg is not None
        assert blocked_msg.status == "blocked"  # docstring: message.status 必须 blocked
        assert blocked_msg.error_message == "no_evidence"
        blocked_debug = response_blocked.get(DEBUG_KEY) or {}
        blocked_record_id = blocked_debug.get("retrieval_record_id")
        assert blocked_record_id  # docstring: blocked 也必须落 retrieval_record
        blocked_hits = await retrieval_repo.list_hits(blocked_record_id)  # docstring: blocked hits 为空
        assert len(blocked_hits) == 0
    finally:
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
