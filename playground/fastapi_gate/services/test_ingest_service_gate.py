# playground/fastapi_gate/services/test_ingest_service_gate.py

"""
[职责] ingest_service gate：验证服务层编排（状态机 + 两阶段提交 + Gate 裁决）的最小可信闭环。
[边界] 仅验证 ingest_service 行为；不覆盖 retrieval/generation/evaluator。
[上游关系] 依赖 ingest_service、ingest pipeline、SQLite FTS、Milvus 环境。
[下游关系] 保障后续 API/前端能获取稳定的 ingest 状态与审计锚点。
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

_REPO_ROOT = Path(__file__).resolve().parents[3]  # docstring: 定位项目根目录
_SRC_ROOT = _REPO_ROOT / "src"  # docstring: 源码目录
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))  # docstring: 优先使用本地源码而非 site-packages

from uae_law_rag.backend.db.fts import ensure_sqlite_fts, search_nodes
from uae_law_rag.backend.db.repo import IngestRepo, UserRepo
from uae_law_rag.backend.kb.client import MilvusClient
from uae_law_rag.backend.kb.index import MilvusIndexManager
from uae_law_rag.backend.kb.repo import MilvusRepo
from uae_law_rag.backend.kb.schema import (
    KB_ID_FIELD,
    NODE_ID_FIELD,
    PAGE_FIELD,
    build_collection_spec,
    build_expr_for_scope,
)
from uae_law_rag.backend.pipelines.ingest import embed as embed_mod
from uae_law_rag.backend.schemas.audit import TraceContext
from uae_law_rag.backend.services.ingest_service import ingest_file
from uae_law_rag.backend.utils.constants import (
    DEBUG_KEY,
    REQUEST_ID_KEY,
    TIMING_MS_KEY,
    TIMING_TOTAL_MS_KEY,
    TRACE_ID_KEY,
)


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
    [下游关系] ingest_service 使用该路径。
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
    [职责] 创建并加载 Milvus collection（供 ingest_service 写入）。
    [边界] 不处理异常恢复；调用方负责清理。
    [上游关系] gate test 调用。
    [下游关系] ingest_service 使用 MilvusRepo 写入向量。
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


def _pick_sample_text(nodes: List[Any]) -> str:
    """
    [职责] 选择一段可用的节点文本用于向量检索。
    [边界] 不做语义筛选；仅保证非空。
    [上游关系] gate test 调用。
    [下游关系] embed_texts 生成 query 向量。
    """
    for n in nodes:
        text = str(getattr(n, "text", "") or "")
        if len(text.strip()) > 10:
            return text  # docstring: 优先选择非空长文本
    return str(getattr(nodes[0], "text", "") or "")  # docstring: 兜底选择首条


@pytest.mark.asyncio
async def test_ingest_service_gate_end_to_end(session: AsyncSession) -> None:
    """
    [职责] 验证 ingest_service 的最小可信闭环（状态机 + Gate + 两阶段提交）。
    [边界] 不测试 HTTP 路由层；仅验证 service 行为与落库一致性。
    [上游关系] 依赖 SQLite FTS + Milvus + ingest pipeline。
    [下游关系] 为后续 API/前端提供可信 ingest 状态。
    """
    if _should_skip():
        pytest.skip(
            "Milvus env not configured. Set MILVUS_URI/MILVUS_URL or MILVUS_HOST/MILVUS_PORT."
        )  # docstring: 环境不可用直接跳过

    pdf_file = _pdf_path()  # docstring: PDF fixture 路径
    if not pdf_file.exists():
        pytest.skip("PDF fixture not found for ingest service gate.")  # docstring: 缺少测试文件直接跳过

    await ensure_sqlite_fts(session)  # docstring: 确保 FTS 触发器可用

    user_repo = UserRepo(session)  # docstring: 用户仓储
    ingest_repo = IngestRepo(session)  # docstring: ingest 仓储

    uniq = time.time_ns()
    user = await user_repo.create(username=f"ingest_service_gate_{uniq}")  # docstring: 创建测试用户
    collection_name = f"ingest_service_gate_{uniq}"  # docstring: 唯一 collection 名称
    kb = await ingest_repo.create_kb(
        user_id=user.id,
        kb_name=f"ingest_service_kb_{uniq}",
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

        trace_context = TraceContext()  # docstring: trace 上下文
        response = await ingest_file(
            session=session,
            kb_id=kb.id,
            source_uri=pdf_file.resolve().as_uri(),
            file_name=pdf_file.name,
            ingest_profile={"parser": "pymupdf4llm", "parse_version": "v1", "segment_version": "v1"},
            milvus_repo=repo,
            trace_context=trace_context,
            debug=True,
        )  # docstring: 调用 ingest_service

        assert response.get("status") == "success"  # docstring: service 成功状态
        assert response.get("file_id")  # docstring: file_id 必须存在
        assert int(response.get("node_count", 0)) > 0  # docstring: 节点数量必须为正

        timing = response.get(TIMING_MS_KEY, {})
        assert TIMING_TOTAL_MS_KEY in timing  # docstring: 必须包含 total_ms
        assert float(timing[TIMING_TOTAL_MS_KEY]) >= 0.0  # docstring: timing 值合法

        assert response.get(TRACE_ID_KEY) == str(trace_context.trace_id)  # docstring: trace_id 透传
        assert response.get(REQUEST_ID_KEY) == str(trace_context.request_id)  # docstring: request_id 透传

        debug_payload = response.get(DEBUG_KEY) or {}
        gate = debug_payload.get("gate") or {}
        assert gate.get("passed") is True  # docstring: gate 必须通过
        assert debug_payload.get("node_ids_count") == response.get("node_count")  # docstring: 节点数量一致
        assert debug_payload.get("vector_ids_count") == response.get("node_count")  # docstring: 向量数量一致

        file_row = await ingest_repo.get_file(response["file_id"])  # docstring: 读取文件记录
        assert file_row is not None  # docstring: file 记录必须存在
        assert file_row.ingest_status == "success"  # docstring: DB 状态必须为 success
        assert int(file_row.node_count) == response.get("node_count")  # docstring: DB 统计一致
        if file_row.pages is not None:
            assert file_row.pages > 0  # docstring: 若有页码，必须为正

        doc = await ingest_repo.get_document(debug_payload.get("document_id", ""))  # docstring: 读取文档记录
        assert doc is not None  # docstring: document 记录必须存在
        nodes = await ingest_repo.list_nodes_by_document(doc.id)  # docstring: 读取节点列表
        assert len(nodes) == response.get("node_count")  # docstring: node_count 对齐

        nodes_sorted = sorted(list(nodes), key=lambda n: int(n.node_index))
        assert [n.node_index for n in nodes_sorted] == list(range(len(nodes_sorted)))  # docstring: node_index 连续

        hits = await search_nodes(
            session,
            kb_id=kb.id,
            query="Article",
            top_k=5,
            file_id=file_row.id,
        )  # docstring: FTS 关键词召回
        assert len(hits) > 0  # docstring: FTS 必须有命中
        node_ids = {n.id for n in nodes}
        assert any(h.node_id in node_ids for h in hits)  # docstring: 命中必须属于本文件

        maps = await ingest_repo.list_node_vector_maps_by_file(file_row.id)  # docstring: 读取映射表
        assert len(maps) == len(nodes)  # docstring: node↔vector 数量一致

        sample_text = _pick_sample_text(nodes)  # docstring: 选取文本生成 query 向量
        vectors = await embed_mod.embed_texts(
            texts=[sample_text],
            provider=kb.embed_provider or "hash",
            model=kb.embed_model or "hash",
            dim=kb.embed_dim,
        )  # docstring: 构造 query 向量
        expr = build_expr_for_scope(kb_id=kb.id, file_id=file_row.id)  # docstring: Milvus 过滤表达式
        res = await repo.search(
            collection=spec.name,
            query_vectors=vectors,
            top_k=5,
            expr=expr,
            output_fields=spec.search.output_fields,
        )  # docstring: Milvus 向量检索
        assert len(res) == 1  # docstring: 单 query 向量
        assert len(res[0]) > 0  # docstring: Milvus 必须有命中
        payload = res[0][0]["payload"]
        assert payload.get(KB_ID_FIELD) == kb.id  # docstring: KB 作用域一致
        assert payload.get(NODE_ID_FIELD) in node_ids  # docstring: node_id 必须可回查
        node_lookup = {n.id: n for n in nodes}  # docstring: node_id → Node 索引
        if payload.get(NODE_ID_FIELD) in node_lookup:
            node_page = node_lookup[payload[NODE_ID_FIELD]].page  # docstring: SQL 节点页码
            if node_page is not None:
                assert payload.get(PAGE_FIELD) == node_page  # docstring: page 一致性

        response2 = await ingest_file(
            session=session,
            kb_id=kb.id,
            source_uri=pdf_file.resolve().as_uri(),
            file_name=pdf_file.name,
            ingest_profile={"parser": "pymupdf4llm", "parse_version": "v1", "segment_version": "v1"},
            milvus_repo=repo,
            trace_context=TraceContext(),
            debug=False,
        )  # docstring: 幂等重跑
        assert response2.get("file_id") == response.get("file_id")  # docstring: 幂等 file_id
        assert response2.get("status") == "success"  # docstring: 幂等状态稳定
        assert response2.get("node_count") == response.get("node_count")  # docstring: 幂等统计一致
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
