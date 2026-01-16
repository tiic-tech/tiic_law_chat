# playground/ingest_gate/test_ingest_gate.py

"""
[职责] ingest gate：验证 PDF→Markdown→Node→Embedding→Milvus→DB 的最小可信闭环。
[边界] 仅验证 ingest 的端到端行为；不覆盖 retrieval/generation。
[上游关系] 依赖 ingest pipeline、Milvus 环境、SQLite FTS 触发器。
[下游关系] 保障后续 retrieval/generation 的证据链与可回放性。
"""

from __future__ import annotations

import os
import sys
import time
import json
from pathlib import Path
from typing import Dict
from collections import Counter

import pytest
from sqlalchemy.ext.asyncio import AsyncSession

_REPO_ROOT = Path(__file__).resolve().parents[2]  # docstring: 定位项目根目录
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))  # docstring: 优先使用本地源码而非 site-packages

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
from uae_law_rag.backend.pipelines.ingest.pipeline import run_ingest_pdf


pytestmark = pytest.mark.ingest_gate


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


def _debug_snapshot(*, f, doc, nodes, result) -> str:
    pages = [n.page for n in nodes]
    c = Counter("null" if p is None else ("zero" if int(p) == 0 else "pos") for p in pages)

    # sample nodes for inspection (stable small sample)
    sample = []
    for n in sorted(nodes, key=lambda x: int(x.node_index))[:10]:
        sample.append(
            {
                "node_id": str(n.id),
                "node_index": int(n.node_index),
                "page": n.page,
                "start_offset": getattr(n, "start_offset", None),
                "end_offset": getattr(n, "end_offset", None),
                "article_id": getattr(n, "article_id", None),
                "section_path": getattr(n, "section_path", None),
                "text_head": (n.text or "")[:80],
            }
        )

    snap = {
        "file": {"id": str(f.id), "pages": f.pages, "status": f.ingest_status},
        "document": {"id": str(doc.id)},
        "result": {
            "node_count": result.node_count,
            "vector_count": getattr(result, "vector_count", None),
        },
        "node_page_distribution": dict(c),
        "node_sample": sample,
    }
    return json.dumps(snap, ensure_ascii=False, indent=2)


def _ensure_env_loaded() -> None:
    """
    [职责] 确保 .env 已加载到进程环境变量。
    [边界] 仅加载一次；不覆盖已有 env。
    [上游关系] _milvus_env 调用。
    [下游关系] 让 gate tests 能读取 MILVUS_*。
    """
    global _ENV_BOOTSTRAPPED
    if _ENV_BOOTSTRAPPED:
        return  # docstring: 避免重复加载
    _load_dotenv(_REPO_ROOT / ".env")  # docstring: 加载项目根目录 .env
    _ENV_BOOTSTRAPPED = True  # docstring: 标记已加载


def _milvus_env() -> Dict[str, str]:
    """
    [职责] 读取 Milvus 连接环境变量。
    [边界] 仅检查变量是否存在；不验证连通性。
    [上游关系] gate test 调用。
    [下游关系] _should_skip 使用。
    """
    _ensure_env_loaded()  # docstring: 先加载 .env
    uri = os.getenv("MILVUS_URI", "").strip()
    url = os.getenv("MILVUS_URL", "").strip()  # docstring: 兼容 infra/compose 常用变量
    if not uri and url:
        uri = url  # docstring: 将 URL 视为 URI
        os.environ["MILVUS_URI"] = uri  # docstring: 兼容 MilvusClient.from_env
    return {
        "uri": uri,
        "host": os.getenv("MILVUS_HOST", "").strip(),
        "port": os.getenv("MILVUS_PORT", "").strip(),
    }


def _should_skip() -> bool:
    """
    [职责] 判断 Milvus 环境是否可用。
    [边界] 仅基于环境变量判断。
    [上游关系] gate test 调用。
    [下游关系] pytest.skip 触发。
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
    [下游关系] run_ingest_pdf 使用该路径。
    """
    return Path(
        "src/uae_law_rag/raw_data/Cabinet Resolution No. (109) of 2023 Regulating the Real Beneficiary Procedures.pdf"
    )


@pytest.mark.asyncio
async def test_ingest_gate_end_to_end(session: AsyncSession) -> None:
    """
    [职责] 验证 ingest pipeline 的最小可信闭环（SQL + Milvus）。
    [边界] 仅验证 gate 断言，不做性能评估。
    [上游关系] 依赖 SQLite FTS、Milvus 环境、PDF 解析依赖。
    [下游关系] 保障 retrieval/generation 使用的证据一致性。
    """
    if _should_skip():
        pytest.skip(
            "Milvus env not configured. Set MILVUS_URI/MILVUS_URL or MILVUS_HOST/MILVUS_PORT."
        )  # docstring: 环境不可用

    pdf_file = _pdf_path()  # docstring: PDF fixture 路径
    if not pdf_file.exists():
        pytest.skip("PDF fixture not found for ingest gate.")  # docstring: 缺少测试文件直接跳过

    await ensure_sqlite_fts(session)  # docstring: 确保 FTS 触发器可用

    user_repo = UserRepo(session)  # docstring: 用户仓储
    ingest_repo = IngestRepo(session)  # docstring: ingest 仓储

    uniq = time.time_ns()
    u = await user_repo.create(username=f"ingest_gate_{uniq}")  # docstring: 创建测试用户
    collection_name = f"ingest_gate_{uniq}"  # docstring: 唯一 collection 名称

    kb = await ingest_repo.create_kb(
        user_id=u.id,
        kb_name=f"ingest_gate_kb_{uniq}",
        milvus_collection=collection_name,
        embed_model="hash",
        embed_dim=32,
        embed_provider="hash",
        chunking_config={"enable_sentence_window": True, "window_size": 2},
    )  # docstring: 创建 KB 配置

    client = MilvusClient.from_env(force_reconnect=True)  # docstring: Milvus 连接
    await client.healthcheck()  # docstring: Milvus 必须可用
    spec = build_collection_spec(
        name=collection_name,
        embed_dim=kb.embed_dim,
        metric_type="COSINE",
        index_type="HNSW",
        default_top_k=5,
    )  # docstring: collection 契约

    await client.create_collection(spec, drop_if_exists=True)  # docstring: 创建 collection
    idx = MilvusIndexManager(client)
    await idx.ensure_index(spec)  # docstring: 建索引
    await idx.load_collection(spec.name)  # docstring: load collection
    repo = MilvusRepo(client)  # docstring: Milvus repo 封装

    try:
        result = await run_ingest_pdf(
            session=session,
            kb_id=kb.id,
            pdf_path=str(pdf_file),
            milvus_repo=repo,
            milvus_collection=collection_name,
        )  # docstring: 执行 ingest pipeline
        await session.commit()  # docstring: 提交以确保 FTS 可查询

        f = await ingest_repo.get_file(result.file_id)  # docstring: 读取文件记录
        assert f is not None
        assert f.ingest_status == "success"
        assert f.pages is not None and f.pages > 0

        doc = await ingest_repo.get_document(result.document_id)  # docstring: 读取文档记录
        assert doc is not None

        nodes = await ingest_repo.list_nodes_by_document(doc.id)  # docstring: 读取节点列表
        assert len(nodes) > 0
        assert len(nodes) == result.node_count
        nodes_sorted = sorted(list(nodes), key=lambda n: int(n.node_index))
        assert [n.node_index for n in nodes_sorted] == list(range(len(nodes_sorted)))

        # --- after nodes loaded and basic checks done ---
        dbg = _debug_snapshot(f=f, doc=doc, nodes=nodes, result=result)

        # Hard signal: multi-page file but all node.page are NULL
        if (f.pages or 0) > 1:
            page_null = sum(1 for n in nodes if n.page is None)
            if page_null == len(nodes):
                pytest.fail(
                    "All node.page are NULL while file.pages > 1. "
                    "This strongly suggests page marks were not extracted OR persist_nodes did not write page.\n"
                    f"DEBUG:\n{dbg}"
                )

        if (f.pages or 0) > 1:
            any_page = any(n.page is not None for n in nodes)
            if not any_page:
                pytest.fail("Expected at least 1 node with non-null page for multi-page PDF.\n" + dbg)

        article_nodes = 0
        section_nodes = 0
        non_trivial_nodes = 0
        for n in nodes:
            if n.text and len(n.text.strip()) > 10:
                non_trivial_nodes += 1  # docstring: 统计有效文本节点
            if n.page is not None:
                assert n.page > 0  # docstring: 若有页码，必须为正
            if n.article_id and str(n.article_id).strip():
                article_nodes += 1  # docstring: 统计含 article_id 的节点
            if n.section_path and str(n.section_path).strip():
                section_nodes += 1  # docstring: 统计含 section_path 的节点

        assert non_trivial_nodes > 0  # docstring: 至少存在有效文本节点
        if article_nodes == 0:
            article_text_nodes = sum(1 for n in nodes if "Article" in (n.text or ""))  # docstring: 退化检查
            assert article_text_nodes > 0  # docstring: 至少有 Article 文本线索
        else:
            assert article_nodes > 0  # docstring: 至少存在 Article 级节点
        assert section_nodes > 0  # docstring: 至少存在 Section 级节点

        hits = await search_nodes(session, kb_id=kb.id, query="Article", top_k=5, file_id=f.id)  # docstring: FTS 检索
        assert len(hits) > 0
        node_ids = {n.id for n in nodes}
        assert any(h.node_id in node_ids for h in hits)

        maps = await ingest_repo.list_node_vector_maps_by_file(f.id)  # docstring: 读取映射表
        assert len(maps) == len(nodes)
        assert {m.node_id for m in maps} == node_ids

        assert result.sample_query_vector is not None
        expr = build_expr_for_scope(kb_id=kb.id, file_id=f.id)  # docstring: Milvus 过滤表达式
        res = await repo.search(
            collection=spec.name,
            query_vectors=[result.sample_query_vector],
            top_k=5,
            expr=expr,
            output_fields=spec.search.output_fields,
        )  # docstring: Milvus 向量检索
        assert len(res) == 1
        assert len(res[0]) > 0
        payload = res[0][0]["payload"]
        assert payload.get(KB_ID_FIELD) == kb.id
        assert payload.get(NODE_ID_FIELD) in node_ids
        node_lookup = {n.id: n for n in nodes}
        if payload.get(NODE_ID_FIELD) in node_lookup:
            node_page = node_lookup[payload[NODE_ID_FIELD]].page  # docstring: SQL 节点页码
            if node_page is not None:
                assert payload.get(PAGE_FIELD) == node_page  # docstring: page 一致性
            else:
                assert payload.get(PAGE_FIELD) in (None, 0)  # docstring: 无页码时允许 Milvus 0

        # idempotent rerun
        result2 = await run_ingest_pdf(
            session=session,
            kb_id=kb.id,
            pdf_path=str(pdf_file),
            milvus_repo=repo,
            milvus_collection=collection_name,
        )  # docstring: 幂等重跑
        assert result2.file_id == result.file_id
        assert result2.node_count == result.node_count
        assert result2.vector_count == result.vector_count
        maps2 = await ingest_repo.list_node_vector_maps_by_file(f.id)
        assert len(maps2) == len(maps)
    finally:
        try:
            await idx.release_collection(spec.name)  # docstring: 释放 collection 资源
        except Exception:
            pass
        try:
            await client.drop_collection(spec.name)  # docstring: 清理测试 collection
        except Exception:
            pass
        client.disconnect()  # docstring: 释放 Milvus alias
