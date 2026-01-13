# src/uae_law_rag/backend/scripts/init_milvus.py

"""
[职责] 初始化 Milvus collection（建表/索引/加载），提供可复现、可幂等的 CLI 入口。
[边界] 不执行 ingest；不访问 DB；仅负责向量库的结构初始化与健康检查。
[上游关系] 本地开发/CI/部署脚本调用；依赖 kb/client 与 kb/schema 的契约。
[下游关系] ingest/retrieval pipeline 可直接使用已初始化的 collection。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from typing import Any, Dict, Optional, Sequence

from uae_law_rag.backend.kb.client import MilvusClient
from uae_law_rag.backend.kb.index import MilvusIndexManager
from uae_law_rag.backend.kb.schema import IndexType, MetricType, build_collection_spec


def _build_parser() -> argparse.ArgumentParser:
    """
    [职责] 构建 CLI 参数解析器。
    [边界] 仅定义参数；不解析 argv。
    [上游关系] main 调用。
    [下游关系] _parse_args 使用返回的 parser。
    """
    parser = argparse.ArgumentParser(description="Initialize Milvus collection schema and index.")
    parser.add_argument("--collection", required=True)  # docstring: collection 名称（必填）
    parser.add_argument("--embed-dim", type=int, default=1024)  # docstring: 向量维度（默认 1024）
    parser.add_argument("--metric-type", default="COSINE", choices=["IP", "L2", "COSINE"])  # docstring: 距离度量
    parser.add_argument(
        "--index-type",
        default="HNSW",
        choices=["HNSW", "IVF_FLAT", "IVF_SQ8", "AUTOINDEX"],
    )  # docstring: 索引类型
    parser.add_argument("--top-k", type=int, default=50)  # docstring: 默认 search top_k
    parser.add_argument("--description", default="UAE Law RAG KB collection")  # docstring: collection 描述
    parser.add_argument("--drop", action="store_true")  # docstring: 先 drop 再 create
    parser.add_argument("--skip-existing", action="store_true")  # docstring: 已存在时直接跳过
    parser.add_argument("--skip-load", action="store_true")  # docstring: 跳过 load collection
    parser.add_argument("--json", action="store_true")  # docstring: 仅输出 JSON 结果
    return parser


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """
    [职责] 解析 CLI 参数。
    [边界] 仅解析参数；不做业务校验。
    [上游关系] main 传入 argv。
    [下游关系] _run_async 使用参数执行动作。
    """
    parser = _build_parser()  # docstring: 构造 parser
    return parser.parse_args(list(argv) if argv is not None else None)


def _normalize_int(value: int, *, name: str, minimum: int = 1) -> int:
    """
    [职责] 归一化并校验整数参数。
    [边界] 非法值直接抛错；不做容错替换。
    [上游关系] _run_async 调用。
    [下游关系] collection spec 构建使用。
    """
    val = int(value)
    if val < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return val


async def _run_async(
    *,
    collection: str,
    embed_dim: int,
    metric_type: MetricType,
    index_type: IndexType,
    top_k: int,
    description: str,
    drop: bool,
    skip_existing: bool,
    skip_load: bool,
) -> Dict[str, Any]:
    """
    [职责] 执行 Milvus 初始化主流程（healthcheck / create / index / load）。
    [边界] 不吞异常；由上层统一捕获并输出摘要。
    [上游关系] main 解析参数后调用。
    [下游关系] _print_summary 输出结果。
    """
    start_ms = time.perf_counter() * 1000.0  # docstring: 记录开始时间
    client = MilvusClient.from_env()  # docstring: 从环境变量建立连接
    index_manager = MilvusIndexManager(client)  # docstring: 索引管理器
    name = str(collection or "").strip()  # docstring: collection 名称归一化
    if not name:
        raise ValueError("collection is required")  # docstring: collection 必填

    result: Dict[str, Any] = {
        "ok": True,
        "collection": name,
        "embed_dim": int(embed_dim),
        "metric_type": str(metric_type),
        "index_type": str(index_type),
        "top_k": int(top_k),
        "description": str(description or ""),
        "existed": False,
        "dropped": False,
        "created": False,
        "index_ensured": False,
        "loaded": False,
        "skipped": False,
        "duration_ms": 0.0,
        "error": None,
    }

    try:
        await client.healthcheck()  # docstring: Milvus 健康检查
        existed = await client.has_collection(name)  # docstring: collection 是否存在
        result["existed"] = bool(existed)  # docstring: 记录存在状态
        if existed and skip_existing and not drop:
            result["skipped"] = True  # docstring: 标记跳过
            return result

        if drop and existed:
            await client.drop_collection(name)  # docstring: drop 已存在 collection
            result["dropped"] = True  # docstring: 标记 drop 完成

        spec = build_collection_spec(
            name=name,
            embed_dim=_normalize_int(embed_dim, name="embed_dim"),
            metric_type=metric_type,
            index_type=index_type,
            default_top_k=_normalize_int(top_k, name="top_k"),
            description=description,
        )  # docstring: 构造 collection spec
        await client.create_collection(spec, drop_if_exists=False)  # docstring: 创建 collection（幂等）
        result["created"] = (not existed) or drop  # docstring: 标记是否真正创建/重建

        await index_manager.ensure_index(spec)  # docstring: 确保索引存在
        result["index_ensured"] = True  # docstring: 标记索引完成

        if not skip_load:
            await index_manager.load_collection(spec.name)  # docstring: load 以便检索
            result["loaded"] = True  # docstring: 标记 load 完成
    except Exception as exc:
        result["ok"] = False  # docstring: 标记失败
        result["error"] = f"{exc.__class__.__name__}: {exc}"  # docstring: 记录异常信息
    finally:
        result["duration_ms"] = round(time.perf_counter() * 1000.0 - start_ms, 2)  # docstring: 计算耗时
    return result


def _print_summary(*, result: Dict[str, Any], as_json: bool) -> None:
    """
    [职责] 输出执行摘要或 JSON 结果。
    [边界] 仅打印信息；不修改 result。
    [上游关系] main 调用。
    [下游关系] CLI 用户/CI 读取输出。
    """
    if as_json:
        print(json.dumps(result, ensure_ascii=True, default=str))  # docstring: JSON 输出
        return
    status = "ok" if result.get("ok") else "failed"  # docstring: 摘要状态
    print(f"[init_milvus] status={status}")  # docstring: 摘要行
    print(
        f"[init_milvus] collection={result.get('collection')} embed_dim={result.get('embed_dim')} "
        f"metric_type={result.get('metric_type')} index_type={result.get('index_type')} top_k={result.get('top_k')}"
    )  # docstring: 核心配置摘要
    print(
        f"[init_milvus] existed={result.get('existed')} dropped={result.get('dropped')} "
        f"created={result.get('created')} index_ensured={result.get('index_ensured')} "
        f"loaded={result.get('loaded')} skipped={result.get('skipped')}"
    )  # docstring: 执行动作摘要
    if result.get("error"):
        print(f"[init_milvus] error={result.get('error')}")  # docstring: 错误摘要
    print(f"[init_milvus] duration_ms={result.get('duration_ms')}")  # docstring: 耗时输出


def main(argv: Optional[Sequence[str]] = None) -> int:
    """
    [职责] CLI 入口：解析参数并执行 init_milvus 流程。
    [边界] 捕获异常并转为非 0 退出码。
    [上游关系] 命令行或脚本调用。
    [下游关系] 触发 _run_async 并输出摘要。
    """
    args = _parse_args(argv)  # docstring: 解析参数
    try:
        result = asyncio.run(
            _run_async(
                collection=args.collection,
                embed_dim=args.embed_dim,
                metric_type=args.metric_type,
                index_type=args.index_type,
                top_k=args.top_k,
                description=args.description,
                drop=bool(args.drop),
                skip_existing=bool(args.skip_existing),
                skip_load=bool(args.skip_load),
            )
        )  # docstring: 执行主流程
        _print_summary(result=result, as_json=bool(args.json))  # docstring: 输出摘要/JSON
        return 0 if result.get("ok") else 1  # docstring: 返回退出码
    except Exception as exc:
        error_result = {
            "ok": False,
            "collection": args.collection,
            "embed_dim": args.embed_dim,
            "metric_type": args.metric_type,
            "index_type": args.index_type,
            "top_k": args.top_k,
            "description": args.description,
            "existed": False,
            "dropped": bool(args.drop),
            "created": False,
            "index_ensured": False,
            "loaded": False,
            "skipped": False,
            "duration_ms": 0.0,
            "error": f"{exc.__class__.__name__}: {exc}",
        }  # docstring: 异常结果
        _print_summary(result=error_result, as_json=bool(args.json))  # docstring: 输出错误摘要/JSON
        return 1


if __name__ == "__main__":
    raise SystemExit(main())  # docstring: CLI 入口
