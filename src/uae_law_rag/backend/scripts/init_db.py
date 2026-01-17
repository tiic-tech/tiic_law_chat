# src/uae_law_rag/backend/scripts/init_db.py

"""
[职责] 初始化数据库结构（create_all / 可选 drop），提供可复现、可幂等的 CLI 入口。
[边界] 不执行 pipeline；仅在 --seed 显式开启时插入最小可用业务数据（dev-user + default KB）；可选初始化 FTS。
[上游关系] 本地开发/CI/部署脚本调用；依赖 db.engine 的 init_db/drop_db。
[下游关系] DB schema 准备完成后供 services/pipelines/api 使用。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
import uuid
from typing import Any, Dict, Optional, Sequence, Tuple

from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker
from sqlalchemy import select

from uae_law_rag.backend.db.engine import create_engine, drop_db, init_db
from uae_law_rag.backend.db.models.user import UserModel
from uae_law_rag.backend.db.models.doc import KnowledgeBaseModel
from uae_law_rag.backend.db.fts import ensure_sqlite_fts, rebuild_sqlite_fts


def _build_parser() -> argparse.ArgumentParser:
    """
    [职责] 构建 CLI 参数解析器。
    [边界] 仅定义参数；不解析 argv。
    [上游关系] main 调用。
    [下游关系] _parse_args 使用返回的 parser。
    """
    parser = argparse.ArgumentParser(description="Initialize database schema (create_all).")
    parser.add_argument("--db-url", dest="db_url", default=None)  # docstring: 显式 DB 连接串
    parser.add_argument("--drop", action="store_true")  # docstring: 先 drop 再 create
    parser.add_argument("--seed", action="store_true")  # docstring: 显式种子数据入口
    parser.add_argument("--seed-fts", action="store_true")  # docstring: 初始化 SQLite FTS（node_fts + triggers）
    parser.add_argument("--rebuild-fts", action="store_true")  # docstring: 重建 FTS 内容（DELETE+INSERT）
    # docstring: SQL echo 三态开关：默认 None（由 engine/环境决定）；--echo 强制 True；--no-echo 强制 False
    echo_group = parser.add_mutually_exclusive_group()
    echo_group.add_argument("--echo", dest="echo", action="store_true", default=None)
    echo_group.add_argument("--no-echo", dest="echo", action="store_false")
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


async def _ensure_dev_user_and_default_kb(*, session: AsyncSession) -> Tuple[str, str]:
    """
    [职责] 幂等确保存在 dev-user 与 default KB（id 固定为 "default"）。
    [边界] 仅写入最小可用字段；不触发 ingest/milvus 等外部依赖。
    [上游关系] _maybe_seed 调用。
    [下游关系] /api/chat /api/ingest 能直接使用 kb_id="default"。
    """
    # 1) ensure dev-user
    q_user = await session.execute(select(UserModel).where(UserModel.username == "dev-user"))
    user = q_user.scalar_one_or_none()
    if user is None:
        user = UserModel(
            id=str(uuid.uuid4()),
            username="dev-user",
            password_hash=None,
            is_active=True,
        )
        session.add(user)
        await session.flush()  # docstring: 确保拿到 user.id

    user_id = str(user.id)

    # 2) ensure default KB (IMPORTANT: id="default" to match existing curl/front usage)
    q_kb = await session.execute(select(KnowledgeBaseModel).where(KnowledgeBaseModel.id == "default"))
    kb = q_kb.scalar_one_or_none()
    if kb is None:
        kb = KnowledgeBaseModel(
            id="default",
            user_id=user_id,
            kb_name="default",
            kb_info=None,
            vs_type="milvus",
            milvus_collection="kb_default",
            milvus_partition=None,
            embed_provider="local",
            embed_model="bge-small",
            embed_dim=384,
            rerank_provider=None,
            rerank_model=None,
            chunking_config={},
            file_count=0,
        )
        session.add(kb)
        await session.flush()
    else:
        # docstring: 幂等修正关键字段（避免旧数据漂移影响联调）
        kb.user_id = user_id
        kb.kb_name = kb.kb_name or "default"
        kb.vs_type = kb.vs_type or "milvus"
        kb.milvus_collection = kb.milvus_collection or "kb_default"
        kb.embed_provider = kb.embed_provider or "local"
        kb.embed_model = kb.embed_model or "bge-small"
        kb.embed_dim = int(kb.embed_dim or 384)
        kb.chunking_config = kb.chunking_config or {}
        kb.file_count = int(kb.file_count or 0)
        await session.flush()

    return user_id, str(kb.id)


async def _maybe_seed(*, engine: AsyncEngine, enabled: bool) -> Dict[str, Any]:
    """
    [职责] 可选种子数据入口（显式 --seed 才启用）。
    [边界] 仅插入最小可用 dev 数据（dev-user + default KB）；不依赖 Milvus；幂等可重复执行。
    [上游关系] _run_async 在 init_db 之后调用。
    [下游关系] 返回 seed 结果用于 summary 输出。
    """
    if not enabled:
        return {"seeded": False, "seed_status": "skipped"}  # docstring: 未开启则跳过
    Session = async_sessionmaker(bind=engine, expire_on_commit=False, class_=AsyncSession)
    async with Session() as session:
        async with session.begin():
            user_id, kb_id = await _ensure_dev_user_and_default_kb(session=session)
    return {
        "seeded": True,
        "seed_status": "ok",
        "seed": {
            "user_id": user_id,
            "username": "dev-user",
            "kb_id": kb_id,
            "kb_name": "default",
            "milvus_collection": "kb_default",
            "embed_provider": "local",
            "embed_model": "bge-small",
            "embed_dim": 384,
        },
    }


async def _run_async(
    *,
    db_url: Optional[str],
    drop: bool,
    seed: bool,
    seed_fts: bool,
    rebuild_fts: bool,
    echo: Optional[bool],
) -> Dict[str, Any]:
    """
    [职责] 执行 init_db 的主流程（可选 drop/seed），输出 JSON-safe 结果。
    [边界] 不捕获 KeyboardInterrupt；异常交由上层统一处理。
    [上游关系] main 解析参数后调用。
    [下游关系] _print_summary 输出结果。
    """
    start_ms = time.perf_counter() * 1000.0  # docstring: 记录开始时间
    engine = create_engine(url=db_url, echo=echo)  # docstring: 创建 DB 引擎
    result: Dict[str, Any] = {
        "ok": True,
        "db_url": str(engine.url),
        "echo": engine.echo,
        "dropped": False,
        "created": False,
        "seeded": False,
        "seed_status": "skipped",
        "fts": {"ensured": False, "rebuilt": False},
        "duration_ms": 0.0,
        "error": None,
    }
    try:
        if drop:
            await drop_db(engine=engine)  # docstring: 可选 drop 所有表
            result["dropped"] = True  # docstring: 标记 drop 完成
        await init_db(engine=engine)  # docstring: 创建所有表
        result["created"] = True  # docstring: 标记 create 完成
        seed_result = await _maybe_seed(engine=engine, enabled=seed)  # docstring: 可选 seed
        result.update(seed_result)  # docstring: 合并 seed 结果

        Session = async_sessionmaker(engine, expire_on_commit=False)

        # docstring: FTS 初始化建议与 schema 创建绑定（避免 /api/chat keyword stage 缺表）
        if seed_fts:
            async with Session() as session:
                await ensure_sqlite_fts(session)
                result["fts"]["ensured"] = True
                if rebuild_fts:
                    await rebuild_sqlite_fts(session)
                    result["fts"]["rebuilt"] = True

    except Exception as exc:
        result["ok"] = False  # docstring: 标记失败
        result["error"] = f"{exc.__class__.__name__}: {exc}"  # docstring: 记录异常信息
    finally:
        await engine.dispose()  # docstring: 释放连接池
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
    print(f"[init_db] status={status}")  # docstring: 摘要行
    print(f"[init_db] db_url={result.get('db_url')} echo={result.get('echo')}")  # docstring: 连接信息
    print(
        f"[init_db] dropped={result.get('dropped')} created={result.get('created')} "
        f"seeded={result.get('seeded')} seed_status={result.get('seed_status')}"
    )  # docstring: 执行动作摘要
    if result.get("fts"):
        print(f"[init_db] fts={result.get('fts')}")  # docstring: FTS 初始化摘要
    if result.get("error"):
        print(f"[init_db] error={result.get('error')}")  # docstring: 错误摘要
    print(f"[init_db] duration_ms={result.get('duration_ms')}")  # docstring: 耗时输出


def main(argv: Optional[Sequence[str]] = None) -> int:
    """
    [职责] CLI 入口：解析参数并执行 init_db 流程。
    [边界] 捕获异常并转为非 0 退出码。
    [上游关系] 命令行或脚本调用。
    [下游关系] 触发 _run_async 并输出摘要。
    """
    args = _parse_args(argv)  # docstring: 解析参数
    try:
        result = asyncio.run(
            _run_async(
                db_url=args.db_url,
                drop=bool(args.drop),
                seed=bool(args.seed),
                seed_fts=bool(args.seed_fts),
                rebuild_fts=bool(args.rebuild_fts),
                echo=args.echo,
            )
        )  # docstring: 执行主流程
        _print_summary(result=result, as_json=bool(args.json))  # docstring: 输出摘要/JSON
        return 0 if result.get("ok") else 1  # docstring: 返回退出码
    except Exception as exc:
        error_result = {
            "ok": False,
            "db_url": args.db_url,
            "echo": args.echo,
            "dropped": bool(args.drop),
            "created": False,
            "seeded": False,
            "seed_status": "skipped",
            "duration_ms": 0.0,
            "error": f"{exc.__class__.__name__}: {exc}",
        }  # docstring: 异常结果
        _print_summary(result=error_result, as_json=bool(args.json))  # docstring: 输出错误摘要/JSON
        return 1


if __name__ == "__main__":
    raise SystemExit(main())  # docstring: CLI 入口
