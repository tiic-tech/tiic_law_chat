# src/uae_law_rag/backend/scripts/init_fts.py

"""
[职责] 初始化 SQLite FTS（node_fts + triggers），并提供可复现、可幂等的 CLI 入口。
[边界] 仅负责 FTS 结构与索引重建；不创建业务数据；不执行 ingest/chat。
[上游关系] 本地开发/CI/部署脚本调用；依赖 db.engine + db.fts。
[下游关系] retrieval/keyword.py 依赖 node_fts；确保 /api/chat keyword stage 不因缺表失败。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from typing import Any, Dict, Optional, Sequence, Tuple

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

from uae_law_rag.backend.db.engine import create_engine
from uae_law_rag.backend.db.fts import FTS_TABLE, ensure_sqlite_fts, rebuild_sqlite_fts


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Initialize SQLite FTS (node_fts) and triggers.")
    parser.add_argument("--db-url", dest="db_url", default=None)  # docstring: 显式 DB 连接串
    parser.add_argument("--drop", action="store_true")  # docstring: 删除 FTS 虚表与 triggers
    parser.add_argument("--rebuild", action="store_true")  # docstring: 从 node 表重建 FTS 内容
    parser.add_argument("--check", action="store_true")  # docstring: 仅校验/打印状态（不修改）
    echo_group = parser.add_mutually_exclusive_group()
    echo_group.add_argument("--echo", dest="echo", action="store_true", default=None)
    echo_group.add_argument("--no-echo", dest="echo", action="store_false")
    parser.add_argument("--json", action="store_true")  # docstring: 仅输出 JSON
    return parser


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    return _build_parser().parse_args(list(argv) if argv is not None else None)


async def _exists(session: AsyncSession, *, name: str, type_: str) -> bool:
    row = (
        await session.execute(
            text("SELECT 1 FROM sqlite_master WHERE type=:t AND name=:n LIMIT 1"),
            {"t": type_, "n": name},
        )
    ).first()
    return bool(row)


async def _list_triggers(session: AsyncSession) -> Dict[str, bool]:
    out: Dict[str, bool] = {}
    for trig in ("node_ai", "node_au", "node_ad"):
        out[trig] = await _exists(session, name=trig, type_="trigger")
    return out


async def _counts(session: AsyncSession) -> Tuple[int, int]:
    # returns (node_count, fts_count) ; if fts missing, fts_count = -1
    node_count = int((await session.execute(text("SELECT COUNT(*) FROM node"))).scalar() or 0)
    if not await _exists(session, name=FTS_TABLE, type_="table"):
        return node_count, -1
    fts_count = int((await session.execute(text(f"SELECT COUNT(*) FROM {FTS_TABLE}"))).scalar() or 0)
    return node_count, fts_count


async def _drop_fts(session: AsyncSession) -> Dict[str, Any]:
    """
    Drop triggers and FTS virtual table (best-effort, idempotent).
    """
    dropped: Dict[str, Any] = {"triggers": {}, "table": False}

    # Drop triggers first
    for trig in ("node_ai", "node_au", "node_ad"):
        try:
            await session.execute(text(f"DROP TRIGGER IF EXISTS {trig};"))
            dropped["triggers"][trig] = True
        except Exception:
            dropped["triggers"][trig] = False

    # Drop FTS table
    try:
        await session.execute(text(f"DROP TABLE IF EXISTS {FTS_TABLE};"))
        dropped["table"] = True
    except Exception:
        dropped["table"] = False

    await session.commit()
    return dropped


async def _check_status(session: AsyncSession) -> Dict[str, Any]:
    table_ok = await _exists(session, name=FTS_TABLE, type_="table")
    triggers = await _list_triggers(session)
    node_count, fts_count = await _counts(session)
    return {
        "fts_table_exists": table_ok,
        "triggers": triggers,
        "node_count": node_count,
        "fts_count": fts_count,
    }


async def _run_async(
    *,
    db_url: Optional[str],
    drop: bool,
    rebuild: bool,
    check: bool,
    echo: Optional[bool],
) -> Dict[str, Any]:
    start_ms = time.perf_counter() * 1000.0
    engine: AsyncEngine = create_engine(url=db_url, echo=echo)

    result: Dict[str, Any] = {
        "ok": True,
        "db_url": str(engine.url),
        "echo": engine.echo,
        "dropped": None,
        "ensured": False,
        "rebuilt": False,
        "checked": None,
        "duration_ms": 0.0,
        "error": None,
    }

    try:
        from sqlalchemy.ext.asyncio import async_sessionmaker

        Session = async_sessionmaker(engine, expire_on_commit=False)

        async with Session() as session:
            # check-only mode
            if check and not drop and not rebuild:
                result["checked"] = await _check_status(session)
                return result

            if drop:
                result["dropped"] = await _drop_fts(session)

            # ensure structure
            await ensure_sqlite_fts(session)
            result["ensured"] = True

            # rebuild content if asked
            if rebuild:
                await rebuild_sqlite_fts(session)
                result["rebuilt"] = True

            # final check snapshot
            result["checked"] = await _check_status(session)

    except Exception as exc:
        result["ok"] = False
        result["error"] = f"{exc.__class__.__name__}: {exc}"
    finally:
        await engine.dispose()
        result["duration_ms"] = round(time.perf_counter() * 1000.0 - start_ms, 2)

    return result


def _print_summary(*, result: Dict[str, Any], as_json: bool) -> None:
    if as_json:
        print(json.dumps(result, ensure_ascii=True, default=str))
        return

    status = "ok" if result.get("ok") else "failed"
    print(f"[init_fts] status={status}")
    print(f"[init_fts] db_url={result.get('db_url')} echo={result.get('echo')}")
    if result.get("dropped") is not None:
        print(f"[init_fts] dropped={result.get('dropped')}")
    print(f"[init_fts] ensured={result.get('ensured')} rebuilt={result.get('rebuilt')}")

    chk = result.get("checked") or {}
    if chk:
        print("[init_fts] check=" + json.dumps(chk, ensure_ascii=True, default=str))

    if result.get("error"):
        print(f"[init_fts] error={result.get('error')}")
    print(f"[init_fts] duration_ms={result.get('duration_ms')}")


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    try:
        result = asyncio.run(
            _run_async(
                db_url=args.db_url,
                drop=bool(args.drop),
                rebuild=bool(args.rebuild),
                check=bool(args.check),
                echo=args.echo,
            )
        )
        _print_summary(result=result, as_json=bool(args.json))
        return 0 if result.get("ok") else 1
    except Exception as exc:
        error_result = {
            "ok": False,
            "db_url": args.db_url,
            "echo": args.echo,
            "dropped": None,
            "ensured": False,
            "rebuilt": False,
            "checked": None,
            "duration_ms": 0.0,
            "error": f"{exc.__class__.__name__}: {exc}",
        }
        _print_summary(result=error_result, as_json=bool(args.json))
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
