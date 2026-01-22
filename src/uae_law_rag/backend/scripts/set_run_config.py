# src/uae_law_rag/backend/scripts/set_run_config.py

"""
[职责] set_run_config：写入全局运行配置（run_config）到 DB，供服务层默认读取。
[边界] 不触发 ingest/chat；仅写入配置快照。
[上游关系] 本地开发/部署脚本调用。
[下游关系] services/chat_service 与 ingest_service 读取 run_config。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from typing import Any, Dict, Optional, Sequence

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from uae_law_rag.config import settings
from uae_law_rag.backend.db.engine import create_engine
from uae_law_rag.backend.db.repo.run_config_repo import RunConfigRepo
from uae_law_rag.backend.utils.constants import DEFAULT_PROMPT_NAME, DEFAULT_PROMPT_VERSION


def _build_default_config() -> Dict[str, Any]:
    """
    [职责] 构造默认 run_config 快照（来源于 Settings 与系统内置默认值）。
    [边界] 不读取 DB；仅生成 JSON-safe dict。
    """
    default_chat_provider = "ollama" if settings.LOCAL_MODELS else "dashscope"
    default_chat_model = settings.OLLAMA_CHAT_MODEL if settings.LOCAL_MODELS else settings.QWEN_CHAT_MODEL
    default_embed_provider = "ollama" if settings.LOCAL_MODELS else "dashscope"
    default_embed_model = settings.OLLAMA_EMBED_MODEL if settings.LOCAL_MODELS else settings.QWEN_EMBED_MODEL

    cfg: Dict[str, Any] = {
        # retrieval defaults
        "keyword_top_k": 200,
        "vector_top_k": 50,
        "fusion_top_k": 50,
        "rerank_top_k": int(settings.RERANKER_TOP_N),
        "fusion_strategy": "union",
        "rerank_strategy": "bge_reranker" if str(settings.RERANKER_MODEL_PATH).strip() else "none",
        "rerank_model": str(settings.RERANKER_MODEL_PATH).strip() or None,
        "rerank_config": {"device": str(settings.RERANKER_DEVICE).strip()}
        if str(settings.RERANKER_DEVICE).strip()
        else {},
        "metric_type": "COSINE",
        # generation defaults
        "model_provider": str(default_chat_provider),
        "model_name": str(default_chat_model),
        "prompt_name": str(DEFAULT_PROMPT_NAME),
        "prompt_version": str(DEFAULT_PROMPT_VERSION),
        "generation_config": {},
        "prompt_config": {},
        "postprocess_config": {},
        "no_evidence_use_llm": False,
        # evaluator defaults
        "evaluator_config": {},
        # embed defaults (KB overrides)
        "embed_provider": str(default_embed_provider),
        "embed_model": str(default_embed_model),
        # ingest defaults
        "parser": "pymupdf4llm",
        "parse_version": "v1",
        "segment_version": "v1",
    }

    # prune None values
    return {k: v for k, v in cfg.items() if v is not None}


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Upsert run_config defaults.")
    parser.add_argument("--db-url", dest="db_url", default=None)  # docstring: 显式 DB 连接串
    parser.add_argument("--config-json", dest="config_json", default=None)  # docstring: 传入 JSON 字符串
    parser.add_argument("--merge", action="store_true")  # docstring: 与现有 config 合并
    parser.add_argument("--json", action="store_true")  # docstring: JSON 输出
    return parser


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    return _build_parser().parse_args(list(argv) if argv is not None else None)


async def _run_async(
    *,
    db_url: Optional[str],
    config_json: Optional[str],
    merge: bool,
) -> Dict[str, Any]:
    start_ms = time.perf_counter() * 1000.0
    engine = create_engine(url=db_url)
    Session = async_sessionmaker(bind=engine, expire_on_commit=False, class_=AsyncSession)
    result: Dict[str, Any] = {
        "ok": True,
        "db_url": str(engine.url),
        "merged": bool(merge),
        "updated": False,
        "config_keys": [],
        "duration_ms": 0.0,
        "error": None,
    }
    try:
        if config_json:
            config = json.loads(config_json)
        else:
            config = _build_default_config()
        if not isinstance(config, dict):
            raise ValueError("config must be a JSON object")

        async with Session() as session:
            async with session.begin():
                repo = RunConfigRepo(session)
                if merge:
                    existing = await repo.get_default_config()
                    merged = dict(existing or {})
                    merged.update(config)
                    config = merged
                row = await repo.upsert_default(config=config)
                result["updated"] = True
                result["config_keys"] = sorted(list((row.config or {}).keys()))
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
    print(f"[set_run_config] status={status}")
    print(f"[set_run_config] db_url={result.get('db_url')}")
    print(f"[set_run_config] merged={result.get('merged')} updated={result.get('updated')}")
    print(f"[set_run_config] keys={result.get('config_keys')}")
    if result.get("error"):
        print(f"[set_run_config] error={result.get('error')}")


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _parse_args(argv)
    result = asyncio.run(
        _run_async(
            db_url=args.db_url,
            config_json=args.config_json,
            merge=bool(args.merge),
        )
    )
    _print_summary(result=result, as_json=bool(args.json))


if __name__ == "__main__":
    main()
