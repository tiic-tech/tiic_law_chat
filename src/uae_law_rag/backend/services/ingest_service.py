# src/uae_law_rag/backend/services/ingest_service.py

"""
[职责] ingest_service：编排文件导入全流程（幂等判定 + 两阶段提交 + pipeline 编排 + Gate 裁决）。
[边界] 不暴露 HTTP 语义；不做底层 SDK 调用；不提交跨服务事务（仅控制 DB 事务）。
[上游关系] api/routers/ingest.py 或 scripts 调用 ingest_file 作为业务入口；依赖 TraceContext 与 session 注入。
[下游关系] pipelines/ingest 各步骤、IngestRepo 写入文件/文档/节点事实；MilvusRepo 写入向量。
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
from urllib.parse import urlparse
from uuid import NAMESPACE_URL, uuid5

from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from uae_law_rag.backend.kb.schema import (
    ARTICLE_ID_FIELD,
    DOCUMENT_ID_FIELD,
    EMBEDDING_FIELD,
    FILE_ID_FIELD,
    KB_ID_FIELD,
    NODE_ID_FIELD,
    PAGE_FIELD,
    SECTION_PATH_FIELD,
    VECTOR_ID_FIELD,
)
from uae_law_rag.backend.pipelines.base.context import PipelineContext
from uae_law_rag.backend.pipelines.ingest import embed as embed_mod
from uae_law_rag.backend.pipelines.ingest import pdf_parse as pdf_parse_mod
from uae_law_rag.backend.pipelines.ingest import persist_db as persist_db_mod
from uae_law_rag.backend.pipelines.ingest import persist_milvus as persist_milvus_mod
from uae_law_rag.backend.pipelines.ingest import segment as segment_mod
from uae_law_rag.backend.pipelines.ingest import pipeline as pipeline_mod
from uae_law_rag.backend.schemas.audit import TraceContext
from uae_law_rag.backend.utils.constants import (
    DEBUG_KEY,
    REQUEST_ID_KEY,
    TIMING_MS_KEY,
    TIMING_TOTAL_MS_KEY,
    TRACE_ID_KEY,
)
from uae_law_rag.backend.utils.errors import (
    BadRequestError,
    ConflictError,
    DomainError,
    ExternalDependencyError,
    InternalError,
    NotFoundError,
    PipelineError,
)
from uae_law_rag.backend.utils.logging_ import get_logger, log_event, truncate_text
from uae_law_rag.backend.utils.artifacts import (
    get_parsed_markdown_path,
    write_text_atomic,
    normalize_offsets_to_page_local,
)

INGEST_STATUS_PENDING = "pending"
INGEST_STATUS_SUCCESS = "success"
INGEST_STATUS_FAILED = "failed"

STATE_PENDING = "PENDING"
STATE_PARSING = "PARSING"
STATE_SEGMENTING = "SEGMENTING"
STATE_EMBEDDING = "EMBEDDING"
STATE_PERSISTING = "PERSISTING"
STATE_SUCCESS = "SUCCESS"
STATE_FAILED = "FAILED"

STATE_FLOW = {
    STATE_PENDING: {STATE_PARSING, STATE_FAILED},
    STATE_PARSING: {STATE_SEGMENTING, STATE_FAILED},
    STATE_SEGMENTING: {STATE_EMBEDDING, STATE_FAILED},
    STATE_EMBEDDING: {STATE_PERSISTING, STATE_FAILED},
    STATE_PERSISTING: {STATE_SUCCESS, STATE_FAILED},
}


@dataclass(frozen=True)
class IngestGateDecision:
    """
    [职责] IngestGateDecision：封装 ingest gate 裁决结果（是否通过 + 原因列表）。
    [边界] 仅表达裁决结果，不负责 DB 写回或异常抛出。
    [上游关系] ingest_service 在 pipeline 完成后调用 gate 评估。
    [下游关系] ingest_service 根据裁决决定 status=success/failed 并写回。
    """

    passed: bool
    reasons: List[str]


def _normalize_ingest_profile(ingest_profile: Optional[Dict[str, Any]]) -> Dict[str, str]:
    """
    [职责] 归一化 ingest_profile（parser/parse_version/segment_version）。
    [边界] 不读取外部配置；不做复杂 schema 校验。
    [上游关系] ingest_file 入参 ingest_profile。
    [下游关系] create_file 的 ingest_profile 快照。
    """
    raw = ingest_profile or {}
    parser_name = str(raw.get("parser") or "pymupdf4llm").strip()
    parse_version = str(raw.get("parse_version") or "v1").strip()
    segment_version = str(raw.get("segment_version") or "v1").strip()
    if not parser_name:
        parser_name = "pymupdf4llm"  # docstring: parser 默认值
    if not parse_version:
        parse_version = "v1"  # docstring: parse_version 默认值
    if not segment_version:
        segment_version = "v1"  # docstring: segment_version 默认值
    return {
        "parser": parser_name,
        "parse_version": parse_version,
        "segment_version": segment_version,
    }


def _resolve_source_uri(source_uri: str) -> Path:
    """
    [职责] 将 source_uri 解析为本地文件 Path。
    [边界] 仅支持 file:// 与本地路径；不处理远程下载。
    [上游关系] ingest_file 传入 source_uri。
    [下游关系] pdf_parse 需要本地 PDF 路径。
    """
    raw = str(source_uri or "").strip()
    if not raw:
        raise BadRequestError(message="source_uri is required")  # docstring: 源 URI 必填

    parsed = urlparse(raw)
    if parsed.scheme in ("", "file"):
        path_str = parsed.path if parsed.scheme == "file" else raw  # docstring: file:// 与本地路径兼容
        return Path(path_str).expanduser().resolve()

    raise BadRequestError(message=f"unsupported source_uri scheme: {parsed.scheme}")  # docstring: 限定本地路径


def _advance_state(current: str, next_state: str) -> str:
    """
    [职责] 显式状态机推进（校验状态转移合法性）。
    [边界] 仅维护 ingest_service 内部状态，不落库。
    [上游关系] ingest_file 编排各 pipeline 步骤时调用。
    [下游关系] debug 输出与日志可使用 state 追踪。
    """
    allowed = STATE_FLOW.get(current, set())
    if next_state not in allowed:
        raise InternalError(
            message="invalid ingest state transition",
            detail={"from": current, "to": next_state},
        )  # docstring: 状态机约束违反
    return next_state


def _evaluate_ingest_gate(*, pages: Optional[int], node_count: int, vector_count: int) -> IngestGateDecision:
    """
    [职责] 执行最小 ingest gate 裁决（结构/数量一致性）。
    [边界] 不做 FTS/Milvus 搜索校验；仅做基础计数校验。
    [上游关系] ingest_file 在 pipeline 完成后调用。
    [下游关系] ingest_service 写回 success/failed。
    """
    reasons: List[str] = []
    if pages is not None and int(pages) <= 0:
        reasons.append("pages_must_be_positive")  # docstring: 页数必须为正
    if int(node_count) <= 0:
        reasons.append("node_count_must_be_positive")  # docstring: 节点数量必须大于 0
    if int(vector_count) <= 0:
        reasons.append("vector_count_must_be_positive")  # docstring: 向量数量必须大于 0
    if int(vector_count) != int(node_count):
        reasons.append("vector_count_mismatch")  # docstring: 向量数量必须与节点一致
    return IngestGateDecision(passed=not reasons, reasons=reasons)


def _build_node_payload_index(
    node_dicts: Sequence[Dict[str, Any]],
    embeddings: Sequence[Any],
) -> Dict[int, Tuple[Dict[str, Any], Any]]:
    """
    [职责] 基于 node_index 构建 payload+embedding 对齐索引，避免 nodes_out 顺序变化导致错配。
    [边界] node_index 缺失时回退到输入顺序索引；冲突时以最后一次为准（应由上游保证唯一）。
    """
    by_index: Dict[int, Tuple[Dict[str, Any], Any]] = {}
    for i, payload in enumerate(node_dicts):
        try:
            idx = int(payload.get("node_index", i))
        except Exception:
            idx = i
        by_index[idx] = (payload, embeddings[i])
    return by_index


def _build_ingest_response(
    *,
    kb_id: str,
    file_id: str,
    status: str,
    node_count: int,
    timing_ms: Dict[str, Any],
    trace_id: str,
    request_id: str,
    debug_payload: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    [职责] 组装 ingest_service 的 JSON-safe 返回对象。
    [边界] 不做 HTTP 语义映射；字段名保持服务层合同。
    [上游关系] ingest_file 在成功/失败/幂等分支调用。
    [下游关系] routers 映射为 HTTP response（schemas_http）。
    """
    payload: Dict[str, Any] = {
        "kb_id": kb_id,
        "file_id": file_id,
        "status": status,
        "node_count": int(node_count),
        TIMING_MS_KEY: dict(timing_ms),
        TRACE_ID_KEY: str(trace_id),
        REQUEST_ID_KEY: str(request_id),
    }
    if debug_payload is not None:
        payload[DEBUG_KEY] = debug_payload  # docstring: debug 仅在显式开启时返回
    return payload


def _build_debug_payload(
    *,
    state: str,
    sha256: Optional[str],
    document_id: Optional[str],
    node_ids: Sequence[str],
    vector_ids: Sequence[str],
    gate: Optional[IngestGateDecision] = None,
) -> Dict[str, Any]:
    """
    [职责] 组装 debug 输出（审计锚点 + gate 决策）。
    [边界] 不返回全文；node_ids/vector_ids 做采样与计数。
    [上游关系] ingest_file 在 debug=True 时调用。
    [下游关系] debug 输出用于排障与回放。
    """
    node_sample = list(node_ids[:10])  # docstring: 控制 debug 体积
    vector_sample = list(vector_ids[:10])  # docstring: 控制 debug 体积
    payload: Dict[str, Any] = {
        "state": state,
        "sha256": sha256,
        "document_id": document_id,
        "node_ids_sample": node_sample,
        "node_ids_count": len(node_ids),
        "vector_ids_sample": vector_sample,
        "vector_ids_count": len(vector_ids),
    }
    if gate is not None:
        payload["gate"] = {"passed": gate.passed, "reasons": list(gate.reasons)}  # docstring: gate 裁决摘要
    return payload


def _classify_pipeline_error(exc: Exception, *, stage: Optional[str]) -> DomainError:
    """
    [职责] 将 pipeline 异常映射为 DomainError（Pipeline/ExternalDependency）。
    [边界] 仅基于 stage 与异常类型做最小分类。
    [上游关系] ingest_file 捕获异常后调用。
    [下游关系] routers/errors.py 映射为 HTTP。
    """
    if isinstance(exc, DomainError):
        return exc  # docstring: 已是领域错误直接透传

    detail = {
        "stage": stage or "",
        "error_type": exc.__class__.__name__,
        "error": str(exc),
    }
    if stage == "milvus":
        return ExternalDependencyError(
            message="milvus dependency failed",
            detail=detail,
            cause=exc,
        )  # docstring: Milvus 作为外部依赖
    return PipelineError(
        message="ingest pipeline failed",
        detail=detail,
        cause=exc,
    )


async def ingest_file(
    *,
    session: AsyncSession,
    kb_id: str,
    source_uri: str,
    file_name: Optional[str] = None,
    ingest_profile: Optional[Dict[str, Any]] = None,
    milvus_repo: Any,
    trace_context: Optional[TraceContext] = None,
    dry_run: bool = False,
    debug: bool = False,
) -> Dict[str, Any]:
    """
    [职责] ingest_file：执行文件导入（两阶段提交 + pipeline 编排 + gate 裁决）。
    [边界] 不负责 HTTP 处理；不负责 Milvus collection 创建；不负责重试调度；dry_run=True 仅做解析/切分/embedding 校验，不落库、不写 Milvus。
    [上游关系] routers/ingest.py 调用；trace_context 来自 middleware/deps。
    [下游关系] ingest pipeline 写入 DB/Milvus；返回可映射的 ingest 响应。
    """
    logger = get_logger("services.ingest")  # docstring: 统一服务日志入口

    kb_id = str(kb_id or "").strip()
    if not kb_id:
        raise BadRequestError(message="kb_id is required")  # docstring: KB ID 必填

    ctx = PipelineContext.from_session(session, trace_context=trace_context)  # docstring: 装配 pipeline ctx
    log_event(
        logger,
        logging.INFO,
        "ingest.start",
        context=ctx,
        fields={"kb_id": kb_id, "source_uri": truncate_text(source_uri)},
    )  # docstring: ingest 开始日志

    pdf_path = _resolve_source_uri(source_uri)  # docstring: 解析本地 PDF 路径
    if not pdf_path.exists():
        raise BadRequestError(message=f"source_uri not found: {pdf_path}")  # docstring: 文件必须存在
    if not pdf_path.is_file():
        raise BadRequestError(message=f"source_uri is not a file: {pdf_path}")  # docstring: 必须为文件

    file_name = (str(file_name).strip() if file_name is not None else "") or pdf_path.name  # docstring: 文件名兜底
    profile = _normalize_ingest_profile(ingest_profile)  # docstring: ingest_profile 归一化
    if profile["parser"] != "pymupdf4llm":
        raise BadRequestError(message=f"unsupported parser: {profile['parser']}")  # docstring: 固定 parser 策略

    with ctx.timing.stage("sha256"):
        sha256 = _sha256_file(pdf_path)  # docstring: 幂等指纹

    with ctx.timing.stage("idempotency"):
        existing = await ctx.ingest_repo.get_file_by_sha256(kb_id=kb_id, sha256=sha256)  # docstring: 幂等判定
    # docstring: 仅对 success 做短路；failed/pending 允许继续重试以实现可回放与自愈
    if (
        not dry_run
        and existing is not None
        and str(getattr(existing, "ingest_status", "") or "").lower() == INGEST_STATUS_SUCCESS
    ):
        # docstring: idempotent fast-path MUST still support page replay
        document_id = await pipeline_mod._get_document_id_by_file_id(session=session, file_id=str(existing.id))

        # docstring: ensure parsed markdown exists (backfill if missing)
        doc_row = await ctx.ingest_repo.get_document(document_id)  # or DocumentRepo(session).get_document(...)
        meta_doc = dict(getattr(doc_row, "meta_data", {}) or {}) if doc_row is not None else {}
        md_path_raw = str(meta_doc.get("parsed_markdown_path") or "").strip()
        md_ok = False
        if md_path_raw:
            try:
                p = Path(md_path_raw).expanduser().resolve()
                md_ok = bool(p.exists() and p.is_file())
            except Exception:
                md_ok = False

        if not md_ok:
            # docstring: backfill by re-parse only (no re-seg/embed/milvus)
            with ctx.timing.stage("parse_backfill"):
                parsed = await pdf_parse_mod.parse_pdf(
                    pdf_path=str(pdf_path),
                    parser_name=profile["parser"],
                    parse_version=profile["parse_version"],
                )
            md_text = str((parsed or {}).get("markdown") or "")
            if not md_text.strip():
                raise ValueError("parse produced empty markdown (backfill)")
            parsed_md_path = get_parsed_markdown_path(kb_id=kb_id, file_id=str(existing.id))
            write_text_atomic(parsed_md_path, md_text)
            meta_doc["parsed_markdown_path"] = str(parsed_md_path)
            try:
                pages = _infer_pages(parsed)
            except Exception:
                pages = None
            if pages is not None:
                meta_doc["pages"] = int(pages)
            if doc_row is not None:
                doc_row.meta_data = meta_doc
                await session.flush()
                await session.commit()

        timing_ms = ctx.timing_ms(total_key=TIMING_TOTAL_MS_KEY)  # docstring: 幂等分支 timing
        debug_payload = (
            _build_debug_payload(
                state=STATE_PENDING,
                sha256=sha256,
                document_id=document_id,
                node_ids=[],
                vector_ids=[],
            )
            if debug
            else None
        )
        log_event(
            logger,
            logging.INFO,
            "ingest.idempotent",
            context=ctx,
            fields={"kb_id": kb_id, "file_id": existing.id, "status": existing.ingest_status},
        )  # docstring: 幂等返回日志
        return _build_ingest_response(
            kb_id=kb_id,
            file_id=str(existing.id),
            status=str(existing.ingest_status or INGEST_STATUS_PENDING),
            node_count=int(existing.node_count or 0),
            timing_ms=timing_ms,
            trace_id=str(ctx.trace_id),
            request_id=str(ctx.request_id),
            debug_payload=debug_payload,
        )

    with ctx.timing.stage("load_kb"):
        kb = await ctx.ingest_repo.get_kb(kb_id)  # docstring: 加载 KB 配置
    if kb is None:
        raise NotFoundError(message=f"kb not found: {kb_id}")  # docstring: KB 必须存在

    collection = getattr(kb, "milvus_collection", None)
    if not collection:
        raise InternalError(message="milvus collection not configured", detail={"kb_id": kb_id})  # docstring: 配置缺失

    file_ext = _safe_ext(file_name)  # docstring: 扩展名归一化
    file_mtime = _safe_mtime(pdf_path)  # docstring: 文件 mtime 快照
    file_size = _safe_size(pdf_path)  # docstring: 文件 size 快照

    state = STATE_PENDING  # docstring: 显式状态机起点
    file_id = ""

    if dry_run:
        file_id = _dry_run_file_id(kb_id=kb_id, sha256=sha256)  # docstring: dry_run 稳定 file_id
        current_stage: Optional[str] = None
        node_dicts: List[Dict[str, Any]] = []
        embeddings: List[Any] = []
        parsed_pages: Optional[int] = None

        try:
            state = _advance_state(state, STATE_PARSING)  # docstring: dry_run 进入解析
            current_stage = "parse"
            ctx.with_provider(
                "parser", {"name": profile["parser"], "parse_version": profile["parse_version"]}
            )  # docstring: parser 快照
            with ctx.timing.stage("parse"):
                parsed = await pdf_parse_mod.parse_pdf(
                    pdf_path=str(pdf_path),
                    parser_name=profile["parser"],
                    parse_version=profile["parse_version"],
                )  # docstring: PDF -> Markdown（dry_run）

            parsed_pages = _infer_pages(parsed)  # docstring: 解析页数（dry_run）

            state = _advance_state(state, STATE_SEGMENTING)  # docstring: dry_run 进入切分
            current_stage = "segment"
            ctx.meta["segment_version"] = profile["segment_version"]  # docstring: 记录切分版本
            with ctx.timing.stage("segment"):
                node_dicts = await segment_mod.segment_nodes(
                    parsed=parsed,
                    chunking_config=getattr(kb, "chunking_config", {}) or {},
                    segment_version=profile["segment_version"],
                )  # docstring: Markdown -> Node payloads（dry_run）
            if not node_dicts:
                raise ValueError("segment produced no nodes")  # docstring: 禁止空节点集合

            # docstring: normalize offsets to page-local (P2-0g)
            try:
                md_text = str((parsed or {}).get("markdown") or "")
            except Exception:
                md_text = ""
            if md_text.strip():
                node_dicts = normalize_offsets_to_page_local(node_dicts=node_dicts, markdown=md_text)

            state = _advance_state(state, STATE_EMBEDDING)  # docstring: dry_run 进入 embedding
            current_stage = "embed"
            embed_provider = getattr(kb, "embed_provider", "ollama")
            embed_model = getattr(kb, "embed_model", "")
            embed_dim = getattr(kb, "embed_dim", None)
            ctx.with_provider(
                "embed",
                {"provider": embed_provider, "model": embed_model, "dim": embed_dim},
            )  # docstring: embedding 快照
            with ctx.timing.stage("embed"):
                embeddings = await embed_mod.embed_texts(
                    texts=[n.get("text") or "" for n in node_dicts],
                    provider=embed_provider,
                    model=embed_model,
                    dim=embed_dim,
                )  # docstring: 生成 embedding（dry_run）
            if len(embeddings) != len(node_dicts):
                raise ValueError("embedding count mismatch")  # docstring: 向量数量必须与节点一致

            gate = _evaluate_ingest_gate(
                pages=parsed_pages,
                node_count=len(node_dicts),
                vector_count=len(embeddings),
            )  # docstring: dry_run gate 裁决

            if gate.passed:
                state = _advance_state(state, STATE_PERSISTING)  # docstring: dry_run 虚拟持久化阶段
                state = _advance_state(state, STATE_SUCCESS)  # docstring: dry_run 成功
            else:
                state = _advance_state(state, STATE_FAILED)  # docstring: dry_run 失败

            timing_ms = ctx.timing_ms(total_key=TIMING_TOTAL_MS_KEY)  # docstring: dry_run timing
            timing_ms["dry_run"] = True  # docstring: dry_run 标记（避免误读为落库）
            debug_payload = (
                _build_debug_payload(
                    state=state,
                    sha256=sha256,
                    document_id=None,
                    node_ids=[],
                    vector_ids=[],
                    gate=gate,
                )
                if debug
                else None
            )
            if debug_payload is not None:
                debug_payload["dry_run"] = True  # docstring: debug 标注 dry_run
                debug_payload["node_ids_count"] = len(node_dicts)  # docstring: dry_run 节点数量
                debug_payload["vector_ids_count"] = len(embeddings)  # docstring: dry_run 向量数量

            log_event(
                logger,
                logging.INFO,
                "ingest.dry_run",
                context=ctx,
                fields={
                    "kb_id": kb_id,
                    "file_id": file_id,
                    "node_count": len(node_dicts),
                    "status": INGEST_STATUS_SUCCESS if gate.passed else INGEST_STATUS_FAILED,
                },
            )  # docstring: dry_run 日志

            return _build_ingest_response(
                kb_id=kb_id,
                file_id=file_id,
                status=INGEST_STATUS_SUCCESS if gate.passed else INGEST_STATUS_FAILED,
                node_count=len(node_dicts) if gate.passed else 0,
                timing_ms=timing_ms,
                trace_id=str(ctx.trace_id),
                request_id=str(ctx.request_id),
                debug_payload=debug_payload,
            )
        except Exception as exc:
            error = _classify_pipeline_error(exc, stage=current_stage)  # docstring: dry_run 异常归类
            log_event(
                logger,
                logging.ERROR,
                "ingest.dry_run_failed",
                context=ctx,
                fields={"kb_id": kb_id, "file_id": file_id, "stage": current_stage},
                exc_info=exc,
            )  # docstring: dry_run 失败日志
            raise error

    try:
        with ctx.timing.stage("db", accumulate=True):
            file_row = await persist_db_mod.create_file(
                repo=ctx.ingest_repo,
                kb_id=kb_id,
                file_name=file_name,
                sha256=sha256,
                source_uri=source_uri,
                file_ext=file_ext,
                file_version=1,
                file_mtime=file_mtime,
                file_size=file_size,
                pages=None,
                ingest_profile=profile,
            )  # docstring: phase-1 创建 pending 文件
        file_id = str(file_row.id)  # docstring: 稳定 file_id 锚点
        await session.commit()  # docstring: phase-1 提交 pending 记录
    except IntegrityError as exc:
        await session.rollback()  # docstring: 冲突时回滚
        raise ConflictError(
            message="file already ingested",
            detail={"kb_id": kb_id, "sha256": sha256},
            cause=exc,
        )
    except Exception as exc:
        await session.rollback()  # docstring: 未知异常回滚
        raise InternalError(message="failed to create ingest file", detail={"kb_id": kb_id}, cause=exc)

    current_stage: Optional[str] = None
    nodes_out: List[Any] = []
    vector_ids: List[str] = []
    document_id: Optional[str] = None
    parsed_pages: Optional[int] = None
    milvus_upsert_done = False

    try:
        state = _advance_state(state, STATE_PARSING)  # docstring: 状态推进到 PARSING
        current_stage = "parse"
        ctx.with_provider(
            "parser", {"name": profile["parser"], "parse_version": profile["parse_version"]}
        )  # docstring: parser 快照
        with ctx.timing.stage("parse"):
            parsed = await pdf_parse_mod.parse_pdf(
                pdf_path=str(pdf_path),
                parser_name=profile["parser"],
                parse_version=profile["parse_version"],
            )  # docstring: PDF -> Markdown

        # docstring: persist parsed markdown for page replay (P2-0e)
        md_text = str((parsed or {}).get("markdown") or "")
        if not md_text.strip():
            raise ValueError("parse produced empty markdown")
        parsed_md_path = get_parsed_markdown_path(kb_id=kb_id, file_id=str(file_id))
        write_text_atomic(parsed_md_path, md_text)
        ctx.meta["parsed_markdown_path"] = str(parsed_md_path)

        parsed_pages = _infer_pages(parsed)  # docstring: 解析页数
        if parsed_pages is not None:
            file_row.pages = parsed_pages  # docstring: 回填页数
            await session.flush()  # docstring: 持久化页数

        state = _advance_state(state, STATE_SEGMENTING)  # docstring: 状态推进到 SEGMENTING
        current_stage = "segment"
        ctx.meta["segment_version"] = profile["segment_version"]  # docstring: 记录切分版本
        with ctx.timing.stage("segment"):
            node_dicts = await segment_mod.segment_nodes(
                parsed=parsed,
                chunking_config=getattr(kb, "chunking_config", {}) or {},
                segment_version=profile["segment_version"],
            )  # docstring: Markdown -> Node payloads
        if not node_dicts:
            raise ValueError("segment produced no nodes")  # docstring: 禁止空节点集合

        # docstring: attach page-local offsets for replay/highlight (P0-2d)
        try:
            md_text = str((parsed or {}).get("markdown") or "")
        except Exception:
            md_text = ""
        if md_text.strip():
            node_dicts = normalize_offsets_to_page_local(node_dicts=node_dicts, markdown=md_text)

        state = _advance_state(state, STATE_EMBEDDING)  # docstring: 状态推进到 EMBEDDING
        current_stage = "embed"
        embed_provider = getattr(kb, "embed_provider", "ollama")
        embed_model = getattr(kb, "embed_model", "")
        embed_dim = getattr(kb, "embed_dim", None)
        ctx.with_provider(
            "embed",
            {"provider": embed_provider, "model": embed_model, "dim": embed_dim},
        )  # docstring: embedding 快照
        with ctx.timing.stage("embed"):
            embeddings = await embed_mod.embed_texts(
                texts=[n.get("text") or "" for n in node_dicts],
                provider=embed_provider,
                model=embed_model,
                dim=embed_dim,
            )  # docstring: 生成 embedding
        if len(embeddings) != len(node_dicts):
            raise ValueError("embedding count mismatch")  # docstring: 向量数量必须与节点一致

        by_index = _build_node_payload_index(node_dicts, embeddings)  # docstring: 构建 payload+embedding 对齐索引

        state = _advance_state(state, STATE_PERSISTING)  # docstring: 状态推进到 PERSISTING
        current_stage = "persist_db"
        with ctx.timing.stage("db", accumulate=True):
            doc, nodes_out = await persist_db_mod.persist_document_nodes(
                repo=ctx.ingest_repo,
                kb_id=kb_id,
                file_id=file_id,
                nodes=node_dicts,
                title=file_name,
                source_name=file_name,
                meta_data={"parser": profile["parser"], "parse_version": profile["parse_version"]},
            )  # docstring: 落库 Document + Nodes
        document_id = str(doc.id)  # docstring: 文档 ID 记录

        # docstring: write replay pointer into Document meta_data (P2-0e)
        try:
            meta_doc = dict(getattr(doc, "meta_data", {}) or {})
        except Exception:
            meta_doc = {}
        meta_doc["parsed_markdown_path"] = str(parsed_md_path)
        meta_doc["source_uri"] = str(source_uri)
        meta_doc["sha256"] = str(sha256)
        if parsed_pages is not None:
            meta_doc["pages"] = int(parsed_pages)
        if "language" not in meta_doc:
            meta_doc["language"] = "en"  # docstring: reserved for bilingual alignment
        doc.meta_data = meta_doc
        await session.flush()

        nodes_out = sorted(
            list(nodes_out), key=lambda n: int(getattr(n, "node_index", 0))
        )  # docstring: 统一按 node_index 输出顺序

        entities: List[Dict[str, Any]] = []
        for i, node in enumerate(nodes_out):
            node_index = int(getattr(node, "node_index", i))
            if node_index not in by_index:
                raise ValueError("node_index mismatch between persisted nodes and input payloads")
            payload, embedding = by_index[node_index]  # docstring: 基于 node_index 对齐 payload+embedding
            page = payload.get("page")
            if page is None:
                page = getattr(node, "page", None)  # docstring: 回退到 DB 节点页码
            if page is None and parsed_pages == 1:
                page = 1  # docstring: 单页文档兜底 page=1

            vector_id = _new_vector_id(kb_id=kb_id, node_id=node.id)  # docstring: 稳定向量主键
            vector_ids.append(vector_id)
            entities.append(
                {
                    VECTOR_ID_FIELD: vector_id,
                    EMBEDDING_FIELD: embedding,
                    NODE_ID_FIELD: node.id,
                    KB_ID_FIELD: kb_id,
                    FILE_ID_FIELD: file_id,
                    DOCUMENT_ID_FIELD: document_id,
                    PAGE_FIELD: page,
                    ARTICLE_ID_FIELD: getattr(node, "article_id", None) or payload.get("article_id") or "",
                    SECTION_PATH_FIELD: getattr(node, "section_path", None) or payload.get("section_path") or "",
                }
            )  # docstring: Milvus payload（与 schema 对齐）

        current_stage = "milvus"
        with ctx.timing.stage("milvus"):
            await persist_milvus_mod.upsert(
                milvus_repo=milvus_repo,
                collection=collection,
                entities=entities,
                embed_dim=embed_dim,
            )  # docstring: 写入 Milvus 向量实体
        milvus_upsert_done = True  # docstring: 标记 Milvus 写入完成

        maps = [{"node_id": nodes_out[i].id, "vector_id": vector_ids[i]} for i in range(len(nodes_out))]
        with ctx.timing.stage("db", accumulate=True):
            await persist_db_mod.persist_node_vector_maps(
                repo=ctx.ingest_repo,
                kb_id=kb_id,
                file_id=file_id,
                maps=maps,
            )  # docstring: 写入 node↔vector 映射

        gate = _evaluate_ingest_gate(
            pages=parsed_pages,
            node_count=len(nodes_out),
            vector_count=len(vector_ids),
        )  # docstring: ingest gate 裁决

        if not gate.passed:
            await session.rollback()  # docstring: gate 失败回滚 DB 事务
            await _cleanup_milvus(milvus_repo, collection=collection, file_id=file_id)  # docstring: 清理向量
            await _mark_file_failed(session=session, repo=ctx.ingest_repo, file_id=file_id)  # docstring: 写回 failed
            state = _advance_state(state, STATE_FAILED)  # docstring: 状态推进到 FAILED
            timing_ms = ctx.timing_ms(total_key=TIMING_TOTAL_MS_KEY)  # docstring: gate 失败 timing
            debug_payload = (
                _build_debug_payload(
                    state=state,
                    sha256=sha256,
                    document_id=document_id,
                    node_ids=[str(n.id) for n in nodes_out],
                    vector_ids=vector_ids,
                    gate=gate,
                )
                if debug
                else None
            )
            log_event(
                logger,
                logging.WARNING,
                "ingest.gate_failed",
                context=ctx,
                fields={"kb_id": kb_id, "file_id": file_id, "reasons": gate.reasons},
            )  # docstring: gate 失败日志
            return _build_ingest_response(
                kb_id=kb_id,
                file_id=file_id,
                status=INGEST_STATUS_FAILED,
                node_count=0,
                timing_ms=timing_ms,
                trace_id=str(ctx.trace_id),
                request_id=str(ctx.request_id),
                debug_payload=debug_payload,
            )

        with ctx.timing.stage("db", accumulate=True):
            await persist_db_mod.mark_file_ingested(
                repo=ctx.ingest_repo,
                file_id=file_id,
                node_count=len(nodes_out),
                last_ingested_at=datetime.now(timezone.utc),
            )  # docstring: 标记文件导入成功
        await session.commit()  # docstring: phase-2 提交成功路径
        state = _advance_state(state, STATE_SUCCESS)  # docstring: 状态推进到 SUCCESS

        timing_ms = ctx.timing_ms(total_key=TIMING_TOTAL_MS_KEY)  # docstring: 成功路径 timing
        debug_payload = (
            _build_debug_payload(
                state=state,
                sha256=sha256,
                document_id=document_id,
                node_ids=[str(n.id) for n in nodes_out],
                vector_ids=vector_ids,
                gate=gate,
            )
            if debug
            else None
        )
        log_event(
            logger,
            logging.INFO,
            "ingest.success",
            context=ctx,
            fields={"kb_id": kb_id, "file_id": file_id, "node_count": len(nodes_out)},
        )  # docstring: ingest 成功日志
        return _build_ingest_response(
            kb_id=kb_id,
            file_id=file_id,
            status=INGEST_STATUS_SUCCESS,
            node_count=len(nodes_out),
            timing_ms=timing_ms,
            trace_id=str(ctx.trace_id),
            request_id=str(ctx.request_id),
            debug_payload=debug_payload,
        )
    except Exception as exc:
        await session.rollback()  # docstring: pipeline 异常回滚
        await _cleanup_milvus(
            milvus_repo, collection=collection, file_id=file_id, enabled=milvus_upsert_done
        )  # docstring: 清理向量
        await _mark_file_failed(session=session, repo=ctx.ingest_repo, file_id=file_id)  # docstring: 写回 failed
        error = _classify_pipeline_error(exc, stage=current_stage)  # docstring: 异常归类
        log_event(
            logger,
            logging.ERROR,
            "ingest.failed",
            context=ctx,
            fields={"kb_id": kb_id, "file_id": file_id, "stage": current_stage},
            exc_info=exc,
        )  # docstring: ingest 失败日志
        raise error


async def _mark_file_failed(*, session: AsyncSession, repo: Any, file_id: str) -> None:
    """
    [职责] 写回文件 ingest_status=failed 并提交事务。
    [边界] 不处理不存在的 file_id；异常转为 InternalError。
    [上游关系] ingest_file 失败路径调用。
    [下游关系] KnowledgeFile.ingest_status 更新为 failed。
    """
    if not file_id:
        return  # docstring: file_id 为空直接跳过
    try:
        await persist_db_mod.mark_file_failed(
            repo=repo,
            file_id=file_id,
            node_count=0,
            last_ingested_at=datetime.now(timezone.utc),
        )  # docstring: 标记失败
        await session.commit()  # docstring: 提交 failed 状态
    except Exception as exc:
        await session.rollback()  # docstring: 写回失败时回滚
        raise InternalError(message="failed to mark ingest failed", detail={"file_id": file_id}, cause=exc)


async def _cleanup_milvus(
    milvus_repo: Any,
    *,
    collection: str,
    file_id: str,
    enabled: bool = True,
) -> None:
    """
    [职责] best-effort 清理 Milvus 向量（按 file_id 过滤）。
    [边界] 清理失败不抛异常；仅用于失败路径补偿。
    [上游关系] ingest_file 失败/gate 失败路径调用。
    [下游关系] Milvus collection 删除对应 file_id 向量。
    """
    if not enabled:
        return  # docstring: 未写入向量则跳过
    if not file_id or not hasattr(milvus_repo, "delete_by_expr"):
        return  # docstring: 依赖缺失或 file_id 为空则跳过
    try:
        fid = str(file_id).replace("'", "\\'")
        expr = f"file_id == '{fid}'"  # docstring: 按 file_id 清理向量
        await milvus_repo.delete_by_expr(collection=collection, expr=expr)  # docstring: best-effort 删除
    except Exception:
        return  # docstring: 忽略清理失败，避免覆盖主异常


def _sha256_file(path: Path) -> str:
    """
    [职责] 计算文件 sha256（幂等 key）。
    [边界] 按块读取，避免大文件内存峰值。
    [上游关系] ingest_file 幂等判定前调用。
    [下游关系] KnowledgeFile.sha256。
    """
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(1024 * 1024)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _safe_ext(name: str) -> str:
    """
    [职责] 归一化扩展名。
    [边界] 仅字符串处理；不验证 mime。
    [上游关系] ingest_file 创建 file 记录时调用。
    [下游关系] KnowledgeFile.file_ext。
    """
    ext = Path(name).suffix.lstrip(".").lower().strip()
    return ext or "pdf"


def _safe_mtime(p: Path) -> float:
    """
    [职责] 获取文件 mtime（尽量不抛异常）。
    [边界] 文件系统异常时返回 0。
    [上游关系] ingest_file 创建 file 记录时调用。
    [下游关系] KnowledgeFile.file_mtime。
    """
    try:
        return float(p.stat().st_mtime)
    except Exception:
        return 0.0


def _safe_size(p: Path) -> int:
    """
    [职责] 获取文件 size（尽量不抛异常）。
    [边界] 文件系统异常时返回 0。
    [上游关系] ingest_file 创建 file 记录时调用。
    [下游关系] KnowledgeFile.file_size。
    """
    try:
        return int(p.stat().st_size)
    except Exception:
        return 0


def _new_vector_id(*, kb_id: str, node_id: str) -> str:
    """
    [职责] 生成稳定 vector_id（Milvus PK）。
    [边界] 使用 hash(kb_id:node_id)；不依赖 Milvus auto_id。
    [上游关系] ingest_file 构造 Milvus payload 时调用。
    [下游关系] NodeVectorMap.vector_id / Milvus PK。
    """
    raw = f"{kb_id}:{node_id}".encode("utf-8")
    return hashlib.sha1(raw).hexdigest()[:36]


def _dry_run_file_id(*, kb_id: str, sha256: str) -> str:
    """
    [职责] 生成 dry_run 稳定 file_id（避免与真实落库 ID 冲突）。
    [边界] 仅用于 dry_run；不写入 DB。
    [上游关系] ingest_file(dry_run=True) 调用。
    [下游关系] 返回给上游用于审计与 UI 展示。
    """
    # docstring: 使用 namespace + 指纹生成可复现 UUID
    return str(uuid5(NAMESPACE_URL, f"dryrun:{kb_id}:{sha256}"))


def _infer_pages(parsed: Any) -> Optional[int]:
    """
    [职责] 从解析结果推断页数。
    [边界] 推断失败返回 None。
    [上游关系] ingest_file 在 parse 后调用。
    [下游关系] KnowledgeFile.pages / gate 断言。
    """
    try:
        v = getattr(parsed, "pages", None)
        if isinstance(v, int):
            return v
    except Exception:
        pass
    try:
        if isinstance(parsed, dict):
            for key in ("pages", "page_count", "num_pages"):
                if isinstance(parsed.get(key), int):
                    return int(parsed[key])
    except Exception:
        pass
    return None
