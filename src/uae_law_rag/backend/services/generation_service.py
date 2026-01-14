# src/uae_law_rag/backend/services/generation_service.py

"""
[职责] generation_service：封装 generation pipeline 执行与 gate 记录，输出可审计生成快照。
[边界] 不修改 message/status；不执行 evaluator；不提交事务。
[上游关系] chat_service 调用 execute_generation。
[下游关系] evaluator/chat 使用 GenerationBundle 与 gate/快照。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Sequence

from sqlalchemy.ext.asyncio import AsyncSession

from uae_law_rag.backend.db.repo.generation_repo import GenerationRepo
from uae_law_rag.backend.pipelines.base.context import PipelineContext
from uae_law_rag.backend.pipelines.generation.pipeline import run_generation_pipeline
from uae_law_rag.backend.schemas.generation import GenerationBundle
from uae_law_rag.backend.schemas.retrieval import RetrievalBundle
from uae_law_rag.backend.utils.constants import PROVIDER_SNAPSHOT_KEY, TIMING_MS_KEY

__all__ = [
    "GenerationGateDecision",
    "GenerationServiceResult",
    "execute_generation",
    "_extract_generation_citations",
]


@dataclass(frozen=True)
class GenerationGateDecision:
    """
    [职责] GenerationGateDecision：封装 generation gate 状态与原因。
    [边界] 仅记录生成状态，不裁决 message.status。
    [上游关系] execute_generation 在 pipeline 完成后调用。
    [下游关系] chat_service debug 输出与 evaluator 裁决解释。
    """

    status: str
    reasons: Sequence[str]


@dataclass(frozen=True)
class GenerationServiceResult:
    """
    [职责] GenerationServiceResult：生成阶段输出汇总（bundle + 快照 + gate）。
    [边界] 不包含 message/status 写回；仅提供审计快照。
    [上游关系] execute_generation 返回。
    [下游关系] chat_service 组装 debug/状态机。
    """

    bundle: GenerationBundle
    record_id: str
    provider_snapshot: Dict[str, Any]
    timing_ms: Dict[str, Any]
    gate: GenerationGateDecision


def _evaluate_generation_gate(bundle: GenerationBundle) -> GenerationGateDecision:
    """
    [职责] 执行最小 generation gate 记录（状态 + 原因）。
    [边界] 不裁决 message.status；仅记录 generation 状态。
    [上游关系] execute_generation 在 pipeline 完成后调用。
    [下游关系] debug 输出与 evaluator 裁决解释。
    """
    status = str(getattr(bundle.record, "status", "") or "failed").strip().lower()
    reasons = []
    error_message = str(getattr(bundle.record, "error_message", "") or "").strip()
    if error_message:
        reasons.append(error_message)  # docstring: 透传 generation 错误
    if status and status != "success":
        reasons.append(f"generation_status={status}")  # docstring: 记录状态
    return GenerationGateDecision(status=status or "failed", reasons=tuple(reasons))


def _extract_generation_citations(bundle: Optional[GenerationBundle]) -> list[Dict[str, Any]]:
    """
    [职责] 从 GenerationBundle 提取 citations 列表（JSON-safe）。
    [边界] bundle 缺失或格式异常时返回空列表。
    [上游关系] chat_service generation 结束后调用。
    [下游关系] response.citations。
    """
    if bundle is None:
        return []  # docstring: 无 bundle 直接回退
    payload = getattr(bundle.record, "citations", None)  # docstring: CitationsPayload
    if payload is None:
        return []  # docstring: 缺失 citations 回退空
    if hasattr(payload, "model_dump"):
        items = payload.model_dump().get("items")  # type: ignore[attr-defined]
    elif hasattr(payload, "dict"):
        items = payload.dict().get("items")  # type: ignore[call-arg]
    elif isinstance(payload, Mapping):
        items = payload.get("items")
    else:
        items = None
    if isinstance(items, list):
        return list(items)  # docstring: citations items
    return []  # docstring: 非列表回退空


async def execute_generation(
    *,
    session: AsyncSession,
    generation_repo: GenerationRepo,
    message_id: str,
    retrieval_bundle: RetrievalBundle,
    config: Mapping[str, Any],
    ctx: PipelineContext,
) -> GenerationServiceResult:
    """
    [职责] 执行 generation pipeline 并汇总结果快照。
    [边界] 不提交事务；不创建 message；不触发 evaluator。
    [上游关系] chat_service 调用。
    [下游关系] chat_service 获取 bundle/gate/provider_snapshot。
    """
    bundle = await run_generation_pipeline(
        session=session,
        generation_repo=generation_repo,
        message_id=message_id,
        retrieval_bundle=retrieval_bundle,
        config=dict(config),
        ctx=ctx,
    )  # docstring: 执行 generation pipeline

    record_id = str(bundle.record.id)  # docstring: generation_record_id 快照
    messages_snapshot = dict(getattr(bundle.record, "messages_snapshot", {}) or {})  # docstring: messages_snapshot 快照
    provider_snapshot = dict(messages_snapshot.get(PROVIDER_SNAPSHOT_KEY) or {})  # docstring: provider_snapshot 快照
    timing_ms = dict(messages_snapshot.get(TIMING_MS_KEY) or {})  # docstring: timing_ms 快照
    gate = _evaluate_generation_gate(bundle)  # docstring: gate 记录

    return GenerationServiceResult(
        bundle=bundle,
        record_id=record_id,
        provider_snapshot=provider_snapshot,
        timing_ms=timing_ms,
        gate=gate,
    )  # docstring: 返回结果快照
