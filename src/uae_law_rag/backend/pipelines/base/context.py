# src/uae_law_rag/backend/pipelines/base/context.py

"""
[职责] PipelineContext：pipelines 的运行上下文（依赖装配与共享元数据），统一注入 session/repos/timing/trace 信息。
[边界] 不创建/关闭数据库连接；不管理事务提交；不做业务流程编排；仅提供“依赖聚合 + 轻量可观测性字段”。
[上游关系] services/api/脚本/测试 fixture 创建 AsyncSession 后构造 PipelineContext 并传入 pipeline 入口。
[下游关系] pipelines/* 从 context 获取 repo 与 timing；最终将 timing_ms/provider_snapshot/trace_id 等写入 DB 或返回结果。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from sqlalchemy.ext.asyncio import AsyncSession

from uae_law_rag.backend.schemas.ids import UUIDStr, new_uuid
from uae_law_rag.backend.db.repo.ingest_repo import IngestRepo
from uae_law_rag.backend.db.repo.retrieval_repo import RetrievalRepo
from uae_law_rag.backend.db.repo.generation_repo import GenerationRepo
from uae_law_rag.backend.db.repo.evaluator_repo import EvaluatorRepo

from .timing import TimingCollector


@dataclass
class PipelineContext:
    """
    [职责] PipelineContext：为单次 pipeline 执行提供统一依赖与元数据（session/repos/timing/trace）。
    [边界] 不持有跨请求全局状态；不做 commit/rollback；不做缓存；只做“聚合与透传”。
    [上游关系] FastAPI deps/服务层创建 session 后调用 PipelineContext.from_session(...) 或直接构造。
    [下游关系] pipelines 从 ctx.repo 获取读写能力；从 ctx.timing 获取 timing_ms；从 ctx.trace/request_id 用于审计链路串联。
    """

    session: AsyncSession

    # repos (assembled per session)
    ingest_repo: IngestRepo
    retrieval_repo: RetrievalRepo
    generation_repo: GenerationRepo
    evaluator_repo: EvaluatorRepo

    # observability / audit
    trace_id: UUIDStr = field(
        default_factory=new_uuid
    )  # docstring: 单次链路追踪ID（可跨 retrieval/generation/eval 复用）
    request_id: UUIDStr = field(default_factory=new_uuid)  # docstring: 单次请求ID（可由上游注入覆盖）

    # timing collector
    timing: TimingCollector = field(default_factory=TimingCollector)

    # provider snapshots / misc context (optional)
    provider_snapshot: Dict[str, Any] = field(
        default_factory=dict
    )  # docstring: 可选：embed/llm/rerank provider 参数快照
    meta: Dict[str, Any] = field(default_factory=dict)  # docstring: 可选：额外上下文（debug flags / feature toggles）

    @classmethod
    def from_session(
        cls,
        session: AsyncSession,
        *,
        trace_id: Optional[str] = None,
        request_id: Optional[str] = None,
        provider_snapshot: Optional[Dict[str, Any]] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> "PipelineContext":
        """
        [职责] 从 AsyncSession 装配 PipelineContext（统一 repo 装配点）。
        [边界] 不验证 session 可用性；不触发 DB I/O；仅构造对象。
        [上游关系] services/api/测试 fixture 注入 session 后调用。
        [下游关系] pipelines 使用 ctx.* 访问 repos/timing/trace。
        """
        return cls(
            session=session,
            ingest_repo=IngestRepo(session),
            retrieval_repo=RetrievalRepo(session),
            generation_repo=GenerationRepo(session),
            evaluator_repo=EvaluatorRepo(session),
            trace_id=UUIDStr(trace_id) if trace_id else new_uuid(),
            request_id=UUIDStr(request_id) if request_id else new_uuid(),
            timing=TimingCollector(),
            provider_snapshot=provider_snapshot or {},
            meta=meta or {},
        )

    def timing_ms(self, *, include_total: bool = True, total_key: str = "total") -> Dict[str, float]:
        """
        [职责] 导出 timing_ms（JSON dict）供 DB 落库或返回结果使用。
        [边界] 不做字段映射；key 与 TimingCollector 保持一致。
        [上游关系] pipelines 在各阶段写入 timing 后调用。
        [下游关系] db.models.*.timing_ms / schemas.*.timing_ms。
        """
        return self.timing.to_dict(include_total=include_total, total_key=total_key)

    def with_provider(self, kind: str, snapshot: Dict[str, Any]) -> None:
        """
        [职责] 写入/覆盖某类 provider 快照（如 embed/llm/rerank）。
        [边界] 不做 schema 校验；仅作为审计快照透传；key 冲突由调用方负责。
        [上游关系] pipeline 构造 provider 参数或从 KB 配置读取。
        [下游关系] retrieval_record.provider_snapshot / generation_record.messages_snapshot/citations/meta 等写入。
        """
        k = str(kind).strip()
        if not k:
            return
        self.provider_snapshot[k] = snapshot

    def get_flag(self, key: str, default: Any = None) -> Any:
        """
        [职责] 从 meta 中读取 feature/debug flag。
        [边界] 不解释 flag 语义；由调用方约定。
        [上游关系] services 或测试向 ctx.meta 注入开关。
        [下游关系] pipelines 在编排时分支控制（如跳过 Milvus、跳过 rerank）。
        """
        return self.meta.get(key, default)
