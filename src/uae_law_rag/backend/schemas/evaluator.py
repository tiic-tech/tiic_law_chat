# src/uae_law_rag/backend/schemas/evaluator.py

"""
[职责] Evaluator 契约层：定义在线评估（retrieval/generation/citations）的配置与结果结构，用于质量门禁与可回放审计。
[边界] 不实现具体评估算法（不计算 F1/EM/Rouge 等）；仅表达“评估配置/规则/输出结果”的结构化合同。
[上游关系] retrieval pipeline 产出 RetrievalRecord + Hits；generation pipeline 产出 GenerationRecord + Citations。
[下游关系] pipelines/evaluator 将生成 EvaluationResult；services 可将其写入 DB 或挂载到 debug/审计输出。
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

from .ids import GenerationRecordId, MessageId, RetrievalRecordId, UUIDStr, new_uuid


EvaluationStatus = Literal["pass", "fail", "partial", "skipped"]  # docstring: 评估总状态（门禁/告警）
CheckStatus = Literal["pass", "fail", "warn", "skipped"]  # docstring: 单条规则检查状态


class EvaluatorConfig(BaseModel):
    """
    [职责] EvaluatorConfig：在线评估规则配置（门禁阈值与开关），可用于不同环境/KB 的策略差异。
    [边界] 不包含文件保存/数据集配置；不包含具体 metric 实现；仅用于控制 evaluator 行为。
    [上游关系] services 或 pipeline spec/配置文件注入；也可由 API debug 模式覆盖部分参数。
    [下游关系] pipelines/evaluator 根据此配置生成 checks 与 overall_scores，并决定 pass/fail。
    """

    model_config = ConfigDict(extra="allow")

    rule_version: str = Field(default="v0")  # docstring: 规则版本（用于回放与回归）

    # --- Retrieval quality gates ---
    retrieval_topk: int = Field(default=10, ge=1, le=5000)  # docstring: 用于 recall@k/precision@k 的评估 topk 基准
    retrieval_min_hits: int = Field(default=1, ge=0, le=5000)  # docstring: 最小命中条数门槛（0 表示不强制）
    require_vector_hits: bool = Field(default=False)  # docstring: 是否要求向量侧必须有命中（用于验证 Milvus 链路）
    require_keyword_hits: bool = Field(default=False)  # docstring: 是否要求关键词侧必须有命中（用于验证 FTS 链路）

    # --- Answer quality gates ---
    min_answer_chars: int = Field(default=20, ge=0, le=100000)  # docstring: 回答最小字符数（防止空回答/过短）
    require_structured: bool = Field(default=False)  # docstring: 是否要求结构化输出（JSON dict）存在
    structured_schema_name: Optional[str] = Field(default=None, max_length=200)  # docstring: 结构化 schema 标识（可选）

    # --- Citation / evidence gates ---
    require_citations: bool = Field(default=True)  # docstring: 是否强制必须提供 citations（法律助手建议 True）
    min_citations: int = Field(default=1, ge=0, le=5000)  # docstring: citations 最小数量门槛
    citation_coverage_threshold: float = Field(default=0.0, ge=0.0, le=1.0)  # docstring: 引用覆盖阈值（MVP 可先 0）

    # --- Misc ---
    enable_token_count: bool = Field(default=False)  # docstring: 是否统计 token（需要 tokenizer 支持；MVP 可关）
    tokenizer_name: Optional[str] = Field(default=None, max_length=200)  # docstring: tokenizer 标识（可选）


class EvaluationCheck(BaseModel):
    """
    [职责] EvaluationCheck：单条规则检查的结构化结果（可解释、可回放、可审计）。
    [边界] 不存储大体量原文；detail 仅保存必要摘要与数字证据。
    [上游关系] pipelines/evaluator 对 retrieval/generation 数据运行规则得到。
    [下游关系] services/前端 debug 面板展示；回归测试用于断言规则稳定性。
    """

    model_config = ConfigDict(extra="allow")

    name: str = Field(..., min_length=1, max_length=200)  # docstring: 检查项名称（例如 "require_citations"）
    status: CheckStatus = Field(...)  # docstring: 检查状态（pass/fail/warn/skipped）
    message: str = Field(default="")  # docstring: 人类可读说明（失败原因/建议）
    detail: Dict[str, Any] = Field(default_factory=dict)  # docstring: 结构化细节（阈值、实际值、样本等）


class EvaluationScores(BaseModel):
    """
    [职责] EvaluationScores：评估得分汇总（数值型指标容器）。
    [边界] 不限定具体指标集合；以 key-value 扩展方式支持未来添加 metric。
    [上游关系] evaluator 计算得到。
    [下游关系] 监控/报表/回归测试。
    """

    model_config = ConfigDict(extra="allow")

    overall: Dict[str, float] = Field(default_factory=dict)  # docstring: 总体分数（如 {"retrieval_recall@10": 1.0}）
    per_metric: Dict[str, Any] = Field(default_factory=dict)  # docstring: 指标细节（如分布/分项，必要时使用）


class EvaluationResult(BaseModel):
    """
    [职责] EvaluationResult：一次在线评估的最终输出（配置快照 + 检查列表 + 得分 + 状态）。
    [边界] 不替代 retrieval/generation record；仅引用其 ID 并提供评估视角的结论与证据。
    [上游关系] retrieval_record + hits 与 generation_record + citations 作为输入。
    [下游关系] 可写入 DB（若你后续增加 EvaluationRecordModel），或挂载到 ChatResponse.debug/audit。
    """

    model_config = ConfigDict(extra="forbid")

    id: UUIDStr = Field(default_factory=new_uuid)  # docstring: 本次评估结果ID（UUID str）
    status: EvaluationStatus = Field(default="pass")  # docstring: 总状态（门禁判断）

    message_id: MessageId = Field(...)  # docstring: 评估所对应的 message
    retrieval_record_id: RetrievalRecordId = Field(...)  # docstring: 评估所对应的 retrieval record
    generation_record_id: Optional[GenerationRecordId] = Field(
        default=None
    )  # docstring: 评估所对应的 generation record（可选）

    config: EvaluatorConfig = Field(default_factory=EvaluatorConfig)  # docstring: 评估配置快照（用于回放）
    checks: List[EvaluationCheck] = Field(default_factory=list)  # docstring: 规则检查列表（可解释证据）
    scores: EvaluationScores = Field(default_factory=EvaluationScores)  # docstring: 数值指标汇总

    error_message: Optional[str] = Field(default=None)  # docstring: evaluator 运行失败原因（若 status=partial/fail）
    meta: Dict[str, Any] = Field(default_factory=dict)  # docstring: 扩展元信息（trace_id/request_id/耗时摘要等）
