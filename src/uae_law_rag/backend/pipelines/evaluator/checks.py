# src/uae_law_rag/backend/pipelines/evaluator/checks.py

"""
[职责] evaluator checks：提供确定性的规则检查函数，输出 EvaluationCheck 供门禁裁决使用。
[边界] 不访问 DB/LLM；不进行 pipeline 编排；仅基于输入快照做可解释检查。
[上游关系] evaluator pipeline 传入 retrieval/generation 输入与 EvaluatorConfig。
[下游关系] pipeline 汇总 checks 形成 EvaluationResult.status 与审计记录。
"""

from __future__ import annotations

from typing import Any, Iterable, List, Mapping, Optional, Sequence, TypedDict

from uae_law_rag.backend.schemas.evaluator import CheckStatus, EvaluatorConfig, EvaluationCheck
from uae_law_rag.backend.schemas.generation import Citation, GenerationRecord
from uae_law_rag.backend.schemas.retrieval import RetrievalHit, RetrievalRecord


class EvaluatorInput(TypedDict, total=False):
    """
    [职责] EvaluatorInput：checks 层的最小输入合同（便于测试与编排层复用）。
    [边界] 仅描述必要字段；允许 pipeline 扩展更多键值。
    [上游关系] evaluator pipeline 组装并传入。
    [下游关系] checks 函数读取此结构输出 EvaluationCheck。
    """

    conversation_id: str
    message_id: str
    retrieval_record: RetrievalRecord
    retrieval_hits: Sequence[RetrievalHit]
    generation_record: GenerationRecord
    generation_output: Mapping[str, Any]
    config: EvaluatorConfig


__all__ = [
    "EvaluatorInput",
    "check_require_citations",
    "check_citation_coverage",
    "check_min_answer_length",
    "check_no_empty_answer",
    "check_min_retrieval_hits",
    "check_require_vector_hits",
    "check_require_keyword_hits",
    "check_require_structured",
]


def _coerce_str(value: Any) -> str:
    """
    [职责] 将任意 value 转为去空白字符串。
    [边界] 空值返回空字符串。
    [上游关系] 各类字段归一化调用。
    [下游关系] node_id/answer/source 等字段统一。
    """
    return str(value or "").strip()  # docstring: 字符串兜底


def _as_mapping(value: Any) -> Mapping[str, Any]:
    """
    [职责] 将对象降级为 Mapping（兼容 pydantic v2/v1）。
    [边界] 不抛出异常；无法转换则返回空 dict。
    [上游关系] generation_output/output_structured/score_details 读取。
    [下游关系] checks 内部字段解析。
    """
    if value is None:
        return {}  # docstring: 空值回退
    if isinstance(value, Mapping):
        return value  # docstring: mapping 直接返回
    if hasattr(value, "model_dump"):
        try:
            return value.model_dump()  # type: ignore[attr-defined]
        except Exception:
            return {}  # docstring: model_dump 失败兜底
    if hasattr(value, "dict"):
        try:
            return value.dict()  # type: ignore[call-arg]
        except Exception:
            return {}  # docstring: dict 失败兜底
    return {}  # docstring: 不可转换回退空 dict


def _read_field(obj: Any, key: str) -> Any:
    """
    [职责] 从对象或 mapping 安全读取字段值。
    [边界] 仅做浅读取，不抛异常。
    [上游关系] record/hit/citation 字段提取调用。
    [下游关系] checks 内部逻辑判断。
    """
    if obj is None:
        return None  # docstring: 空对象回退
    if isinstance(obj, Mapping):
        return obj.get(key)  # docstring: mapping 读取
    return getattr(obj, key, None)  # docstring: attribute 读取


def _normalize_config(input: Mapping[str, Any]) -> EvaluatorConfig:
    """
    [职责] 解析 EvaluatorConfig（缺省则使用默认值）。
    [边界] 不抛异常；配置非法时回退默认配置。
    [上游关系] checks 函数调用。
    [下游关系] 单条规则检查使用。
    """
    raw = input.get("config")  # docstring: 读取原始配置
    if isinstance(raw, EvaluatorConfig):
        return raw  # docstring: 已是 EvaluatorConfig
    if isinstance(raw, Mapping):
        try:
            return EvaluatorConfig(**dict(raw))  # docstring: mapping 构造配置
        except Exception:
            return EvaluatorConfig()  # docstring: 配置异常回退默认
    return EvaluatorConfig()  # docstring: 缺省配置


def _extract_generation_output(input: Mapping[str, Any]) -> Mapping[str, Any]:
    """
    [职责] 读取 generation_output（若缺失则回退空 dict）。
    [边界] 不做业务校验；仅做结构兜底。
    [上游关系] evaluator pipeline 传入 generation_output。
    [下游关系] answer/citations/output_structured 解析。
    """
    raw = input.get("generation_output")  # docstring: 原始 generation_output
    return _as_mapping(raw)  # docstring: 统一为 mapping


def _extract_output_structured(input: Mapping[str, Any]) -> Optional[Mapping[str, Any]]:
    """
    [职责] 读取结构化输出 output_structured。
    [边界] 仅做读取与类型兜底；不做 schema 校验。
    [上游关系] generation pipeline 输出或 record 快照。
    [下游关系] require_structured 检查使用。
    """
    output = _extract_generation_output(input)  # docstring: generation_output
    structured = output.get("output_structured")  # docstring: 直接读取 output_structured
    if structured is None:
        record = input.get("generation_record")  # docstring: fallback 到 generation_record
        structured = _read_field(record, "output_structured")  # docstring: record.output_structured
    if isinstance(structured, Mapping):
        return structured  # docstring: 合法 mapping 返回
    return None  # docstring: 非 mapping 视为缺失


def _extract_answer(input: Mapping[str, Any]) -> str:
    """
    [职责] 提取 answer 字符串（多来源兜底）。
    [边界] 不做内容清洗；仅负责空值/缺失回退。
    [上游关系] generation_output/output_structured/generation_record 提供答案。
    [下游关系] no_empty_answer/min_answer_length 检查使用。
    """
    output = _extract_generation_output(input)  # docstring: generation_output
    answer = output.get("answer")  # docstring: 优先使用 generation_output.answer
    if isinstance(answer, str) and answer.strip():
        return answer.strip()  # docstring: 返回 answer

    structured = _extract_output_structured(input)  # docstring: fallback output_structured
    if isinstance(structured, Mapping):
        struct_answer = structured.get("answer")  # docstring: 读取结构化 answer
        if isinstance(struct_answer, str) and struct_answer.strip():
            return struct_answer.strip()  # docstring: 使用结构化 answer

    record = input.get("generation_record")  # docstring: fallback 到 generation_record
    raw = _read_field(record, "output_raw")  # docstring: record.output_raw
    return _coerce_str(raw)  # docstring: raw 转为字符串


def _extract_citations_raw(input: Mapping[str, Any]) -> Any:
    """
    [职责] 提取原始 citations 输入（不做结构归一化）。
    [边界] 仅选择来源；不校验合法性。
    [上游关系] generation_output/output_structured/generation_record.citations。
    [下游关系] citation node_id 解析。
    """
    output = _extract_generation_output(input)  # docstring: generation_output
    citations = output.get("citations")  # docstring: 读取 citations 字段
    if citations is None:
        structured = _extract_output_structured(input)  # docstring: fallback output_structured
        if structured:
            citations = structured.get("citations")  # docstring: structured.citations
    if citations is None:
        record = input.get("generation_record")  # docstring: fallback generation_record
        payload = _read_field(record, "citations")  # docstring: record.citations
        if payload is not None:
            if isinstance(payload, Mapping):
                citations = payload.get("items") or payload.get("nodes")  # docstring: dict payload 读取 items/nodes
            else:
                items = _read_field(payload, "items")  # docstring: payload.items
                nodes = _read_field(payload, "nodes")  # docstring: payload.nodes
                citations = items or nodes  # docstring: pydantic payload 读取 items/nodes
    return citations  # docstring: 返回原始 citations


def _iter_citation_items(raw: Any) -> Iterable[Any]:
    """
    [职责] 将 citations 原始输入展开为可迭代条目。
    [边界] 不保证类型合法；仅做最小展开。
    [上游关系] _extract_citations_raw 调用。
    [下游关系] node_id 解析。
    """
    if raw is None:
        return []  # docstring: 空输入回退
    if isinstance(raw, (str, bytes, bytearray)):
        return [raw]  # docstring: 单字符串视为单条 citation
    if isinstance(raw, Mapping):
        return [raw]  # docstring: 单条 mapping citation
    if isinstance(raw, Sequence):
        return list(raw)  # docstring: 序列直接展开
    return []  # docstring: 其他类型视为空


def _citation_node_id(item: Any) -> str:
    """
    [职责] 从 citation 条目中提取 node_id。
    [边界] 不校验 node_id 是否存在于 hits。
    [上游关系] _extract_citation_node_ids 调用。
    [下游关系] citations 去重与覆盖率计算。
    """
    if isinstance(item, Citation):
        return _coerce_str(getattr(item, "node_id", ""))  # docstring: Citation 对象
    if isinstance(item, Mapping):
        return _coerce_str(item.get("node_id") or item.get("id") or item.get("nodeId"))  # docstring: mapping citation
    if hasattr(item, "node_id"):
        return _coerce_str(getattr(item, "node_id", ""))  # docstring: attribute citation
    if isinstance(item, str):
        return _coerce_str(item)  # docstring: 字符串视为 node_id
    return ""  # docstring: 不支持类型


def _extract_citation_node_ids(input: Mapping[str, Any]) -> List[str]:
    """
    [职责] 提取 citations 的 node_id 列表（去重）。
    [边界] 不做 DB 校验；仅处理结构与去重。
    [上游关系] generation_output/citations 提供原始数据。
    [下游关系] require_citations/citation_coverage 使用。
    """
    raw = _extract_citations_raw(input)  # docstring: 原始 citations
    node_ids: List[str] = []  # docstring: 输出 node_id 列表
    seen: set[str] = set()  # docstring: 去重集合
    for item in _iter_citation_items(raw):
        node_id = _citation_node_id(item)  # docstring: 提取 node_id
        if not node_id or node_id in seen:
            continue  # docstring: 空/重复跳过
        seen.add(node_id)  # docstring: 记录已见 node_id
        node_ids.append(node_id)  # docstring: 追加 node_id
    return node_ids  # docstring: 返回去重结果


def _extract_retrieval_hits(input: Mapping[str, Any]) -> List[Any]:
    """
    [职责] 读取 retrieval hits（兼容 bundle 兜底）。
    [边界] 不校验 hit 结构；仅做列表归一化。
    [上游关系] retrieval pipeline 或 evaluator 编排传入。
    [下游关系] citations coverage / hit count / source 判定。
    """
    hits = input.get("retrieval_hits")  # docstring: 优先 retrieval_hits
    if hits is None:
        bundle = input.get("retrieval_bundle")  # docstring: fallback retrieval_bundle
        hits = _read_field(bundle, "hits")  # docstring: bundle.hits
    if hits is None:
        return []  # docstring: 缺失回退为空
    if isinstance(hits, Sequence) and not isinstance(hits, (str, bytes, bytearray)):
        return list(hits)  # docstring: 序列转换为 list
    return []  # docstring: 非序列回退为空


def _extract_hit_node_ids(hits: Sequence[Any]) -> List[str]:
    """
    [职责] 提取 hits 中的 node_id 列表（去重）。
    [边界] 不校验 node_id 真实性。
    [上游关系] retrieval hits 输入。
    [下游关系] citation_coverage 使用。
    """
    node_ids: List[str] = []  # docstring: 输出 node_id 列表
    seen: set[str] = set()  # docstring: 去重集合
    for hit in hits:
        node_id = _coerce_str(_read_field(hit, "node_id"))  # docstring: 读取 hit.node_id
        if not node_id or node_id in seen:
            continue  # docstring: 空/重复跳过
        seen.add(node_id)  # docstring: 记录已见 node_id
        node_ids.append(node_id)  # docstring: 追加 node_id
    return node_ids  # docstring: 返回去重结果


def _hit_source(hit: Any) -> str:
    """
    [职责] 提取 hit 的 source 字段（lowercase）。
    [边界] 缺失时返回空字符串。
    [上游关系] require_vector/keyword_hits 调用。
    [下游关系] source 判断。
    """
    return _coerce_str(_read_field(hit, "source")).lower()  # docstring: 统一小写 source


def _hit_score_details(hit: Any) -> Mapping[str, Any]:
    """
    [职责] 提取 hit.score_details 并转为 mapping。
    [边界] 缺失返回空 dict；不做深层校验。
    [上游关系] require_vector/keyword_hits 调用。
    [下游关系] source 辅助判断。
    """
    return _as_mapping(_read_field(hit, "score_details"))  # docstring: score_details 兜底


def _hit_has_signal(hit: Any, signal: str) -> bool:
    """
    [职责] 判断 hit 是否包含指定信号（keyword/vector）。
    [边界] 仅依赖 source/score_details；不做 DB 校验。
    [上游关系] require_vector_hits/require_keyword_hits。
    [下游关系] 返回命中布尔值。
    """
    if _hit_source(hit) == signal:
        return True  # docstring: source 直接命中
    details = _hit_score_details(hit)  # docstring: 读取 score_details
    if not details:
        return False  # docstring: 无分数细节直接否定
    if signal in details and details.get(signal):
        return True  # docstring: 细节包含 signal
    if f"{signal}_score" in details:
        return True  # docstring: 命中 signal_score
    nested = details.get(signal)  # docstring: 读取 nested signal
    if isinstance(nested, Mapping) and nested:
        return True  # docstring: nested signal 细节
    return False  # docstring: 未发现信号


def _build_check(
    *,
    name: str,
    status: CheckStatus,
    message: str,
    detail: Optional[Mapping[str, Any]] = None,
) -> EvaluationCheck:
    """
    [职责] 统一构造 EvaluationCheck（确保结构稳定）。
    [边界] 不做深层 JSON 安全转换；调用侧应提供可序列化 detail。
    [上游关系] 各 check 函数调用。
    [下游关系] pipeline 汇总 checks。
    """
    return EvaluationCheck(
        name=str(name),  # docstring: 检查名称
        status=status,  # docstring: 检查状态
        message=str(message or ""),  # docstring: 人类可读消息
        detail=dict(detail or {}),  # docstring: 结构化细节
    )


def check_require_citations(*, input: EvaluatorInput) -> EvaluationCheck:
    """
    [职责] require_citations：要求 citations 不为空且数量达标。
    [边界] 不校验 citations 是否真实存在于 DB；仅计数与结构判断。
    [上游关系] generation_output.citations 输入。
    [下游关系] evaluator pipeline 汇总门禁状态。
    """
    cfg = _normalize_config(input)  # docstring: 解析配置
    if not cfg.require_citations:
        return _build_check(
            name="require_citations",
            status="skipped",
            message="require_citations disabled",
            detail={"required": False},
        )  # docstring: 配置关闭则跳过

    citations = _extract_citation_node_ids(input)  # docstring: 提取 citations
    required = max(int(cfg.min_citations), 1)  # docstring: 最小 citations 数
    ok = len(citations) >= required  # docstring: 是否达标
    status: CheckStatus = "pass" if ok else "fail"  # docstring: 状态判定
    message = "citations present" if ok else "citations below minimum"  # docstring: 结果消息
    return _build_check(
        name="require_citations",
        status=status,
        message=message,
        detail={
            "required": True,
            "min_citations": required,
            "citations_count": len(citations),
            "citations": citations,
        },
    )  # docstring: 构造检查结果


def check_citation_coverage(*, input: EvaluatorInput) -> EvaluationCheck:
    """
    [职责] citation_coverage：检查 citations 是否覆盖于 retrieval hits。
    [边界] 不做 DB 回查；仅判断 node_id 关系。
    [上游关系] retrieval_hits + generation_output.citations。
    [下游关系] evaluator pipeline 用于质量裁决。
    """
    cfg = _normalize_config(input)  # docstring: 解析配置
    citations = _extract_citation_node_ids(input)  # docstring: citations node_id
    if not citations:
        return _build_check(
            name="citation_coverage",
            status="skipped",
            message="no citations for coverage",
            detail={"coverage": 0.0, "total_citations": 0},
        )  # docstring: 无 citations 跳过

    hits = _extract_retrieval_hits(input)  # docstring: retrieval hits
    hit_node_ids = set(_extract_hit_node_ids(hits))  # docstring: hits node_id set
    matched = [cid for cid in citations if cid in hit_node_ids]  # docstring: 匹配 citations
    coverage = float(len(matched)) / float(len(citations))  # docstring: coverage 计算
    threshold = float(cfg.citation_coverage_threshold)  # docstring: 覆盖率阈值
    ok = coverage >= threshold  # docstring: 是否达标
    status: CheckStatus = "pass" if ok else "fail"  # docstring: 状态判定
    message = "citation coverage ok" if ok else "citation coverage below threshold"  # docstring: 消息
    missing = [cid for cid in citations if cid not in hit_node_ids]  # docstring: 缺失 citations
    return _build_check(
        name="citation_coverage",
        status=status,
        message=message,
        detail={
            "coverage": coverage,
            "threshold": threshold,
            "matched": len(matched),
            "total_citations": len(citations),
            "missing_citations": missing,
        },
    )  # docstring: 构造检查结果


def check_min_answer_length(*, input: EvaluatorInput) -> EvaluationCheck:
    """
    [职责] min_answer_length：检查 answer 是否达到最小长度。
    [边界] 不对文本做语义判断；仅基于字符长度。
    [上游关系] generation_output.answer。
    [下游关系] evaluator pipeline 汇总质量状态。
    """
    cfg = _normalize_config(input)  # docstring: 解析配置
    min_chars = int(cfg.min_answer_chars)  # docstring: 最小字符数
    if min_chars <= 0:
        return _build_check(
            name="min_answer_length",
            status="skipped",
            message="min_answer_chars disabled",
            detail={"min_answer_chars": min_chars},
        )  # docstring: 未启用则跳过

    answer = _extract_answer(input)  # docstring: 提取 answer
    answer_len = len(answer)  # docstring: answer 长度
    if answer_len == 0:
        status: CheckStatus = "fail"  # docstring: 空答案直接失败
        message = "answer is empty"  # docstring: 空答案消息
    elif answer_len < min_chars:
        status = "warn"  # docstring: 过短答案给出警告
        message = "answer shorter than minimum"  # docstring: 过短消息
    else:
        status = "pass"  # docstring: 满足长度要求
        message = "answer length ok"  # docstring: 成功消息
    return _build_check(
        name="min_answer_length",
        status=status,
        message=message,
        detail={"min_answer_chars": min_chars, "answer_length": answer_len},
    )  # docstring: 构造检查结果


def check_no_empty_answer(*, input: EvaluatorInput) -> EvaluationCheck:
    """
    [职责] no_empty_answer：确保 answer 不为空。
    [边界] 仅判断空字符串；不评估语义质量。
    [上游关系] generation_output.answer。
    [下游关系] evaluator pipeline 汇总门禁状态。
    """
    answer = _extract_answer(input)  # docstring: 提取 answer
    ok = bool(answer)  # docstring: 判定 answer 是否为空
    status: CheckStatus = "pass" if ok else "fail"  # docstring: 状态判定
    message = "answer present" if ok else "answer empty"  # docstring: 消息
    return _build_check(
        name="no_empty_answer",
        status=status,
        message=message,
        detail={"answer_length": len(answer)},
    )  # docstring: 构造检查结果


def check_min_retrieval_hits(*, input: EvaluatorInput) -> EvaluationCheck:
    """
    [职责] min_retrieval_hits：确保 retrieval hits 数量达到最小门槛。
    [边界] 不校验 hit 质量；仅统计数量。
    [上游关系] retrieval_hits 输入。
    [下游关系] evaluator pipeline 汇总门禁状态。
    """
    cfg = _normalize_config(input)  # docstring: 解析配置
    min_hits = int(cfg.retrieval_min_hits)  # docstring: 最小命中数
    if min_hits <= 0:
        return _build_check(
            name="min_retrieval_hits",
            status="skipped",
            message="retrieval_min_hits disabled",
            detail={"retrieval_min_hits": min_hits},
        )  # docstring: 未启用则跳过

    hits = _extract_retrieval_hits(input)  # docstring: retrieval hits
    hit_count = len(hits)  # docstring: 命中数量
    ok = hit_count >= min_hits  # docstring: 是否达标
    status: CheckStatus = "pass" if ok else "fail"  # docstring: 状态判定
    message = "retrieval hits ok" if ok else "retrieval hits below minimum"  # docstring: 消息
    return _build_check(
        name="min_retrieval_hits",
        status=status,
        message=message,
        detail={"retrieval_min_hits": min_hits, "retrieval_hits": hit_count},
    )  # docstring: 构造检查结果


def check_require_vector_hits(*, input: EvaluatorInput) -> EvaluationCheck:
    """
    [职责] require_vector_hits：要求 hits 中存在 vector 信号。
    [边界] 仅基于 source/score_details 判断；不访问 Milvus/DB。
    [上游关系] retrieval_hits 输入。
    [下游关系] evaluator pipeline 汇总门禁状态。
    """
    cfg = _normalize_config(input)  # docstring: 解析配置
    if not cfg.require_vector_hits:
        return _build_check(
            name="require_vector_hits",
            status="skipped",
            message="require_vector_hits disabled",
            detail={"required": False},
        )  # docstring: 配置关闭则跳过

    hits = _extract_retrieval_hits(input)  # docstring: retrieval hits
    vector_hits = sum(1 for hit in hits if _hit_has_signal(hit, "vector"))  # docstring: 统计 vector 命中数
    ok = vector_hits > 0  # docstring: 是否存在 vector 信号
    status: CheckStatus = "pass" if ok else "fail"  # docstring: 状态判定
    message = "vector hits present" if ok else "vector hits missing"  # docstring: 消息
    return _build_check(
        name="require_vector_hits",
        status=status,
        message=message,
        detail={"vector_hits": vector_hits},
    )  # docstring: 构造检查结果


def check_require_keyword_hits(*, input: EvaluatorInput) -> EvaluationCheck:
    """
    [职责] require_keyword_hits：要求 hits 中存在 keyword 信号。
    [边界] 仅基于 source/score_details 判断；不访问 FTS/DB。
    [上游关系] retrieval_hits 输入。
    [下游关系] evaluator pipeline 汇总门禁状态。
    """
    cfg = _normalize_config(input)  # docstring: 解析配置
    if not cfg.require_keyword_hits:
        return _build_check(
            name="require_keyword_hits",
            status="skipped",
            message="require_keyword_hits disabled",
            detail={"required": False},
        )  # docstring: 配置关闭则跳过

    hits = _extract_retrieval_hits(input)  # docstring: retrieval hits
    keyword_hits = sum(1 for hit in hits if _hit_has_signal(hit, "keyword"))  # docstring: 统计 keyword 命中数
    ok = keyword_hits > 0  # docstring: 是否存在 keyword 信号
    status: CheckStatus = "pass" if ok else "fail"  # docstring: 状态判定
    message = "keyword hits present" if ok else "keyword hits missing"  # docstring: 消息
    return _build_check(
        name="require_keyword_hits",
        status=status,
        message=message,
        detail={"keyword_hits": keyword_hits},
    )  # docstring: 构造检查结果


def check_require_structured(*, input: EvaluatorInput) -> EvaluationCheck:
    """
    [职责] require_structured：要求 generation 输出包含结构化 payload。
    [边界] 不校验 payload 内容完整性；仅检查存在性/可选 schema 标识。
    [上游关系] generation_output.output_structured 或 generation_record.output_structured。
    [下游关系] evaluator pipeline 汇总门禁状态。
    """
    cfg = _normalize_config(input)  # docstring: 解析配置
    if not cfg.require_structured:
        return _build_check(
            name="require_structured",
            status="skipped",
            message="require_structured disabled",
            detail={"required": False},
        )  # docstring: 配置关闭则跳过

    structured = _extract_output_structured(input)  # docstring: 读取结构化输出
    if not structured:
        return _build_check(
            name="require_structured",
            status="fail",
            message="structured output missing",
            detail={"required": True, "structured_present": False},
        )  # docstring: 结构化输出缺失

    expected_schema = _coerce_str(cfg.structured_schema_name)  # docstring: 期望 schema 名称
    actual_schema = ""  # docstring: 实际 schema 占位
    for key in ("schema", "schema_name", "name"):
        actual_schema = _coerce_str(structured.get(key))  # docstring: 提取 schema 字段
        if actual_schema:
            break  # docstring: 找到 schema 即停止

    if expected_schema and actual_schema and expected_schema != actual_schema:
        status: CheckStatus = "fail"  # docstring: schema 不匹配
        message = "structured schema mismatch"  # docstring: 不匹配消息
    elif expected_schema and not actual_schema:
        status = "warn"  # docstring: 缺少 schema 标识给出警告
        message = "structured schema missing"  # docstring: 缺失 schema 消息
    else:
        status = "pass"  # docstring: 结构化输出满足
        message = "structured output present"  # docstring: 成功消息

    return _build_check(
        name="require_structured",
        status=status,
        message=message,
        detail={
            "required": True,
            "structured_present": True,
            "expected_schema": expected_schema or None,
            "actual_schema": actual_schema or None,
        },
    )  # docstring: 构造检查结果
