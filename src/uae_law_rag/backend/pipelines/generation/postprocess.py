# src/uae_law_rag/backend/pipelines/generation/postprocess.py

"""
[职责] generation postprocess：解析 LLM 输出，校验结构与证据引用，并补全 citation 定位信息。
[边界] 不调用 LLM；不访问 DB；不执行落库；仅处理输出结构与引用对齐。
[上游关系] generator 返回 raw_text；retrieval 提供 hits 作为证据集合。
[下游关系] persist 写入 output_structured/citations；evaluator 消费 citations 做审计。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from uuid import UUID
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, cast

from uae_law_rag.backend.schemas.generation import Citation
from uae_law_rag.backend.schemas.ids import NodeId, UUIDStr
from uae_law_rag.backend.schemas.retrieval import RetrievalHit


__all__ = ["postprocess_generation"]

_STATUS_ORDER = {"success": 0, "partial": 1, "blocked": 2, "failed": 3}  # docstring: status 严重程度顺序
_DEFAULT_MAX_QUOTE_CHARS = 240  # docstring: citation.quote 最大长度


@dataclass(frozen=True)
class _PostprocessConfig:
    """Normalized postprocess config."""  # docstring: 内部使用的配置快照

    strict_json: bool
    require_citations: bool
    parse_error_status: str
    missing_citations_status: str
    invalid_citations_status: str
    max_citations: Optional[int]
    max_quote_chars: int


def _normalize_page_value(val: Any) -> Optional[int]:
    """Normalize page to 1-based positive int; 0/negative/invalid -> None."""  # docstring: 对外证据页码防御性归一化
    if val is None:
        return None
    try:
        n = int(val)
    except (TypeError, ValueError):
        return None
    return n if n > 0 else None


def _normalize_config(config: Optional[Mapping[str, Any]]) -> _PostprocessConfig:
    """
    [职责] 归一化 postprocess 配置。
    [边界] 仅处理缺省与类型转换；不做业务策略校验。
    [上游关系] postprocess_generation 调用。
    [下游关系] 解析/校验/对齐逻辑使用。
    """
    cfg = dict(config or {})  # docstring: 复制配置

    def _as_bool(key: str, default: bool) -> bool:
        val = cfg.get(key, default)
        return bool(default if val is None else val)  # docstring: bool 兜底

    def _as_int(key: str, default: int) -> int:
        val = cfg.get(key, default)
        try:
            return int(default if val is None else val)  # docstring: int 兜底
        except (TypeError, ValueError):
            return int(default)  # docstring: 异常回退

    def _as_opt_int(key: str) -> Optional[int]:
        val = cfg.get(key)
        if val is None:
            return None
        try:
            num = int(val)
            return num if num > 0 else None  # docstring: 非正数回退 None
        except (TypeError, ValueError):
            return None

    def _as_status(key: str, default: str) -> str:
        raw = str(cfg.get(key, default) or default).strip().lower()  # docstring: status 归一化
        if raw in _STATUS_ORDER:
            return raw
        return default  # docstring: 未知 status 回退

    return _PostprocessConfig(
        strict_json=_as_bool("strict_json", True),
        require_citations=_as_bool("require_citations", True),
        parse_error_status=_as_status("parse_error_status", "partial"),
        missing_citations_status=_as_status("missing_citations_status", "blocked"),
        invalid_citations_status=_as_status("invalid_citations_status", "blocked"),
        max_citations=_as_opt_int("max_citations"),
        max_quote_chars=_as_int("max_quote_chars", _DEFAULT_MAX_QUOTE_CHARS),
    )


def _merge_status(current: str, new_status: str) -> str:
    """
    [职责] 合并 status（选择更严重的状态）。
    [边界] 未知 status 视为 current。
    [上游关系] postprocess_generation 调用。
    [下游关系] 输出 PostprocessResult.status。
    """
    if new_status not in _STATUS_ORDER:
        return current  # docstring: 未知 status 不更新
    if current not in _STATUS_ORDER:
        return new_status  # docstring: current 异常时回退 new_status
    return new_status if _STATUS_ORDER[new_status] > _STATUS_ORDER[current] else current  # docstring: 选择更严重


def _coerce_int(value: Any) -> Optional[int]:
    """
    [职责] 将 value 转为 int（失败返回 None）。
    [边界] 仅处理常见数值与数字字符串。
    [上游关系] citation/locator 解析调用。
    [下游关系] rank/page/offset 归一化。
    """
    if value is None:
        return None  # docstring: 空值直接返回
    if isinstance(value, int):
        return value  # docstring: int 直接返回
    if isinstance(value, float):
        return int(value)  # docstring: float 转 int
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())  # docstring: 数字字符串转 int
    return None  # docstring: 不可解析返回 None


def _coerce_str(value: Any) -> Optional[str]:
    """
    [职责] 将 value 转为非空字符串（空值返回 None）。
    [边界] 仅做字符串归一化。
    [上游关系] citation/locator 解析调用。
    [下游关系] node_id/article_id/section_path 等字段。
    """
    if value is None:
        return None  # docstring: 空值直接返回
    text = str(value).strip()
    return text if text else None  # docstring: 空字符串回退 None


def _truncate_text(text: str, *, max_chars: int) -> str:
    """
    [职责] 截断文本长度（用于 quote）。
    [边界] 仅按字符截断；不做语义裁剪。
    [上游关系] citation 构造调用。
    [下游关系] Citation.quote。
    """
    limit = int(max_chars)  # docstring: 最大长度兜底
    if limit <= 0:
        return ""  # docstring: 非法 max_chars 返回空
    if len(text) <= limit:
        return text  # docstring: 长度足够直接返回
    return text[: max(limit - 3, 0)].rstrip() + "..."  # docstring: 超长截断


def _read_hit_field(hit: Any, key: str, default: Any = None) -> Any:
    """
    [职责] 从 hit 对象读取字段（兼容 BaseModel/Mapping）。
    [边界] 不做类型校验；仅做字段读取。
    [上游关系] 证据对齐逻辑调用。
    [下游关系] locator/quote 补全。
    """
    if hasattr(hit, key):
        return getattr(hit, key)  # docstring: 优先读取属性
    if isinstance(hit, Mapping):
        return hit.get(key, default)  # docstring: mapping 兜底读取
    return default  # docstring: 无字段时返回默认值


def _build_hit_index(hits: Sequence[RetrievalHit]) -> Dict[str, RetrievalHit]:
    """
    [职责] 构建 node_id -> RetrievalHit 映射。
    [边界] 重复 node_id 取 rank 更小者。
    [上游关系] postprocess_generation 调用。
    [下游关系] citation 对齐使用。
    """
    hit_map: Dict[str, RetrievalHit] = {}  # docstring: 命中映射容器
    for hit in hits or []:
        node_id = _coerce_str(_read_hit_field(hit, "node_id"))  # docstring: 读取 node_id
        if not node_id:
            continue
        current = hit_map.get(node_id)
        if current is None:
            hit_map[node_id] = hit  # docstring: 首次命中直接写入
            continue
        prev_rank = _coerce_int(_read_hit_field(current, "rank"))  # docstring: 当前 rank
        next_rank = _coerce_int(_read_hit_field(hit, "rank"))  # docstring: 新 rank
        if prev_rank is None or (next_rank is not None and next_rank < prev_rank):
            hit_map[node_id] = hit  # docstring: 选择更高排名
    return hit_map


def _parse_json(raw_text: str, *, strict: bool) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    [职责] 解析 raw_text 为 JSON dict。
    [边界] strict=True 时要求全文是 JSON；不尝试修复格式。
    [上游关系] postprocess_generation 调用。
    [下游关系] answer/citations 解析。
    """
    text = str(raw_text or "").strip()  # docstring: raw_text 归一化
    if not text:
        return None, "raw_text is empty"  # docstring: 空输出错误
    if strict:
        try:
            data = json.loads(text)  # docstring: 严格 JSON 解析
        except json.JSONDecodeError as exc:
            return None, f"json parse error: {exc}"  # docstring: 解析失败
        if not isinstance(data, dict):
            return None, "json root must be an object"  # docstring: root 结构校验
        return dict(data), None  # docstring: 返回解析结果

    start = text.find("{")  # docstring: 尝试提取 JSON 对象
    end = text.rfind("}")  # docstring: 末尾 JSON 结束位置
    if start < 0 or end < 0 or end <= start:
        return None, "json object not found"  # docstring: 未找到 JSON
    try:
        data = json.loads(text[start : end + 1])  # docstring: 非严格 JSON 解析
    except json.JSONDecodeError as exc:
        return None, f"json parse error: {exc}"  # docstring: 解析失败
    if not isinstance(data, dict):
        return None, "json root must be an object"  # docstring: root 结构校验
    return dict(data), None  # docstring: 返回解析结果


def _extract_answer(payload: Mapping[str, Any]) -> Tuple[str, Optional[str]]:
    """
    [职责] 解析 answer 字段。
    [边界] 仅接受字符串；空值视为错误。
    [上游关系] postprocess_generation 调用。
    [下游关系] PostprocessResult.answer。
    """
    if "answer" not in payload:
        return "", "missing answer"  # docstring: answer 缺失
    raw = payload.get("answer")
    if not isinstance(raw, str):
        return "", "answer must be string"  # docstring: answer 类型错误
    answer = raw.strip()
    if not answer:
        return "", "answer is empty"  # docstring: answer 为空
    return answer, None


def _parse_citation_item(item: Any) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    [职责] 解析单条 citation 结构。
    [边界] 仅接受 node_id；未知字段忽略。
    [上游关系] postprocess_generation 调用。
    [下游关系] citation 对齐逻辑。
    """
    if isinstance(item, str):
        node_id = _coerce_str(item)  # docstring: 字符串视为 node_id
        if not node_id:
            return None, "citation node_id empty"
        return {"node_id": node_id}, None  # docstring: 返回最小 citation
    if not isinstance(item, Mapping):
        return None, "citation must be object or string"  # docstring: 类型错误

    node_id = (
        _coerce_str(item.get("node_id")) or _coerce_str(item.get("nodeId")) or _coerce_str(item.get("id"))
    )  # docstring: 解析 node_id
    if not node_id:
        return None, "citation node_id missing"  # docstring: node_id 缺失

    rank = _coerce_int(item.get("rank"))  # docstring: rank 解析
    quote = _coerce_str(item.get("quote")) or _coerce_str(item.get("excerpt"))  # docstring: quote 解析
    locator_raw = item.get("locator")  # docstring: locator 原始值
    locator = locator_raw if isinstance(locator_raw, Mapping) else {}  # docstring: locator 解析

    return {"node_id": node_id, "rank": rank, "quote": quote, "locator": dict(locator)}, None


def _build_locator_from_hit(hit: RetrievalHit) -> Dict[str, Any]:
    """
    [职责] 从 hit 构建 locator 快照。
    [边界] 仅使用 hit 的可用字段；缺失字段返回 None。
    [上游关系] citation 对齐逻辑调用。
    [下游关系] Citation.locator。
    """
    locator = {
        "page": _coerce_int(_read_hit_field(hit, "page")),
        "start_offset": _coerce_int(_read_hit_field(hit, "start_offset")),
        "end_offset": _coerce_int(_read_hit_field(hit, "end_offset")),
        "article_id": _coerce_str(_read_hit_field(hit, "article_id") or _read_hit_field(hit, "article")),
        "section_path": _coerce_str(_read_hit_field(hit, "section_path") or _read_hit_field(hit, "section")),
        "source": _coerce_str(_read_hit_field(hit, "source")),
    }  # docstring: locator 快照
    return locator


def _merge_locator(base: Optional[Mapping[str, Any]], fallback: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """
    [职责] 合并 locator（base 优先，缺失字段由 fallback 补全）。
    [边界] 仅在 base 缺失/空值时覆盖。
    [上游关系] citation 对齐逻辑调用。
    [下游关系] Citation.locator。
    """
    out = dict(base or {})  # docstring: base locator 复制
    for key, val in (fallback or {}).items():
        if key not in out or out.get(key) in (None, "", []):
            out[key] = val  # docstring: 使用 fallback 补全
    return out


def _is_uuid(value: str) -> bool:
    """
    [职责] 判断字符串是否为 UUID（用于 node_id 运行时校验）。
    [边界] 仅做格式校验；不校验其是否存在于 hits。
    """
    try:
        UUID(str(value))
        return True
    except Exception:
        return False


def _build_citation(
    *,
    parsed: Mapping[str, Any],
    hit: RetrievalHit,
    rank_fallback: int,
    max_quote_chars: int,
) -> Citation:
    """
    [职责] 构造 Citation（补全 rank/locator/quote）。
    [边界] 不校验 node_id 是否存在于 hits（由上游保证）。
    [上游关系] postprocess_generation 调用。
    [下游关系] PostprocessResult.citations。
    """
    node_id_raw = _coerce_str(parsed.get("node_id")) or ""  # docstring: node_id 原始值
    # docstring: 运行时校验 UUID，避免 Citation 构造阶段抛异常导致 postprocess 崩溃
    if not _is_uuid(node_id_raw):
        raise ValueError("node_id is not a valid UUID")
    node_id = cast(NodeId, UUIDStr(node_id_raw))  # docstring: node_id 类型对齐
    rank = _coerce_int(parsed.get("rank"))  # docstring: rank 解析
    if rank is None:
        rank = _coerce_int(_read_hit_field(hit, "rank"))  # docstring: rank 回退 hit.rank
    if rank is None:
        rank = int(rank_fallback)  # docstring: rank 回退索引

    quote = _coerce_str(parsed.get("quote"))  # docstring: quote 解析
    if not quote:
        quote = _coerce_str(_read_hit_field(hit, "excerpt"))  # docstring: quote 回退 excerpt
    if quote:
        quote = _truncate_text(quote, max_chars=max_quote_chars)  # docstring: quote 截断

    parsed_locator_raw = parsed.get("locator")  # docstring: citation locator 原始值
    parsed_locator = parsed_locator_raw if isinstance(parsed_locator_raw, Mapping) else {}  # docstring: locator 解析
    locator = _merge_locator(parsed_locator, _build_locator_from_hit(hit))  # docstring: locator 合并

    return Citation(
        node_id=node_id,
        rank=rank,
        quote=quote or "",
        locator=locator,
    )  # docstring: 构造 Citation


def postprocess_generation(
    *,
    raw_text: str,
    hits: Sequence[RetrievalHit],
    config: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """
    [职责] 解析 LLM 输出并生成结构化 answer/citations 结果。
    [边界] 不调用 LLM；不访问 DB；不落库。
    [上游关系] generator 产出 raw_text；retrieval pipeline 提供 hits。
    [下游关系] persist/evaluator 使用结构化结果与 citations。
    """
    cfg = _normalize_config(config)  # docstring: 归一化配置
    status = "success"  # docstring: 初始状态
    errors: List[str] = []  # docstring: 错误列表

    payload, parse_error = _parse_json(raw_text, strict=cfg.strict_json)  # docstring: 解析 JSON
    if parse_error:
        status = _merge_status(status, cfg.parse_error_status)  # docstring: 解析失败状态
        errors.append(parse_error)  # docstring: 统一错误收集
        return {
            "answer": "",
            "citations": [],
            "output_structured": None,
            "status": status,
            "error_message": "; ".join(errors),
        }  # docstring: JSON 解析失败直接返回

    answer, answer_error = _extract_answer(payload or {})  # docstring: 提取 answer
    if answer_error:
        status = _merge_status(status, "failed")  # docstring: answer 错误设为 failed
        errors.append(answer_error)  # docstring: 记录错误

    raw_citations = (payload or {}).get("citations")  # docstring: 读取 citations
    missing_citations = raw_citations is None  # docstring: 标记 citations 缺失
    if raw_citations is None:
        raw_citations = []  # docstring: 缺失则回退为空列表
    if not isinstance(raw_citations, list):
        status = _merge_status(status, "failed")  # docstring: citations 类型错误
        errors.append("citations must be list")  # docstring: 记录错误
        raw_citations = []  # docstring: 非法类型回退为空

    hit_map = _build_hit_index(hits)  # docstring: 命中索引
    citations: List[Citation] = []  # docstring: 输出 citations
    invalid_count = 0  # docstring: 无效引用计数
    missing_count = 0  # docstring: 未命中引用计数
    seen: set[str] = set()  # docstring: 去重缓存

    for idx, item in enumerate(raw_citations, start=1):
        parsed, err = _parse_citation_item(item)  # docstring: 解析 citation
        if err or not parsed:
            invalid_count += 1  # docstring: 记录无效 citation
            continue
        node_id = str(parsed.get("node_id") or "")  # docstring: 读取 node_id
        if not node_id or node_id in seen:
            invalid_count += 1  # docstring: 空/重复 citation 记为无效
            continue
        hit = hit_map.get(node_id)
        if not hit:
            missing_count += 1  # docstring: 引用不在 hits 中
            continue
        try:
            citation = _build_citation(
                parsed=parsed,
                hit=hit,
                rank_fallback=idx,
                max_quote_chars=cfg.max_quote_chars,
            )  # docstring: 构造对齐 Citation
        except Exception:
            invalid_count += 1  # docstring: citation 构造失败视为无效引用
            continue
        citations.append(citation)  # docstring: 收集 citation
        seen.add(node_id)  # docstring: 记录已处理 node_id

    if cfg.max_citations is not None and len(citations) > cfg.max_citations:
        citations = citations[: cfg.max_citations]  # docstring: 限制 citation 数量

    if missing_citations and cfg.require_citations:
        status = _merge_status(status, cfg.missing_citations_status)  # docstring: citations 缺失策略
        errors.append("citations missing")  # docstring: 记录错误
    if cfg.require_citations and not citations:
        status = _merge_status(status, cfg.missing_citations_status)  # docstring: citations 为空策略
        errors.append("no valid citations")  # docstring: 记录错误

    if invalid_count > 0 or missing_count > 0:
        status = _merge_status(status, cfg.invalid_citations_status)  # docstring: 无效/缺失引用处理
        if invalid_count > 0:
            errors.append(f"invalid citations: {invalid_count}")  # docstring: 记录无效计数
        if missing_count > 0:
            errors.append(f"missing citations: {missing_count}")  # docstring: 记录缺失计数

    # docstring: 兼容 pydantic v2 / 其他实现
    citation_payload: List[Dict[str, Any]] = []
    for c in citations:
        if hasattr(c, "model_dump"):
            citation_payload.append(c.model_dump())  # type: ignore[attr-defined]
        elif hasattr(c, "dict"):
            citation_payload.append(c.dict())  # type: ignore[call-arg]
        else:
            citation_payload.append(dict(getattr(c, "__dict__", {})))
    output_structured = dict(payload or {})  # docstring: 保留原始结构
    output_structured["answer"] = answer  # docstring: 覆盖 answer
    output_structured["citations"] = citation_payload  # docstring: 覆盖 citations

    # --- citations-only failure => blocked (not failed) ---
    if cfg.require_citations and not citations:
        # docstring: 如果仅因 citations 缺失/无效导致失败，则降级为 blocked 并清空 answer，避免“无引用但给出结论”
        citation_error_prefixes = (
            "citations missing",
            "no valid citations",
            "invalid citations:",
            "missing citations:",
        )
        only_citation_errors = bool(errors) and all(str(e).startswith(citation_error_prefixes) for e in errors)

        if only_citation_errors:
            # docstring: 强制 blocked 语义（检索有证据但生成未给出可验证引用）
            status = "blocked"
            answer = ""
            citations = []
            citation_payload = []
            output_structured["answer"] = ""
            output_structured["citations"] = []

    error_message = "; ".join(errors) if errors else None  # docstring: 错误汇总
    return {
        "answer": answer,
        "citations": citations,
        "output_structured": output_structured,
        "status": status,
        "error_message": error_message,
    }  # docstring: PostprocessResult
