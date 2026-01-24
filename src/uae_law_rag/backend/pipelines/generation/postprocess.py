# src/uae_law_rag/backend/pipelines/generation/postprocess.py

"""
[职责] generation postprocess：解析 LLM 输出，校验结构与证据引用，并补全 citation 定位信息。
[边界] 不调用 LLM；不访问 DB；不执行落库；仅处理输出结构与引用对齐。
[上游关系] generator 返回 raw_text；retrieval 提供 hits 作为证据集合。
[下游关系] persist 写入 output_structured/citations；evaluator 消费 citations 做审计。
"""

from __future__ import annotations

import re
import json
from dataclasses import dataclass
from uuid import UUID
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, cast, Protocol

from uae_law_rag.backend.schemas.generation import Citation
from uae_law_rag.backend.schemas.ids import NodeId, UUIDStr


__all__ = ["postprocess_generation"]

_STATUS_ORDER = {"success": 0, "partial": 1, "blocked": 2, "failed": 3}  # docstring: status 严重程度顺序
_DEFAULT_MAX_QUOTE_CHARS = 240  # docstring: citation.quote 最大长度


# ----------------------------
# Typing: RetrievalHitLike
# ----------------------------
class RetrievalHitLike(Protocol):
    """
    [职责] RetrievalHitLike：postprocess 所需的最小 hit 结构合同（结构化子类型）。
    [边界] 不要求 ORM/schema 具体类型；只要能 getattr/读到这些字段即可。
    """

    node_id: Any
    rank: Any
    excerpt: Any
    page: Any
    start_offset: Any
    end_offset: Any
    article_id: Any
    section_path: Any
    source: Any
    score_details: Any


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

    # NEW:
    min_citations: int
    require_quote: bool
    allow_quote_fallback_to_excerpt: bool
    require_rank_markers_in_answer: bool


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
        """
        [职责] 将 cfg[key] 归一为 bool。
        [边界] 仅做最小可预期转换；确保返回类型恒为 bool（避免误返回 tuple 导致类型检查报错）。
        """
        val = cfg.get(key, default)
        if val is None:
            return bool(default)
        # docstring: 常见字符串开关（兼容环境变量/配置文件）
        if isinstance(val, str):
            s = val.strip().lower()
            if s in {"1", "true", "yes", "y", "on"}:
                return True
            if s in {"0", "false", "no", "n", "off"}:
                return False
        return bool(val)  # docstring: 其它类型按 truthy 规则处理

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
        # NEW defaults tuned for auditability (and small models)
        min_citations=max(1, _as_int("min_citations", 1)),
        # NOTE: small/local models often emit "quote" with extra markup (e.g., <!-- page -->, **bold**),
        # which is not a strict substring of hit.excerpt. Default to excerpt fallback to avoid false blocks.
        # docstring: 强制 bool，避免任何 patch 合并/编辑器自动加逗号导致 tuple 推断。
        require_quote=bool(_as_bool("require_quote", True)),
        allow_quote_fallback_to_excerpt=bool(_as_bool("allow_quote_fallback_to_excerpt", True)),
        # NOTE: rank markers in answer are useful but too strict for baseline; keep opt-in.
        require_rank_markers_in_answer=bool(_as_bool("require_rank_markers_in_answer", False)),
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


def _build_hit_index(hits: Sequence[RetrievalHitLike]) -> Dict[str, RetrievalHitLike]:
    """
    [职责] 构建 node_id(uuid str) -> hit 索引，供 citation 对齐使用。
    [边界] 只收集可解析 UUID 的 node_id；其它跳过。
    """
    out: Dict[str, RetrievalHitLike] = {}
    for h in list(hits or []):
        raw = _coerce_str(_read_hit_field(h, "node_id")) or ""
        nid = _extract_uuid(raw)  # docstring: 兼容 UUID / 带噪声字符串
        if not nid:
            continue
        key = str(nid)
        # docstring: 去重策略：同一个 node_id 保留首次
        if key not in out:
            out[key] = h
    return out


def _build_hit_rank_index(hits: Sequence[RetrievalHitLike]) -> Dict[int, RetrievalHitLike]:
    """
    [职责] 构建 rank -> hit 的索引，用于 node_id 缺失/不合法时回退对齐。
    [边界] 仅收集可解析的 int rank；重复 rank 取首次出现。
    """
    out: Dict[int, RetrievalHitLike] = {}
    for h in list(hits or []):
        rr = _coerce_int(_read_hit_field(h, "rank"))
        if rr is None:
            continue
        if rr not in out:
            out[rr] = h
    return out


_CODE_FENCE_OPEN_RE = re.compile(r"^\s*```(?:json)?\s*\n?", re.IGNORECASE)
_CODE_FENCE_CLOSE_RE = re.compile(r"\n?\s*```\s*$", re.IGNORECASE)


def _strip_code_fences(text: str) -> str:
    """
    [职责] 去掉最外层 Markdown code fence（``` / ```json）。
    [边界] 只剥一层；不尝试处理嵌套/多段代码块。
    """
    s = str(text or "").strip()
    if not s:
        return ""
    # Only strip if it looks like a single fenced block
    if s.lstrip().startswith("```"):
        s = _CODE_FENCE_OPEN_RE.sub("", s).strip()
        s = _CODE_FENCE_CLOSE_RE.sub("", s).strip()
    return s


def _extract_first_json_object_region(text: str) -> Tuple[Optional[str], Optional[str]]:
    """
    [职责] 在 text 中提取第一个“完整闭合”的 JSON object 区域（{ ... }）。
    [边界] 仅做括号平衡扫描；不保证一定可被 json.loads 解析。
    """
    s = str(text or "")
    start = s.find("{")
    if start < 0:
        return None, "json parse error: unable to locate json object"

    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(s)):
        ch = s[i]
        if in_str:
            if esc:
                esc = False
                continue
            if ch == "\\":
                esc = True
                continue
            if ch == '"':
                in_str = False
            continue

        # not in string
        if ch == '"':
            in_str = True
            continue
        if ch == "{":
            depth += 1
            continue
        if ch == "}":
            depth -= 1
            if depth == 0:
                return s[start : i + 1], None
            continue

    # ran to end but still not closed
    return None, f"json parse error: incomplete json object (missing closing brace), depth={depth}"


def _parse_json(raw_text: str, *, strict: bool) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """
    [职责] 解析 raw_text 为 JSON dict。
    [边界] strict=True 时要求“最终内容可被视作一个 JSON 对象”；允许剥离最外层 code fence。
    """
    text0 = str(raw_text or "").strip()
    if not text0:
        return None, "raw_text is empty"

    text = _strip_code_fences(text0)  # NEW: tolerate ```json ... ```
    if not text:
        return None, "raw_text is empty"

    # --- strict path ---
    if strict:
        # 1) try strict parse on stripped text
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            # 1.5) repair path: try to close missing braces/brackets deterministically
            repaired = _repair_incomplete_json_object(text)
            if repaired and repaired != text:
                try:
                    data = json.loads(repaired)
                except json.JSONDecodeError:
                    data = None
                else:
                    if not isinstance(data, dict):
                        return None, "json root must be an object"
                    return dict(data), None
            # 2) strict fallback: extract first balanced JSON object region (still enforces root object)
            region, region_err = _extract_first_json_object_region(text)
            if region_err:
                return None, region_err
            try:
                data = json.loads(region or "")
            except json.JSONDecodeError as exc2:
                return None, f"json parse error: {exc2}"
        if not isinstance(data, dict):
            return None, "json root must be an object"
        return dict(data), None

    # --- non-strict path (existing behavior, but now after fence stripping) ---
    region, region_err = _extract_first_json_object_region(text)
    if region_err:
        # keep legacy message category in non-strict mode
        if "unable to locate" in str(region_err):
            return None, "json object not found"
        return None, region_err
    try:
        data = json.loads(region or "")
    except json.JSONDecodeError as exc:
        return None, f"json parse error: {exc}"
    if not isinstance(data, dict):
        return None, "json root must be an object"
    return dict(data), None


def _repair_incomplete_json_object(text: str) -> str:
    """
    [职责] 尝试修复“被截断的单个 JSON object”（缺失右花括号/右中括号）。
    [边界] 仅做括号补全；不做内容改写；只在看起来是从 '{' 开始时生效。
    """
    s = str(text or "").strip()
    if not s:
        return ""
    start = s.find("{")
    if start < 0:
        return ""
    # 仅修复从第一个 '{' 起的片段（避免前面有多余文本）
    frag = s[start:]
    stack: list[str] = []
    in_str = False
    esc = False
    for ch in frag:
        if in_str:
            if esc:
                esc = False
                continue
            if ch == "\\":
                esc = True
                continue
            if ch == '"':
                in_str = False
            continue
        else:
            if ch == '"':
                in_str = True
                continue
            if ch == "{":
                stack.append("}")
            elif ch == "[":
                stack.append("]")
            elif ch == "}" or ch == "]":
                if stack and stack[-1] == ch:
                    stack.pop()
                else:
                    # 结构已经乱了，不修
                    return ""
    if in_str:
        # 字符串都没闭合，修复意义不大
        return ""
    if not stack:
        return frag
    # 补全缺失的闭合符号（后进先出）
    return frag + "".join(reversed(stack))


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


def _build_locator_from_hit(hit: RetrievalHitLike) -> Dict[str, Any]:
    """
    [职责] 从 hit 构建 locator 快照。
    [边界] 仅使用 hit 的可用字段；缺失字段返回 None。
    [上游关系] citation 对齐逻辑调用。
    [下游关系] Citation.locator。
    """
    page = _coerce_int(_read_hit_field(hit, "page"))
    if page == 0:
        page = None  # docstring: 兼容历史 sentinel；locator 不允许出现 page=0
    locator = {
        "page": page,
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


_UUID_RE = re.compile(
    r"(?i)\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b"
)  # docstring: uuid v4-ish pattern matcher (case-insensitive)


def _extract_uuid(text: str) -> str:
    """
    [职责] 从任意文本中提取第一个 UUID（容错 bullet/反引号/尾注）。
    [边界] 找不到则返回空字符串。
    [上游关系] _build_citation 调用。
    [下游关系] node_id 校验与 Citation 构造。
    """
    s = str(text or "").strip()
    if not s:
        return ""
    # docstring: 常见噪声清理（markdown/bullets）
    s = s.strip("`").strip()
    # docstring: 如果整段不是纯 uuid，尝试在文本中搜索 uuid 子串
    m = _UUID_RE.search(s)
    return (m.group(0) if m else "").lower()


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
    hit: RetrievalHitLike,
    rank_fallback: int,
    max_quote_chars: int,
    require_quote: bool,  # NEW
    allow_quote_fallback_to_excerpt: bool,  # NEW
) -> Citation:
    """
    [职责] 构造 Citation（补全 rank/locator/quote）。
    [边界] 不校验 node_id 是否存在于 hits（由上游保证）。
    [上游关系] postprocess_generation 调用。
    [下游关系] PostprocessResult.citations。
    """
    node_id_raw = _coerce_str(parsed.get("node_id")) or ""  # docstring: node_id 原始值
    node_id_norm = _extract_uuid(node_id_raw)  # docstring: 从噪声文本中提取 uuid
    # docstring: 运行时校验 UUID，避免 Citation 构造阶段抛异常导致 postprocess 崩溃
    if not node_id_norm or not _is_uuid(node_id_norm):
        raise ValueError("node_id is not a valid UUID")
    node_id = cast(NodeId, UUIDStr(node_id_norm))  # docstring: node_id 类型对齐
    # docstring: rank 以 hit.rank 为准（LLM rank 仅作为提示，避免 0/1-based 与全局排序差异导致误判）
    rank = _coerce_int(_read_hit_field(hit, "rank"))
    if rank is None:
        rank = _coerce_int(parsed.get("rank"))  # docstring: 回退到 LLM rank
    if rank is None:
        rank = int(rank_fallback)  # docstring: 最终回退索引

    quote = _coerce_str(parsed.get("quote"))  # docstring: quote 解析

    if not quote and allow_quote_fallback_to_excerpt:
        quote = _coerce_str(_read_hit_field(hit, "excerpt"))

    if require_quote and (not quote or not quote.strip()):
        raise ValueError("quote is required")

    # quote must be reproducible from hit excerpt (if excerpt exists)
    hit_excerpt = _coerce_str(_read_hit_field(hit, "excerpt")) or ""
    if quote and hit_excerpt:
        # docstring: normalize common markup so small models won't be punished for adding page markers/bold.
        q0 = re.sub(r"<!--.*?-->", " ", quote, flags=re.DOTALL)
        q0 = re.sub(r"[*_`]+", "", q0)  # markdown emphasis/code
        q0 = re.sub(r"\s+", " ", q0).strip().lower()
        e0 = re.sub(r"\s+", " ", hit_excerpt).strip().lower()
        if q0 and q0 not in e0:
            # docstring: default policy is to fall back to excerpt (verifiable), not to hard-fail.
            if allow_quote_fallback_to_excerpt and hit_excerpt:
                quote = hit_excerpt
            else:
                raise ValueError("quote not found in hit excerpt")

    if quote:
        quote = _truncate_text(quote, max_chars=max_quote_chars)  # docstring: quote 截断

    parsed_locator_raw = parsed.get("locator")  # docstring: citation locator 原始值
    parsed_locator = parsed_locator_raw if isinstance(parsed_locator_raw, Mapping) else {}  # docstring: locator 解析
    locator = _merge_locator(parsed_locator, _build_locator_from_hit(hit))  # docstring: locator 合并

    # docstring: flatten common locator fields for HTTP CitationView convenience
    page = _coerce_int(locator.get("page"))
    article_id = _coerce_str(locator.get("article_id"))
    section_path = _coerce_str(locator.get("section_path"))
    return Citation(
        node_id=node_id,
        rank=rank,
        quote=quote or "",
        locator=locator,
        page=page,
        article_id=article_id,
        section_path=section_path,
    )


def postprocess_generation(
    *,
    raw_text: str,
    hits: Sequence[RetrievalHitLike],
    config: Optional[Mapping[str, Any]] = None,
    allowed_node_ids: Optional[Sequence[str]] = None,  # NEW
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

    rank_map = _build_hit_rank_index(hits)  # docstring: rank -> hit 索引（node_id 回退用）

    # NEW: restrict citations to allowed_node_ids (prompt-valid ids)
    allow: Optional[set[str]] = None
    if allowed_node_ids:
        allow = {str(x).strip().lower() for x in allowed_node_ids if str(x or "").strip()}
        hit_map = {k: v for k, v in hit_map.items() if k in allow}
        # docstring: 同步过滤 rank_map，避免 rank fallback 绕过 allow-list
        rank_map = {
            r: h
            for r, h in rank_map.items()
            if _extract_uuid(_coerce_str(_read_hit_field(h, "node_id")) or "") in allow
        }

    citations: List[Citation] = []  # docstring: 输出 citations
    invalid_count = 0  # docstring: 无效引用计数
    missing_count = 0  # docstring: 未命中引用计数
    seen: set[str] = set()  # docstring: 去重缓存

    for idx, item in enumerate(raw_citations, start=1):
        parsed, err = _parse_citation_item(item)  # docstring: 解析 citation
        if err or not parsed:
            invalid_count += 1  # docstring: 记录无效 citation
            continue
        node_id_raw = _coerce_str(parsed.get("node_id"))  # docstring: 原始 node_id
        node_id_norm = _extract_uuid(node_id_raw or "")  # docstring: 尝试从噪声中提取 uuid

        hit = None  # docstring: 目标命中
        node_id_final = ""  # docstring: 最终 node_id（必须是 uuid str）

        # (A) 优先使用可解析 UUID 的 node_id
        if node_id_norm:
            node_id_final = str(node_id_norm)
            if node_id_final in seen:
                invalid_count += 1
                continue
            hit = hit_map.get(node_id_final)
            if not hit:
                # docstring: UUID 合法但不在 hits 中 -> missing
                missing_count += 1
                continue

        # (B) node_id 不可用时：回退用 rank 映射 hits
        if hit is None:
            rr = _coerce_int(parsed.get("rank"))
            if rr is None:
                rr = idx  # docstring: 连 rank 都没有时，用当前位置兜底（更符合小模型输出）
            hit = rank_map.get(rr)
            if hit is None and 1 <= int(rr) <= len(hits):
                # docstring: 兼容 1-based rank
                hit = hits[int(rr) - 1]
            if hit is None:
                invalid_count += 1
                continue

            # docstring: 从命中的 hit 取 node_id，作为最终对齐结果
            node_id_hit = _extract_uuid(_coerce_str(_read_hit_field(hit, "node_id")) or "")
            if not node_id_hit:
                invalid_count += 1
                continue
            # IMPORTANT: in rank-fallback branch, node_id_final may still be empty here.
            # The allow-list must be checked against the resolved hit node_id.
            if allow is not None and str(node_id_hit).strip().lower() not in allow:
                missing_count += 1
                continue
            node_id_final = str(node_id_hit)
            if node_id_final in seen:
                invalid_count += 1
                continue

        # docstring: 将修正后的 node_id 回写，保证 _build_citation 使用的是对齐后的 uuid
        parsed["node_id"] = node_id_final

        # --- citation-eligible check (must have non-empty excerpt) ---
        hit_excerpt = _coerce_str(_read_hit_field(hit, "excerpt"))
        if not hit_excerpt or not hit_excerpt.strip():
            # docstring: 引用命中但无可验证文本，视为缺失引用（强制进入 blocked 语义）
            missing_count += 1
            continue

        try:
            citation = _build_citation(
                parsed=parsed,
                hit=hit,
                rank_fallback=idx,
                max_quote_chars=cfg.max_quote_chars,
                require_quote=cfg.require_quote,
                allow_quote_fallback_to_excerpt=cfg.allow_quote_fallback_to_excerpt,
            )  # docstring: 构造对齐 Citation
        except Exception:
            invalid_count += 1  # docstring: citation 构造失败视为无效引用
            continue
        citations.append(citation)  # docstring: 收集 citation
        seen.add(node_id_final)  # docstring: 记录已处理 node_id

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

    # NOTE:
    # Do NOT clear/override early here. Defer "min citations" enforcement to the final
    # blocked decision block so that rank normalization / marker auto-fix can run first.

    # NEW: normalize citation ranks to 1..N (avoid duplicated/misaligned ranks from small models)
    if citations:
        normalized: List[Citation] = []
        for i, c in enumerate(citations, start=1):
            normalized.append(
                Citation(
                    node_id=getattr(c, "node_id"),
                    rank=i,
                    quote=getattr(c, "quote", "") or "",
                    locator=getattr(c, "locator", {}) or {},
                    page=getattr(c, "page", None),
                    article_id=getattr(c, "article_id", None),
                    section_path=getattr(c, "section_path", None),
                )
            )
        citations = normalized

        # refresh payload after normalization (keep deterministic)
        citation_payload = []
        for c in citations:
            if hasattr(c, "model_dump"):
                citation_payload.append(c.model_dump())  # type: ignore[attr-defined]
            elif hasattr(c, "dict"):
                citation_payload.append(c.dict())  # type: ignore[call-arg]
            else:
                citation_payload.append(dict(getattr(c, "__dict__", {})))
        output_structured["citations"] = citation_payload

    # NEW: ensure answer contains rank markers expected by alignment gate
    # NOTE: alignment gate is a hard contract; enforce markers whenever citations exist.
    if citations and answer:
        missing_marks: List[str] = []
        for c in citations:
            r = _coerce_int(getattr(c, "rank", None))
            if r is None:
                continue
            mark = f"[{r}]"
            if mark not in answer:
                missing_marks.append(mark)
        if missing_marks:
            # Deterministic, audit-friendly auto-fix: append a citation marker line.
            # This satisfies gate expectations without forcing a hard block.
            answer = answer.rstrip() + "\n\nCitations: " + " ".join(missing_marks) + "\n"
            output_structured["answer"] = answer
            status = _merge_status(status, "partial")
            errors.append(f"auto_appended_rank_markers={','.join(missing_marks)}")

    # --- require_citations => no valid citations => BLOCKED (highest priority) ---
    # Also enforce min_citations here (unified blocked logic).
    need_block = False
    if cfg.require_citations and not citations:
        need_block = True
        if not any(
            str(e).startswith(("citations", "no valid citations", "invalid citations", "missing citations"))
            for e in errors
        ):
            errors.append("no valid citations")
    if cfg.require_citations and citations and len(citations) < cfg.min_citations:
        need_block = True
        errors = [e for e in errors if not str(e).startswith("answer")]
        errors.append(f"insufficient citations: {len(citations)} < {cfg.min_citations}")

    if need_block:
        # docstring: 只要要求 citations 且最终无有效 citations，就必须 blocked；
        # 这表示“证据存在但未能产出可验证引用”，不是基础设施失败。
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
        "meta": {
            "invalid_count": invalid_count,
            "missing_count": missing_count,
            "raw_citations_count": len(raw_citations or []),
        },
    }  # docstring: PostprocessResult
