# src/uae_law_rag/backend/pipelines/generation/prompt.py

"""
[职责] generation prompt：将检索 hits 组织为严格证据约束的 messages_snapshot，用于 LLM 调用与可回放审计。
[边界] 不做 LLM 调用；不访问 DB；不执行输出解析与 citations 对齐。
[上游关系] generation pipeline 传入 query 与 retrieval hits（可附加 node 快照）。
[下游关系] generator 使用 messages_snapshot 生成回答；generation_record 记录 prompt 与 evidence 快照。
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from uae_law_rag.backend.schemas.retrieval import RetrievalHit


__all__ = ["build_messages"]

DEFAULT_PROMPT_NAME = "uae_law_grounded"  # docstring: 默认 prompt 名称
DEFAULT_PROMPT_VERSION = "v1"  # docstring: 默认 prompt 版本
DEFAULT_MAX_EXCERPT_CHARS = 1200  # docstring: 单条证据最大长度

_TEXT_KEYS = (
    "excerpt",
    "text",
    "content",
    "node_text",
    "chunk_text",
    "raw_text",
    "page_text",
)  # docstring: 证据文本候选字段

SYSTEM_PROMPT = """You are UAE Law Assistant.
Rules:
- Use only the EVIDENCE provided.
- If evidence is insufficient, say you cannot answer and set citations to an empty array [].
- Every factual claim must cite at least one node_id from EVIDENCE.
- Do not invent facts or node_id.
- Output JSON only with keys: answer, citations.
"""  # docstring: system 角色与证据约束

OUTPUT_SCHEMA_EXAMPLE = """{
  "answer": "string",
  "citations": [
    {"node_id": "string", "rank": 1, "quote": "optional short quote"}
  ]
}"""  # docstring: 输出结构示例


def _normalize_query(query: str) -> str:
    """
    [职责] 规范化 query 文本（去除多余空白）。
    [边界] 不做语义改写；不做翻译。
    [上游关系] build_messages 调用。
    [下游关系] 用作 prompt QUESTION 内容。
    """
    raw = str(query or "")  # docstring: 兜底为字符串
    return " ".join(raw.strip().split())  # docstring: 合并多余空白


def _normalize_prompt_meta(prompt_name: str, prompt_version: Optional[str]) -> Tuple[str, str]:
    """
    [职责] 规范化 prompt_name 与 prompt_version。
    [边界] 仅做字符串归一化；缺省值使用默认常量。
    [上游关系] build_messages 调用。
    [下游关系] messages_snapshot 记录 prompt 元数据。
    """
    name = str(prompt_name or "").strip()  # docstring: prompt_name 兜底
    if not name:
        name = DEFAULT_PROMPT_NAME  # docstring: 缺省 prompt_name 兜底
    version = str(prompt_version or "").strip()  # docstring: prompt_version 兜底
    if not version:
        version = DEFAULT_PROMPT_VERSION  # docstring: 缺省版本兜底
    return name, version  # docstring: 输出规范化结果


def _coerce_int(value: Any) -> Optional[int]:
    """
    [职责] 将 value 转为 int（失败返回 None）。
    [边界] 仅处理常见数值与数字字符串。
    [上游关系] _build_evidence_item 调用。
    [下游关系] evidence.locator.page/offsets。
    """
    if value is None:
        return None  # docstring: 空值直接返回
    if isinstance(value, int):
        return value  # docstring: int 直接返回
    if isinstance(value, float):
        return int(value)  # docstring: float 转 int
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())  # docstring: 数字字符串转 int
    return None  # docstring: 不可解析时返回 None


def _compact_text(value: Any) -> str:
    """
    [职责] 压缩文本空白（合并换行与多空格）。
    [边界] 不做语义改写；不做去噪。
    [上游关系] _normalize_excerpt 调用。
    [下游关系] prompt evidence.excerpt。
    """
    raw = str(value or "")  # docstring: 兜底为字符串
    return " ".join(raw.split())  # docstring: 合并空白


def _truncate_text(text: str, *, max_chars: int) -> str:
    """
    [职责] 截断文本以控制 prompt 体积。
    [边界] 仅按字符截断；不做语言级截断。
    [上游关系] _normalize_excerpt 调用。
    [下游关系] prompt evidence.excerpt。
    """
    limit = int(max_chars)  # docstring: 最大长度兜底为 int
    if limit <= 0:
        return ""  # docstring: 非法 max_chars 返回空
    if len(text) <= limit:
        return text  # docstring: 长度足够直接返回
    return text[: max(limit - 3, 0)].rstrip() + "..."  # docstring: 超长时截断并加省略号


def _normalize_excerpt(value: Any, *, max_chars: int) -> str:
    """
    [职责] 将 evidence 文本压缩并截断。
    [边界] 不引入新的文本；仅做空白与长度处理。
    [上游关系] _pick_excerpt 调用。
    [下游关系] evidence.excerpt 字段。
    """
    compacted = _compact_text(value)  # docstring: 合并空白
    if not compacted:
        return ""  # docstring: 空文本直接返回
    return _truncate_text(compacted, max_chars=max_chars)  # docstring: 控制长度


def _read_hit_field(hit: Any, key: str, default: Any = None) -> Any:
    """
    [职责] 从 hit 对象读取字段（兼容 BaseModel/Mapping）。
    [边界] 不做类型校验；仅做字段读取。
    [上游关系] _build_evidence_item 调用。
    [下游关系] evidence 字段填充。
    """
    if hasattr(hit, key):
        return getattr(hit, key)  # docstring: 优先读取属性
    if isinstance(hit, Mapping):
        return hit.get(key, default)  # docstring: mapping 兜底读取
    return default  # docstring: 无字段时返回默认值


def _extract_snapshot(
    node_snapshots: Optional[Mapping[str, Mapping[str, Any]]],
    node_id: str,
) -> Dict[str, Any]:
    """
    [职责] 读取 node_id 对应的可选快照信息。
    [边界] 不验证快照结构；仅复制 dict。
    [上游关系] _build_evidence_item 调用。
    [下游关系] evidence.locator/ excerpt 补全。
    """
    if not node_snapshots:
        return {}  # docstring: 无快照时返回空
    snapshot = node_snapshots.get(node_id) or {}  # docstring: 读取 node 快照
    return dict(snapshot)  # docstring: 复制快照避免外部修改


def _pick_excerpt(hit: Any, snapshot: Mapping[str, Any], *, max_chars: int) -> str:
    """
    [职责] 从 hit/snapshot 中选取可用证据文本。
    [边界] 不拼接多段文本；仅取首个可用字段。
    [上游关系] _build_evidence_item 调用。
    [下游关系] evidence.excerpt 内容。
    """
    hit_excerpt = _read_hit_field(hit, "excerpt")  # docstring: 优先使用 hit.excerpt
    if isinstance(hit_excerpt, str) and hit_excerpt.strip():
        return _normalize_excerpt(hit_excerpt, max_chars=max_chars)  # docstring: 使用 hit.excerpt
    for key in _TEXT_KEYS:
        val = snapshot.get(key)
        if isinstance(val, str) and val.strip():
            return _normalize_excerpt(val, max_chars=max_chars)  # docstring: 使用快照文本字段
    return ""  # docstring: 无可用文本时返回空


def _build_evidence_item(
    hit: Any,
    *,
    rank: int,
    node_snapshots: Optional[Mapping[str, Mapping[str, Any]]],
    max_excerpt_chars: int,
) -> Optional[Dict[str, Any]]:
    """
    [职责] 将单条 hit 映射为 evidence item（用于 prompt 与审计快照）。
    [边界] 不读取 DB；仅使用 hit 与可选快照字段。
    [上游关系] build_messages 调用。
    [下游关系] evidence_items 与 prompt evidence 文本。
    """
    node_id = str(_read_hit_field(hit, "node_id", "") or "").strip()  # docstring: 解析 node_id
    if not node_id:
        return None  # docstring: 无 node_id 则跳过

    snapshot = _extract_snapshot(node_snapshots, node_id)  # docstring: 读取 node 快照
    raw_rank = _read_hit_field(hit, "rank")  # docstring: 尝试读取 hit.rank
    rank_out = int(raw_rank) if raw_rank is not None else int(rank)  # docstring: rank 兜底

    page = _coerce_int(_read_hit_field(hit, "page") or snapshot.get("page"))  # docstring: 页码快照
    start_offset = _coerce_int(
        _read_hit_field(hit, "start_offset") or snapshot.get("start_offset")
    )  # docstring: 起始偏移快照
    end_offset = _coerce_int(_read_hit_field(hit, "end_offset") or snapshot.get("end_offset"))  # docstring: 结束偏移

    article_id = snapshot.get("article_id") or snapshot.get("article")  # docstring: article_id 兜底
    section_path = snapshot.get("section_path") or snapshot.get("section")  # docstring: section_path 兜底
    source = _read_hit_field(hit, "source") or snapshot.get("source")  # docstring: 命中来源快照

    locator = {
        "page": page,
        "article_id": str(article_id).strip() if article_id is not None and str(article_id).strip() else None,
        "section_path": str(section_path).strip() if section_path is not None and str(section_path).strip() else None,
        "start_offset": start_offset,
        "end_offset": end_offset,
        "source": str(source).strip() if source is not None and str(source).strip() else None,
    }  # docstring: 证据定位快照

    excerpt = _pick_excerpt(hit, snapshot, max_chars=max_excerpt_chars)  # docstring: 提取证据文本
    return {
        "rank": rank_out,
        "node_id": node_id,
        "excerpt": excerpt,
        "locator": locator,
    }  # docstring: evidence item


def _build_evidence_items(
    hits: Sequence[RetrievalHit],
    *,
    node_snapshots: Optional[Mapping[str, Mapping[str, Any]]],
    max_excerpt_chars: int,
) -> List[Dict[str, Any]]:
    """
    [职责] 将 hits 列表映射为 evidence_items 列表。
    [边界] 不重排 hits；仅按输入顺序赋 rank 兜底。
    [上游关系] build_messages 调用。
    [下游关系] _format_evidence_block / messages_snapshot。
    """
    items: List[Dict[str, Any]] = []  # docstring: evidence_items 容器
    for idx, hit in enumerate(hits or [], start=1):
        item = _build_evidence_item(
            hit,
            rank=idx,
            node_snapshots=node_snapshots,
            max_excerpt_chars=max_excerpt_chars,
        )  # docstring: 构造 evidence item
        if item:
            items.append(item)  # docstring: 收集有效 evidence
    return items  # docstring: 返回 evidence_items


def _format_locator_value(value: Any) -> str:
    """
    [职责] 将 locator 值格式化为可读字符串。
    [边界] 不做语义转换；None 返回 unknown。
    [上游关系] _format_evidence_block 调用。
    [下游关系] prompt evidence 文本。
    """
    if value is None:
        return "unknown"  # docstring: None 兜底
    text = str(value).strip()
    return text if text else "unknown"  # docstring: 空字符串兜底


def _format_offsets(start_offset: Optional[int], end_offset: Optional[int]) -> str:
    """
    [职责] 格式化 offset 范围。
    [边界] 仅拼接数值；不验证区间顺序。
    [上游关系] _format_evidence_block 调用。
    [下游关系] prompt evidence 文本。
    """
    if start_offset is None and end_offset is None:
        return "unknown"  # docstring: 无 offset 兜底
    start = "" if start_offset is None else str(start_offset)  # docstring: start_offset 兜底
    end = "" if end_offset is None else str(end_offset)  # docstring: end_offset 兜底
    return f"{start}-{end}".strip("-")  # docstring: 拼接 offset 范围


def _format_evidence_block(items: Sequence[Mapping[str, Any]]) -> str:
    """
    [职责] 将 evidence_items 格式化为 prompt 文本块。
    [边界] 仅格式化文本；不做排序与去重。
    [上游关系] _build_user_prompt 调用。
    [下游关系] user message 内容。
    """
    if not items:
        return "(no evidence)"  # docstring: 空 evidence 兜底

    lines: List[str] = []  # docstring: prompt 文本行
    for item in items:
        locator = item.get("locator") or {}  # docstring: 定位信息
        rank = item.get("rank")  # docstring: evidence rank
        node_id = item.get("node_id")  # docstring: evidence node_id
        page = _format_locator_value(locator.get("page"))  # docstring: 格式化页码
        article_id = _format_locator_value(locator.get("article_id"))  # docstring: 格式化 article_id
        section_path = _format_locator_value(locator.get("section_path"))  # docstring: 格式化 section_path
        offsets = _format_offsets(locator.get("start_offset"), locator.get("end_offset"))  # docstring: offset 文本

        header = (
            f"[{rank}] node_id={node_id} page={page} article_id={article_id} "
            f"section_path={section_path} offsets={offsets}"
        )  # docstring: evidence 行头部
        lines.append(header)  # docstring: 写入 header

        excerpt = str(item.get("excerpt") or "").strip()  # docstring: 证据文本
        if excerpt:
            lines.append(f'    "{excerpt}"')  # docstring: 写入 excerpt
        else:
            lines.append('    "(no excerpt)"')  # docstring: 无 excerpt 兜底

    return "\n".join(lines)  # docstring: 合并为 evidence block


def _build_system_prompt() -> str:
    """
    [职责] 构建 system prompt（角色与证据约束）。
    [边界] 仅返回静态模板；不注入上下文。
    [上游关系] build_messages 调用。
    [下游关系] messages_snapshot.system。
    """
    return SYSTEM_PROMPT.strip()  # docstring: 去除首尾空白


def _build_user_prompt(
    *,
    query: str,
    evidence_items: Sequence[Mapping[str, Any]],
    valid_node_ids: Sequence[str],
) -> str:
    """
    [职责] 构建 user prompt（证据块 + 问题 + 输出格式）。
    [边界] 仅拼接文本；不做检索与校验。
    [上游关系] build_messages 调用。
    [下游关系] messages_snapshot.user。
    """
    evidence_block = _format_evidence_block(evidence_items)  # docstring: evidence 文本块
    node_ids = ", ".join(valid_node_ids)  # docstring: node_id 列表拼接
    node_line = f"[{node_ids}]" if node_ids else "[]"  # docstring: node_id 列表文本

    sections = [
        "EVIDENCE:",
        evidence_block,
        "",
        f"VALID NODE IDS: {node_line}",
        "",
        "QUESTION:",
        query,
        "",
        "OUTPUT FORMAT (JSON):",
        OUTPUT_SCHEMA_EXAMPLE.strip(),
    ]  # docstring: prompt 分段
    return "\n".join(sections).strip()  # docstring: 拼接 user prompt


def build_messages(
    *,
    query: str,
    hits: Sequence[RetrievalHit],
    prompt_name: str,
    prompt_version: Optional[str] = None,
    node_snapshots: Optional[Mapping[str, Mapping[str, Any]]] = None,
    max_excerpt_chars: int = DEFAULT_MAX_EXCERPT_CHARS,
) -> Dict[str, Any]:
    """
    [职责] 构建 messages_snapshot（system/user + evidence + 输出结构约束）。
    [边界] 不调用 LLM；不访问 DB；不解析输出。
    [上游关系] generation pipeline 传入 query 与 retrieval hits。
    [下游关系] generator 使用 messages_snapshot 调用模型；generation_record 落库快照。
    """
    normalized_query = _normalize_query(query)  # docstring: 规范化 query
    name, version = _normalize_prompt_meta(prompt_name, prompt_version)  # docstring: 规范化 prompt 元数据

    evidence_items = _build_evidence_items(
        hits,
        node_snapshots=node_snapshots,
        max_excerpt_chars=max_excerpt_chars,
    )  # docstring: evidence 列表

    node_ids: List[str] = []  # docstring: node_id 去重列表
    seen: set[str] = set()  # docstring: 去重缓存
    for item in evidence_items:
        node_id = str(item.get("node_id") or "").strip()
        if not node_id or node_id in seen:
            continue
        seen.add(node_id)  # docstring: 记录已见 node_id
        node_ids.append(node_id)  # docstring: 收集 node_id

    system_prompt = _build_system_prompt()  # docstring: system prompt
    user_prompt = _build_user_prompt(
        query=normalized_query,
        evidence_items=evidence_items,
        valid_node_ids=node_ids,
    )  # docstring: user prompt

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]  # docstring: messages 列表

    return {
        "prompt_name": name,
        "prompt_version": version,
        "query": normalized_query,
        "messages": messages,
        "evidence": evidence_items,
        "valid_node_ids": node_ids,
        "evidence_count": len(evidence_items),
        "output_schema": OUTPUT_SCHEMA_EXAMPLE.strip(),
    }  # docstring: messages_snapshot
