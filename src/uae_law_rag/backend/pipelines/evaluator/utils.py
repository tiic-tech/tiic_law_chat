# src/uae_law_rag/backend/pipelines/evaluator/utils.py

"""
[职责] evaluator utils：提供覆盖率计算、集合关系判断与文本规范化工具，供 evaluator 规则层复用。
[边界] 不产生 pass/fail 语义；不访问 DB/LLM；不包含 pipeline 编排逻辑。
[上游关系] evaluator checks/pipeline 调用这些工具处理 citations/hits 与文本。
[下游关系] evaluator checks 输出 EvaluationCheck，pipeline 输出 EvaluationResult。
"""

from __future__ import annotations

from typing import Any, Iterable, List, Mapping, Sequence


__all__ = [
    "compute_citation_coverage",
    "compute_coverage",
    "extract_node_ids",
    "has_intersection",
    "is_subset",
    "normalize_text",
]


def _coerce_str(value: Any) -> str:
    """
    [职责] 将任意值规范为字符串并去空白。
    [边界] 空值返回空字符串；不做语义清洗。
    [上游关系] extract_node_ids/normalize_text 调用。
    [下游关系] node_id 与文本处理的稳定输入。
    """
    return str(value or "").strip()  # docstring: 字符串兜底


def _read_field(obj: Any, key: str) -> Any:
    """
    [职责] 从对象或 mapping 安全读取字段值。
    [边界] 不抛异常；字段不存在返回 None。
    [上游关系] extract_node_ids 调用。
    [下游关系] node_id/items/nodes 解析。
    """
    if obj is None:
        return None  # docstring: 空对象回退
    if isinstance(obj, Mapping):
        return obj.get(key)  # docstring: mapping 读取
    return getattr(obj, key, None)  # docstring: attribute 读取


def _iter_items(value: Any) -> List[Any]:
    """
    [职责] 将输入值规整为可遍历 item 列表。
    [边界] 不保证 item 合法性；仅做类型兜底。
    [上游关系] extract_node_ids 调用。
    [下游关系] node_id 提取循环。
    """
    if value is None:
        return []  # docstring: 空输入回退
    if isinstance(value, (str, bytes, bytearray)):
        return [value]  # docstring: 字符串视为单条
    if isinstance(value, Mapping):
        return [value]  # docstring: mapping 视为单条
    if isinstance(value, Sequence):
        return list(value)  # docstring: 序列展开
    return []  # docstring: 其他类型回退空


def _extract_node_id(item: Any) -> str:
    """
    [职责] 从单条 item 中提取 node_id。
    [边界] 仅做字段读取与类型兜底；不校验存在性。
    [上游关系] extract_node_ids 调用。
    [下游关系] 覆盖率与集合关系计算。
    """
    if isinstance(item, Mapping):
        return _coerce_str(item.get("node_id") or item.get("id") or item.get("nodeId"))  # docstring: mapping node_id
    if isinstance(item, (bytes, bytearray)):
        try:
            return _coerce_str(item.decode("utf-8", errors="ignore"))  # docstring: bytes -> str
        except Exception:
            return ""  # docstring: decode 失败回退空
    if hasattr(item, "node_id"):
        return _coerce_str(getattr(item, "node_id", ""))  # docstring: attribute node_id
    if isinstance(item, str):
        return _coerce_str(item)  # docstring: 字符串视为 node_id
    return _coerce_str(item)  # docstring: 兜底转字符串（兼容 UUID/NodeId 等）


def _build_id_set(values: Iterable[str]) -> set[str]:
    """
    [职责] 将输入值构建为去空的 set。
    [边界] 不保序；仅做去空/去重。
    [上游关系] compute_coverage/is_subset/has_intersection 调用。
    [下游关系] 集合关系与覆盖率计算。
    """
    out: set[str] = set()  # docstring: 输出集合
    for value in values:
        item = _coerce_str(value)  # docstring: 规范化字符串
        if not item:
            continue  # docstring: 空值跳过
        out.add(item)  # docstring: 添加到集合
    return out  # docstring: 返回集合


def extract_node_ids(items: Any) -> List[str]:
    """
    [职责] 从 citations/hits/payload 中提取 node_id 列表（去重，保留顺序）。
    [边界] 不访问 DB；不校验 node_id 真实性；仅做结构解析与去重。
    [上游关系] evaluator checks/pipeline 输入 citations/hits/payload。
    [下游关系] compute_citation_coverage/is_subset/has_intersection 使用。
    """
    payload = items  # docstring: 原始输入
    if isinstance(payload, Mapping):
        nested = payload.get("items") or payload.get("nodes")  # docstring: payload 读取 items/nodes
        if nested is not None:
            payload = nested  # docstring: payload 下钻
    else:
        nested_items = _read_field(payload, "items")  # docstring: object.items
        nested_nodes = _read_field(payload, "nodes")  # docstring: object.nodes
        if nested_items is not None or nested_nodes is not None:
            payload = nested_items or nested_nodes  # docstring: object 下钻

    node_ids: List[str] = []  # docstring: 输出列表
    seen: set[str] = set()  # docstring: 去重集合
    for item in _iter_items(payload):
        node_id = _extract_node_id(item)  # docstring: 提取 node_id
        if not node_id or node_id in seen:
            continue  # docstring: 空/重复跳过
        seen.add(node_id)  # docstring: 记录已见 node_id
        node_ids.append(node_id)  # docstring: 追加 node_id
    return node_ids  # docstring: 返回去重结果


def compute_coverage(target_ids: Iterable[str], reference_ids: Iterable[str]) -> float:
    """
    [职责] 计算 target_ids 在 reference_ids 中的覆盖率。
    [边界] 空 target 返回 0.0；不做加权；仅做集合比例。
    [上游关系] compute_citation_coverage 等调用。
    [下游关系] evaluator scores/metrics 使用。
    """
    target_set = _build_id_set(target_ids)  # docstring: 目标集合去空
    if not target_set:
        return 0.0  # docstring: 空 target 覆盖率为 0
    reference_set = _build_id_set(reference_ids)  # docstring: 参考集合去空
    matched = target_set.intersection(reference_set)  # docstring: 计算交集
    return float(len(matched)) / float(len(target_set))  # docstring: 覆盖率比例


def compute_citation_coverage(citations: Any, hits: Any) -> float:
    """
    [职责] 计算 citations 对 retrieval hits 的覆盖率（node_id 维度）。
    [边界] citations 为空返回 0.0；不校验 node_id 真实性。
    [上游关系] evaluator pipeline/checks 提供 citations 与 hits。
    [下游关系] evaluator scores/门禁辅助逻辑使用。
    """
    citation_ids = extract_node_ids(citations)  # docstring: 提取 citations node_id
    hit_ids = extract_node_ids(hits)  # docstring: 提取 hits node_id
    return compute_coverage(citation_ids, hit_ids)  # docstring: 计算覆盖率


def is_subset(subset_ids: Iterable[str], superset_ids: Iterable[str]) -> bool:
    """
    [职责] 判断 subset_ids 是否为 superset_ids 的子集。
    [边界] 仅做集合关系判断；空集视为 True。
    [上游关系] evaluator checks 使用。
    [下游关系] 用于判断引用是否越界等。
    """
    subset_set = _build_id_set(subset_ids)  # docstring: 子集去空
    superset_set = _build_id_set(superset_ids)  # docstring: 超集去空
    return subset_set.issubset(superset_set)  # docstring: 子集判断


def has_intersection(left_ids: Iterable[str], right_ids: Iterable[str]) -> bool:
    """
    [职责] 判断两个集合是否存在交集。
    [边界] 仅做集合关系判断；不做权重与排序。
    [上游关系] evaluator checks 使用。
    [下游关系] 判断 evidence 是否相交等。
    """
    left_set = _build_id_set(left_ids)  # docstring: 左集合去空
    right_set = _build_id_set(right_ids)  # docstring: 右集合去空
    if not left_set or not right_set:
        return False  # docstring: 任一为空则无交集
    return bool(left_set.intersection(right_set))  # docstring: 交集判断


def normalize_text(text: Any) -> str:
    """
    [职责] 规范化文本（去空白、lower、压缩空白）。
    [边界] 不做分词/语言检测；不替换标点与语义。
    [上游关系] evaluator checks 或 utils 调用。
    [下游关系] 文本相似度/匹配辅助使用。
    """
    raw = _coerce_str(text)  # docstring: 字符串兜底
    if not raw:
        return ""  # docstring: 空文本直接返回
    lowered = raw.lower()  # docstring: lowercase 归一化
    collapsed = " ".join(lowered.split())  # docstring: 压缩连续空白
    return collapsed  # docstring: 返回规范化文本
