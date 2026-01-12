# src/uae_law_rag/backend/pipelines/generation/persist.py

"""
[职责] generation persist：将生成结果写入 DB，形成可回放审计记录。
[边界] 不调用 LLM；不解析输出结构；不提交事务；仅做入参规范化与落库。
[上游关系] generation pipeline 产出 raw/output_structured/citations/messages_snapshot。
[下游关系] GenerationRepo 写入 GenerationRecordModel；evaluator/chat 读取记录回放。
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence, List

from uae_law_rag.backend.db.repo.generation_repo import GenerationRepo
from uae_law_rag.backend.schemas.generation import Citation, CitationsPayload


__all__ = ["persist_generation"]


def _coerce_str(value: Any) -> str:
    """
    [职责] 将 value 转为字符串（去空白）。
    [边界] 空值返回空字符串。
    [上游关系] _normalize_record_params 调用。
    [下游关系] record_params 字段规范化。
    """
    return str(value or "").strip()  # docstring: 字符串兜底


def _coerce_int(value: Any) -> Optional[int]:
    """
    [职责] 将 value 转为 int（失败返回 None）。
    [边界] 仅处理常见数值与数字字符串。
    [上游关系] citations 解析调用。
    [下游关系] citation.rank。
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


def _json_safe(value: Any) -> Any:
    """
    [职责] 将 value 转为可 JSON 序列化结构（dict/list/str/num/bool/None）。
    [边界] 不做业务解释；仅做类型降级与兜底。
    [上游关系] _normalize_record_params 调用。
    [下游关系] messages_snapshot/output_structured 落库稳定性。
    """
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in dict(value).items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_safe(v) for v in list(value)]
    if hasattr(value, "model_dump"):
        try:
            return _json_safe(value.model_dump())  # type: ignore[attr-defined]
        except Exception:
            return None
    if hasattr(value, "dict"):
        try:
            return _json_safe(value.dict())  # type: ignore[call-arg]
        except Exception:
            return None
    return None


def _citation_item_from_any(item: Any) -> Optional[Dict[str, Any]]:
    """
    [职责] 将任意 citation 输入归一化为 dict item。
    [边界] 不校验 node_id 存在性；仅做结构归一化。
    [上游关系] _normalize_citations 调用。
    [下游关系] CitationsPayload.items。
    """
    if isinstance(item, Citation):
        # docstring: 兼容 pydantic v2 / v1 / dataclass-like
        if hasattr(item, "model_dump"):
            raw = item.model_dump()  # type: ignore[attr-defined]
        elif hasattr(item, "dict"):
            raw = item.dict()  # type: ignore[call-arg]
        else:
            raw = dict(getattr(item, "__dict__", {}))
    elif isinstance(item, Mapping):
        raw = dict(item)  # docstring: mapping citation
    elif isinstance(item, str):
        raw = {"node_id": item}  # docstring: 字符串视为 node_id
    else:
        return None  # docstring: 不支持类型

    node_id = _coerce_str(raw.get("node_id") or raw.get("id") or raw.get("nodeId"))
    if not node_id:
        return None  # docstring: node_id 缺失丢弃

    rank = _coerce_int(raw.get("rank"))  # docstring: rank 解析
    quote = _coerce_str(raw.get("quote") or raw.get("excerpt"))  # docstring: quote 解析
    locator_raw = raw.get("locator")  # docstring: locator 原始值
    locator = dict(locator_raw) if isinstance(locator_raw, Mapping) else {}  # docstring: locator 兜底

    return {
        "node_id": node_id,
        "rank": rank,
        "quote": quote,
        "locator": locator,
    }  # docstring: 统一 citation item


def _normalize_citations(citations: Any) -> Dict[str, Any]:
    """
    [职责] 规范化 citations 为 CitationsPayload dict。
    [边界] 不做证据一致性校验；仅负责结构落库。
    [上游关系] persist_generation 调用。
    [下游关系] GenerationRepo.create_record 的 citations 入参。
    """
    if citations is None:
        return {}  # docstring: 无 citations 返回空 dict

    if isinstance(citations, CitationsPayload):
        # docstring: 兼容 pydantic v2 / v1
        if hasattr(citations, "model_dump"):
            return citations.model_dump()  # type: ignore[attr-defined]
        if hasattr(citations, "dict"):
            return citations.dict()  # type: ignore[call-arg]
        return dict(getattr(citations, "__dict__", {}))

    if isinstance(citations, Mapping):
        if any(k in citations for k in ("nodes", "items", "version")):
            return dict(citations)  # docstring: 已是 payload 结构
        item = _citation_item_from_any(citations)  # docstring: 单条 citation
        if item:
            return {"version": "v1", "nodes": [item["node_id"]], "items": [item], "meta": {}}  # docstring: 包装单条
        return {}  # docstring: 无有效 citation

    if isinstance(citations, Sequence) and not isinstance(citations, (str, bytes)):
        items: List[Dict[str, Any]] = []  # docstring: items 容器
        nodes: List[str] = []  # docstring: node_id 列表
        seen: set[str] = set()  # docstring: 去重缓存
        for item in citations:
            normalized = _citation_item_from_any(item)  # docstring: 解析 citation
            if not normalized:
                continue
            node_id = normalized["node_id"]
            if node_id in seen:
                continue  # docstring: 去重 node_id
            seen.add(node_id)  # docstring: 记录已见 node_id
            items.append(normalized)  # docstring: 收集 item
            nodes.append(node_id)  # docstring: 收集 node_id
        if not items:
            return {}  # docstring: 无有效 citations 返回空
        return {"version": "v1", "nodes": nodes, "items": items, "meta": {}}  # docstring: payload 组装

    return {}  # docstring: 未识别类型回退


def _normalize_record_params(record_params: Mapping[str, Any]) -> Dict[str, Any]:
    """
    [职责] 规范化 GenerationRecord 入参（必填字段校验 + 类型转换）。
    [边界] 仅做最小校验；业务策略由 pipeline 保证。
    [上游关系] persist_generation 调用。
    [下游关系] GenerationRepo.create_record。
    """
    required = [
        "message_id",
        "retrieval_record_id",
        "prompt_name",
        "model_provider",
        "model_name",
        "output_raw",
    ]
    missing = [k for k in required if k not in record_params]
    if missing:
        raise ValueError(f"record_params missing: {', '.join(missing)}")  # docstring: 必填字段缺失

    def _require_nonempty(key: str) -> str:
        val = _coerce_str(record_params.get(key))
        if not val:
            raise ValueError(f"record_params empty: {key}")  # docstring: 必填字段不可为空
        return val

    params: Dict[str, Any] = {
        "message_id": _require_nonempty("message_id"),  # docstring: 归属 message
        "retrieval_record_id": _require_nonempty("retrieval_record_id"),  # docstring: 归属检索记录
        "prompt_name": _require_nonempty("prompt_name"),  # docstring: prompt 名称
        "model_provider": _require_nonempty("model_provider"),  # docstring: provider
        "model_name": _require_nonempty("model_name"),  # docstring: 模型名
        "output_raw": _require_nonempty("output_raw"),  # docstring: 原始输出
        "messages_snapshot": _json_safe(record_params.get("messages_snapshot") or {}),  # docstring: messages 快照
        "output_structured": _json_safe(record_params.get("output_structured")),  # docstring: 结构化输出
        "citations": _normalize_citations(record_params.get("citations")),  # docstring: citations 结构
        "prompt_version": record_params.get("prompt_version"),  # docstring: prompt 版本
        "status": _coerce_str(record_params.get("status") or "success") or "success",  # docstring: 状态
        "error_message": record_params.get("error_message"),  # docstring: 错误信息
    }
    return params


async def persist_generation(
    *,
    generation_repo: GenerationRepo,
    record_params: Mapping[str, Any],
) -> str:
    """
    [职责] persist_generation：写入 GenerationRecord，返回 record_id。
    [边界] 不提交事务；不重排 citations；仅按输入落库。
    [上游关系] generation pipeline 产出 record_params。
    [下游关系] GenerationRepo.create_record 写入 DB。
    """
    params = _normalize_record_params(record_params)  # docstring: 规范化入参
    record = await generation_repo.create_record(**params)  # docstring: 落库 generation_record
    return record.id  # docstring: 返回生成记录 ID
