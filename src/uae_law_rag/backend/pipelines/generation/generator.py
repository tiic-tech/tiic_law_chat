# src/uae_law_rag/backend/pipelines/generation/generator.py

"""
[职责] generation generator：基于 LlamaIndex LLM 抽象执行模型调用，产出 raw 文本与使用快照。
[边界] 不解析 JSON；不对 citations 做校验；不负责落库与编排。
[上游关系] generation pipeline 传入 messages_snapshot + provider/model + generation_config。
[下游关系] postprocess 解析输出；persist 写入 GenerationRecordModel.output_raw。
"""

from __future__ import annotations

import inspect
import json
from inspect import Parameter
from typing import Any, Dict, List, Mapping, Optional, Sequence


__all__ = ["run_generation"]


def _load_llama_index() -> Dict[str, Any]:
    """
    [职责] 延迟加载 LlamaIndex LLM 相关类型（LLM/ChatMessage/MessageRole/Response）。
    [边界] 仅负责 import；不执行任何模型逻辑。
    [上游关系] _resolve_llm/_build_chat_messages 调用。
    [下游关系] LLM 构造与消息对象构建。
    """
    try:
        from llama_index.core.llms import LLM  # type: ignore  # docstring: LLM 抽象
    except Exception as exc:  # pragma: no cover - 依赖缺失场景
        raise ImportError("llama_index is required for generation") from exc  # docstring: 强制依赖

    ChatMessage = None
    MessageRole = None
    ChatResponse = None
    CompletionResponse = None
    LLMMetadata = None

    try:
        from llama_index.core.llms import ChatMessage, MessageRole  # type: ignore  # docstring: 消息类型
    except Exception:
        try:
            from llama_index.core.base.llms.types import (  # type: ignore
                ChatMessage,
                MessageRole,
            )  # docstring: 兼容导入路径
        except Exception:
            ChatMessage = None
            MessageRole = None

    try:
        from llama_index.core.llms import ChatResponse, CompletionResponse  # type: ignore  # docstring: 响应类型
    except Exception:
        try:
            from llama_index.core.base.llms.types import (  # type: ignore
                ChatResponse,
                CompletionResponse,
            )  # docstring: 兼容导入路径
        except Exception:
            ChatResponse = None
            CompletionResponse = None

    try:
        from llama_index.core.base.llms.types import LLMMetadata  # type: ignore  # docstring: metadata 类型
    except Exception:
        LLMMetadata = None

    return {
        "LLM": LLM,
        "ChatMessage": ChatMessage,
        "MessageRole": MessageRole,
        "ChatResponse": ChatResponse,
        "CompletionResponse": CompletionResponse,
        "LLMMetadata": LLMMetadata,
    }


def _filter_kwargs(fn: Any, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """
    [职责] 过滤参数，仅保留目标函数支持的关键字。
    [边界] 不做值校验；仅做参数名过滤。
    [上游关系] _resolve_llm/_call_llm 调用。
    [下游关系] LLM 构造与调用参数。
    """
    try:
        sig = inspect.signature(fn)  # docstring: 读取可用参数
    except (TypeError, ValueError):
        return {}  # docstring: 无签名时回退为空
    # docstring: 若函数支持 **kwargs，则允许透传全部配置（避免静默丢参）
    for p in sig.parameters.values():
        if p.kind == Parameter.VAR_KEYWORD:
            return dict(kwargs)
    return {k: v for k, v in kwargs.items() if k in sig.parameters}  # docstring: 保留受支持参数


def _normalize_generation_config(*, provider_key: str, cfg: Mapping[str, Any]) -> Dict[str, Any]:
    """
    [职责] generation_config 归一化：为特定 provider 注入确定性默认值。
    [边界] 仅做缺省填充；不覆盖用户显式配置。
    [上游关系] _resolve_llm/_call_llm 调用。
    [下游关系] LLM init/chat kwargs。
    """
    out = dict(cfg or {})
    if provider_key == "ollama":
        # docstring: 确定性输出优先，避免 JSON 格式漂移
        out.setdefault("temperature", 0)
        out.setdefault("top_p", 1)
        out.setdefault("num_predict", 256)
    return out


def _normalize_provider(provider: str) -> str:
    """
    [职责] 规范化 provider 字符串。
    [边界] 仅做小写与去空白处理。
    [上游关系] run_generation 调用。
    [下游关系] _resolve_llm 选择 provider。
    """
    return str(provider or "").strip().lower()  # docstring: provider 归一化


def _normalize_model_name(model_name: str) -> str:
    """
    [职责] 规范化 model_name 字符串。
    [边界] 仅去空白；不做别名映射。
    [上游关系] run_generation 调用。
    [下游关系] _resolve_llm 构造 LLM。
    """
    return str(model_name or "").strip()  # docstring: model_name 归一化


def _build_mock_response(messages_snapshot: Mapping[str, Any]) -> str:
    """
    [职责] 基于 messages_snapshot 构造确定性 JSON 响应（用于 mock/local）。
    [边界] 不访问 DB；仅使用 evidence 快照；不保证法律结论正确。
    [上游关系] _build_mock_llm 调用。
    [下游关系] MockLLM 输出内容。
    """
    evidence = messages_snapshot.get("evidence") or []  # docstring: evidence 快照
    citations: List[Dict[str, Any]] = []  # docstring: citations 列表

    def _coerce_int(value: Any) -> Optional[int]:
        if value is None:
            return None
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return int(value)
        if isinstance(value, str) and value.strip().isdigit():
            return int(value.strip())
        return None

    for item in evidence[:2]:
        node_id = str(item.get("node_id") or "").strip()
        if not node_id:
            continue
        coerced_rank = _coerce_int(item.get("rank"))
        citations.append(
            {
                "node_id": node_id,
                "rank": int(coerced_rank) if coerced_rank is not None else (len(citations) + 1),
                "quote": str(item.get("excerpt") or "")[:200],
            }
        )  # docstring: 基于证据生成引用

    if citations:
        answer = "Based on the provided evidence, see citations."  # docstring: 有证据回答
    else:
        answer = "The provided evidence is insufficient to answer this question."  # docstring: 无证据拒答

    payload = {"answer": answer, "citations": citations}  # docstring: JSON payload
    return json.dumps(payload, ensure_ascii=True)  # docstring: 生成 JSON 字符串


def _build_mock_llm(*, response_text: str, model_name: str) -> Any:
    """
    [职责] 构造 LlamaIndex Mock LLM（用于离线/测试）。
    [边界] 依赖 LlamaIndex Mock 实现；若缺失则抛错。
    [上游关系] _resolve_llm 调用。
    [下游关系] _call_llm 调用 LLM 生成响应。
    """
    try:
        from llama_index.core.llms import MockLLM  # type: ignore  # docstring: MockLLM
    except Exception:
        try:
            from llama_index.core.llms.mock import MockLLM  # type: ignore  # docstring: 兼容路径
        except Exception as exc:  # pragma: no cover - 依赖缺失场景
            raise ImportError("llama_index MockLLM is required for mock provider") from exc

    kwargs = {
        "response": response_text,
        "responses": [response_text],
        "model_name": model_name,
    }  # docstring: MockLLM 参数快照
    return MockLLM(**_filter_kwargs(MockLLM.__init__, kwargs))  # docstring: 构造 MockLLM


def _resolve_llm(
    *,
    provider: str,
    model_name: str,
    generation_config: Optional[Mapping[str, Any]] = None,
    messages_snapshot: Optional[Mapping[str, Any]] = None,
) -> Any:
    """
    [职责] 根据 provider/model 构造 LlamaIndex LLM 实例。
    [边界] 仅支持已接入 provider；未知 provider 抛错。
    [上游关系] run_generation 调用。
    [下游关系] _call_llm 使用返回的 LLM。
    """
    provider_key = _normalize_provider(provider)  # docstring: provider 归一化
    model = _normalize_model_name(model_name)  # docstring: model_name 归一化
    cfg = dict(generation_config or {})  # docstring: generation_config 透传

    # docstring: Ollama 默认超时较短（常见为 30s）；在本地小模型+长 evidence 时容易 ReadTimeout。
    # LlamaIndex Ollama 支持 request_timeout 参数，用于提升稳定性。
    if provider_key == "ollama":
        cfg.setdefault("request_timeout", 300.0)

    if provider_key in {"mock", "local", "hash"}:
        response_text = _build_mock_response(messages_snapshot or {})  # docstring: 构造 mock 响应
        return _build_mock_llm(response_text=response_text, model_name=model or "mock")  # docstring: MockLLM
    if provider_key == "ollama":
        from llama_index.llms.ollama import Ollama  # type: ignore  # docstring: Ollama LLM

        kwargs = {"model": model, "model_name": model, **cfg}  # docstring: Ollama 参数快照
        return Ollama(**_filter_kwargs(Ollama.__init__, kwargs))  # docstring: 构造 Ollama LLM
    if provider_key == "openai":
        from llama_index.llms.openai import OpenAI  # type: ignore  # docstring: OpenAI LLM

        kwargs = {"model": model, "model_name": model, **cfg}  # docstring: OpenAI 参数快照
        return OpenAI(**_filter_kwargs(OpenAI.__init__, kwargs))  # docstring: 构造 OpenAI LLM
    if provider_key in {"dashscope", "qwen"}:
        from llama_index.llms.dashscope import DashScope  # type: ignore  # docstring: DashScope LLM

        # docstring: 关键兜底：避免 JSON 输出被截断（缺失右括号/右花括号）
        # DashScope LLM 支持 max_tokens 字段。 [oai_citation:1‡developers.llamaindex.ai](https://developers.llamaindex.ai/python/framework-api-reference/llms/dashscope/)
        try:
            mt = int(cfg.get("max_tokens") or 0)
        except Exception:
            mt = 0
        if mt <= 0:
            cfg["max_tokens"] = 2048  # docstring: 默认给足输出空间（你现在输出包含 citations 数组）
        elif mt < 1024:
            cfg["max_tokens"] = 1024  # docstring: 过小则抬高，减少截断概率

        # docstring: 降低随机性，提升“结构化 JSON”稳定性
        cfg.setdefault("temperature", 0.1)

        dashscope_api_key = cfg.pop("api_key", None)  # docstring: 避免把 API key 写入快照
        if not dashscope_api_key:
            from uae_law_rag.config import settings  # docstring: 延迟加载 settings 读取 .env

            dashscope_api_key = settings.DASHSCOPE_API_KEY
        kwargs = {"model": model, "model_name": model, **cfg}  # docstring: DashScope 参数快照
        if dashscope_api_key:
            kwargs["api_key"] = str(dashscope_api_key)
        return DashScope(**_filter_kwargs(DashScope.__init__, kwargs))  # docstring: 构造 DashScope LLM
    if provider_key in {"huggingface", "hf"}:
        from llama_index.llms.huggingface import HuggingFaceLLM  # type: ignore  # docstring: HF LLM

        kwargs = {"model_name": model, "model": model, **cfg}  # docstring: HF 参数快照
        return HuggingFaceLLM(**_filter_kwargs(HuggingFaceLLM.__init__, kwargs))  # docstring: 构造 HF LLM
    if provider_key in {"deepseek"}:
        try:
            from llama_index.llms.deepseek import DeepSeek  # type: ignore  # docstring: DeepSeek LLM

            kwargs = {"model": model, "model_name": model, **cfg}  # docstring: DeepSeek 参数快照
            return DeepSeek(**_filter_kwargs(DeepSeek.__init__, kwargs))  # docstring: 构造 DeepSeek LLM
        except Exception:
            from llama_index.llms.openai_like import OpenAILike  # type: ignore  # docstring: OpenAI-like LLM

            kwargs = {"model": model, "model_name": model, **cfg}  # docstring: OpenAI-like 参数快照
            return OpenAILike(**_filter_kwargs(OpenAILike.__init__, kwargs))  # docstring: 构造 OpenAI-like
    if provider_key in {"openai_like", "openai-like"}:
        from llama_index.llms.openai_like import OpenAILike  # type: ignore  # docstring: OpenAI-like LLM

        kwargs = {"model": model, "model_name": model, **cfg}  # docstring: OpenAI-like 参数快照
        return OpenAILike(**_filter_kwargs(OpenAILike.__init__, kwargs))  # docstring: 构造 OpenAI-like

    raise ValueError(f"unsupported model provider: {provider}")  # docstring: 未接入 provider


def _resolve_role(role: str, message_role: Any) -> Any:
    """
    [职责] 解析消息角色为 LlamaIndex MessageRole。
    [边界] 未知角色回退为 user。
    [上游关系] _build_chat_messages 调用。
    [下游关系] ChatMessage.role。
    """
    raw = str(role or "").strip().lower()  # docstring: role 归一化
    if message_role is None:
        return raw or "user"  # docstring: 无 MessageRole 时回退字符串
    for key in ("SYSTEM", "USER", "ASSISTANT", "TOOL"):
        if raw == key.lower():
            return getattr(message_role, key)  # docstring: 命中标准角色
    try:
        return message_role(raw)  # docstring: 尝试枚举构造
    except Exception:
        return getattr(message_role, "USER", raw or "user")  # docstring: 未知角色回退


def _build_chat_messages(messages: Sequence[Mapping[str, Any]]) -> List[Any]:
    """
    [职责] 将消息 dict 列表转换为 LlamaIndex ChatMessage 列表。
    [边界] 仅处理 role/content；其他字段忽略。
    [上游关系] run_generation 调用。
    [下游关系] _call_llm 输入消息列表。
    """
    li = _load_llama_index()  # docstring: 加载 LlamaIndex 类型
    ChatMessage = li["ChatMessage"]
    MessageRole = li["MessageRole"]
    if ChatMessage is None:
        return list(messages)  # docstring: 无 ChatMessage 时回退原始消息

    out: List[Any] = []  # docstring: ChatMessage 容器
    for msg in messages:
        role = _resolve_role(msg.get("role", "user"), MessageRole)  # docstring: 解析角色
        content = str(msg.get("content") or "")  # docstring: 读取内容
        out.append(ChatMessage(role=role, content=content))  # docstring: 构造 ChatMessage
    return out


def _normalize_messages_snapshot(messages_snapshot: Mapping[str, Any]) -> List[Mapping[str, Any]]:
    """
    [职责] 从 messages_snapshot 解析消息列表。
    [边界] 不做 prompt 拼接；只提供 messages 结构。
    [上游关系] run_generation 调用。
    [下游关系] _build_chat_messages。
    """
    raw_messages = messages_snapshot.get("messages")  # docstring: 读取 messages
    if isinstance(raw_messages, list) and raw_messages:
        return raw_messages  # docstring: 直接返回 messages 列表

    system = messages_snapshot.get("system")  # docstring: 读取 system 字段
    user = messages_snapshot.get("user")  # docstring: 读取 user 字段
    fallback: List[Mapping[str, Any]] = []
    if system:
        fallback.append({"role": "system", "content": system})  # docstring: 构造 system 消息
    if user:
        fallback.append({"role": "user", "content": user})  # docstring: 构造 user 消息
    return fallback


def _message_text(msg: Any) -> str:
    """
    [职责] 读取单条消息的文本内容。
    [边界] 仅提取 content/text；不做格式化。
    [上游关系] _messages_to_prompt 调用。
    [下游关系] prompt 拼接。
    """
    if isinstance(msg, Mapping):
        return str(msg.get("content") or msg.get("text") or "")  # docstring: dict 内容
    if hasattr(msg, "content"):
        return str(getattr(msg, "content") or "")  # docstring: ChatMessage 内容
    return str(msg or "")  # docstring: 兜底转换


def _message_role(msg: Any) -> str:
    """
    [职责] 读取单条消息的角色。
    [边界] 未知角色回退为空字符串。
    [上游关系] _messages_to_prompt 调用。
    [下游关系] prompt 拼接。
    """
    if isinstance(msg, Mapping):
        return str(msg.get("role") or "")  # docstring: dict 角色
    if hasattr(msg, "role"):
        return str(getattr(msg, "role") or "")  # docstring: ChatMessage 角色
    return ""


def _messages_to_prompt(messages: Sequence[Any]) -> str:
    """
    [职责] 将 chat messages 合成为文本 prompt（用于非 chat LLM）。
    [边界] 简单拼接；不做 token 控制。
    [上游关系] _call_llm 在无 chat 接口时调用。
    [下游关系] LLM.complete 输入。
    """
    lines: List[str] = []  # docstring: prompt 行
    for msg in messages:
        role = _message_role(msg).upper()  # docstring: 角色文本
        content = _message_text(msg)  # docstring: 内容文本
        if role:
            lines.append(f"{role}:\n{content}")  # docstring: 带角色前缀
        else:
            lines.append(content)  # docstring: 无角色直接拼接
    return "\n\n".join(lines).strip()  # docstring: 合并为 prompt


async def _call_llm(
    *,
    llm: Any,
    messages: Sequence[Any],
    generation_config: Optional[Mapping[str, Any]] = None,
) -> Any:
    """
    [职责] 调用 LLM（优先 chat，其次 complete）。
    [边界] 不解析输出；仅返回原始响应对象。
    [上游关系] run_generation 调用。
    [下游关系] _extract_text/_extract_usage。
    """
    cfg = dict(generation_config or {})  # docstring: 调用配置透传
    if hasattr(llm, "achat"):
        kwargs = _filter_kwargs(llm.achat, cfg)  # docstring: 过滤 achat 参数
        return await llm.achat(messages, **kwargs)  # docstring: 异步 chat 调用
    if hasattr(llm, "chat"):
        kwargs = _filter_kwargs(llm.chat, cfg)  # docstring: 过滤 chat 参数
        return llm.chat(messages, **kwargs)  # docstring: 同步 chat 调用

    prompt = _messages_to_prompt(messages)  # docstring: chat -> prompt
    if hasattr(llm, "acomplete"):
        kwargs = _filter_kwargs(llm.acomplete, cfg)  # docstring: 过滤 acomplete 参数
        return await llm.acomplete(prompt, **kwargs)  # docstring: 异步 completion 调用
    if hasattr(llm, "complete"):
        kwargs = _filter_kwargs(llm.complete, cfg)  # docstring: 过滤 complete 参数
        return llm.complete(prompt, **kwargs)  # docstring: 同步 completion 调用

    raise AttributeError("LLM instance missing chat/complete interfaces")  # docstring: 强约束


def _extract_text(response: Any) -> str:
    """
    [职责] 从 LLM 响应中提取文本内容。
    [边界] 只做字段探测；不做 JSON 解析。
    [上游关系] run_generation 调用。
    [下游关系] GenerationRawResult.raw_text。
    """
    if response is None:
        return ""  # docstring: 空响应兜底
    if hasattr(response, "message"):
        msg = getattr(response, "message")  # docstring: ChatResponse.message
        if msg is not None and hasattr(msg, "content"):
            return str(getattr(msg, "content") or "")  # docstring: ChatMessage.content
    if hasattr(response, "response"):
        return str(getattr(response, "response") or "")  # docstring: Response.response
    if hasattr(response, "text"):
        return str(getattr(response, "text") or "")  # docstring: CompletionResponse.text
    raw = getattr(response, "raw", None)
    if isinstance(raw, str):
        return raw  # docstring: raw 字符串
    if isinstance(raw, Mapping):
        for key in ("text", "content", "response", "output"):
            val = raw.get(key)
            if isinstance(val, str) and val.strip():
                return val  # docstring: raw dict 字段
    return str(response)  # docstring: 兜底字符串


def _extract_usage(response: Any) -> Optional[Dict[str, Any]]:
    """
    [职责] 从 LLM 响应中提取 usage/计费快照。
    [边界] 仅探测常见字段；不保证存在。
    [上游关系] run_generation 调用。
    [下游关系] GenerationRawResult.usage。
    """
    if response is None:
        return None  # docstring: 空响应无 usage
    for attr in ("usage", "token_usage"):
        val = getattr(response, attr, None)
        if isinstance(val, Mapping):
            return dict(val)  # docstring: response.usage
    raw = getattr(response, "raw", None)
    if isinstance(raw, Mapping) and isinstance(raw.get("usage"), Mapping):
        return dict(raw.get("usage") or {})  # docstring: raw.usage
    extra = getattr(response, "additional_kwargs", None)
    if isinstance(extra, Mapping) and isinstance(extra.get("usage"), Mapping):
        return dict(extra.get("usage") or {})  # docstring: additional_kwargs.usage
    return None


def _resolve_output_model(llm: Any, fallback: str) -> str:
    """
    [职责] 从 LLM metadata 中解析模型名。
    [边界] metadata 不存在时回退输入值。
    [上游关系] run_generation 调用。
    [下游关系] GenerationRawResult.model。
    """
    meta = getattr(llm, "metadata", None)
    name = getattr(meta, "model_name", None) if meta is not None else None
    return str(name or fallback or "")  # docstring: 模型名回退


async def run_generation(
    *,
    messages_snapshot: Mapping[str, Any],
    model_provider: str,
    model_name: str,
    generation_config: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """
    [职责] 执行 LLM 生成并返回 raw_text/provider/model/usage 快照。
    [边界] 不解析 JSON；不校验 citations；不落库。
    [上游关系] generation pipeline 传入 messages_snapshot 与模型配置。
    [下游关系] postprocess/persist 消费返回结果。
    """
    if not isinstance(messages_snapshot, Mapping):
        raise TypeError("messages_snapshot must be a mapping")  # docstring: 强制类型

    provider_key = _normalize_provider(model_provider)
    cfg = _normalize_generation_config(provider_key=provider_key, cfg=(generation_config or {}))

    messages_payload = _normalize_messages_snapshot(messages_snapshot)  # docstring: 解析 messages 列表
    if not messages_payload:
        raise ValueError("messages_snapshot missing messages")  # docstring: messages 必填

    llm = _resolve_llm(
        provider=model_provider,
        model_name=model_name,
        generation_config=cfg,
        messages_snapshot=messages_snapshot,
    )  # docstring: 构造 LLM

    chat_messages = _build_chat_messages(messages_payload)  # docstring: 构造 ChatMessage 列表
    response = await _call_llm(llm=llm, messages=chat_messages, generation_config=cfg)  # docstring: 执行 LLM

    raw_text = _extract_text(response)  # docstring: 提取 raw_text
    usage = _extract_usage(response)  # docstring: 提取 usage
    output_model = _resolve_output_model(llm, _normalize_model_name(model_name))  # docstring: 解析模型名
    output_provider = _normalize_provider(model_provider)  # docstring: provider 归一化

    return {
        "raw_text": str(raw_text or ""),  # docstring: LLM 原始输出
        "provider": output_provider,  # docstring: provider 快照
        "generation_config": cfg,  # docstring: 新增 generation_config 快照
        "model": output_model,  # docstring: model 快照
        "usage": usage,  # docstring: usage 快照
    }
