# src/uae_law_rag/backend/utils/logging_.py

"""
[职责] 定义结构化日志字段规范与统一 logger 获取方式，提供最小 JSON 格式化与安全输出 helper。
[边界] 不绑定具体日志后端；不记录业务日志；不强制 trace_id 注入，仅提供工具。
[上游关系] services/pipelines/api 通过 get_logger/build_log_fields 组织日志上下文。
[下游关系] 日志后端（stdout/file/otlp）或审计系统消费结构化字段做检索与排障。
"""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, Mapping, Optional


DEFAULT_LOGGER_NAME = "uae_law_rag"  # docstring: 统一 logger 根名称
DEFAULT_LOG_LEVEL = logging.INFO  # docstring: 默认日志级别
DEFAULT_MAX_TEXT_LEN = 160  # docstring: 安全文本预览长度

TRACE_FIELD_KEYS = (
    "trace_id",
    "request_id",
    "parent_request_id",
    "conversation_id",
    "message_id",
    "retrieval_record_id",
    "generation_record_id",
    "evaluation_record_id",
)  # docstring: 推荐结构化日志字段

_LOG_RECORD_RESERVED = {
    "name",
    "msg",
    "args",
    "levelname",
    "levelno",
    "pathname",
    "filename",
    "module",
    "exc_info",
    "exc_text",
    "stack_info",
    "lineno",
    "funcName",
    "created",
    "msecs",
    "relativeCreated",
    "thread",
    "threadName",
    "processName",
    "process",
    "message",
    "asctime",
}  # docstring: LogRecord 内置字段（不作为结构化额外字段）


class StructuredLogFormatter(logging.Formatter):
    """
    [职责] 将 LogRecord 转换为 JSON 字符串（含结构化字段）。
    [边界] 不保证字段全量；仅输出基础字段 + extra。
    [上游关系] get_logger/configure_logging 创建 handler 后挂载。
    [下游关系] 日志收集系统解析 JSON 或 grep 关键字段。
    """

    def __init__(self, *, ensure_ascii: bool = True) -> None:
        """
        [职责] 初始化 JSON formatter。
        [边界] 仅控制 JSON 输出；不控制 handler/level。
        [上游关系] configure_logging 调用。
        [下游关系] handler.format(record) 调用。
        """

        super().__init__()
        self._ensure_ascii = ensure_ascii  # docstring: 保持 ASCII 输出，便于终端/存储兼容

    def format(self, record: logging.LogRecord) -> str:
        """
        [职责] 输出结构化 JSON 日志字符串。
        [边界] 不做敏感字段识别；由调用方避免记录 raw 文本。
        [上游关系] logging.Handler.format 调用。
        [下游关系] 下游日志系统解析 JSON 字段。
        """

        payload: Dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=timezone.utc).isoformat(),  # docstring: 统一 UTC 时间戳
            "level": record.levelname,  # docstring: 日志级别
            "logger": record.name,  # docstring: logger 名称
            "message": record.getMessage(),  # docstring: 纯文本消息
        }

        extras = {
            k: v for k, v in record.__dict__.items() if k not in _LOG_RECORD_RESERVED
        }  # docstring: 仅保留 extra 字段
        for key, value in list(extras.items()):
            if value is None:
                extras.pop(key)  # docstring: 去除 None 值以降低噪声
        payload.update(extras)  # docstring: 合并结构化字段

        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)  # docstring: 异常堆栈文本
        if record.stack_info:
            payload["stack_info"] = record.stack_info  # docstring: 显式 stack_info

        return json.dumps(payload, ensure_ascii=self._ensure_ascii, default=str)  # docstring: JSON 输出


def configure_logging(
    *,
    logger_name: str = DEFAULT_LOGGER_NAME,
    level: int = DEFAULT_LOG_LEVEL,
    ensure_ascii: bool = True,
) -> logging.Logger:
    """
    [职责] 配置统一的 base logger（JSON formatter）。
    [边界] 不触碰 root logger；不添加外部 handler。
    [上游关系] 进程入口或测试初始化时调用。
    [下游关系] get_logger 复用已配置的 base logger。
    """

    logger = logging.getLogger(logger_name)
    logger.setLevel(level)  # docstring: 设置 base logger 级别

    has_handler = any(
        isinstance(h, logging.StreamHandler) and getattr(h, "name", "") == "structured_json" for h in logger.handlers
    )
    if not has_handler:
        handler = logging.StreamHandler()
        handler.name = "structured_json"  # docstring: 标记 handler，避免重复挂载
        handler.setLevel(level)  # docstring: handler 级别与 base 对齐
        handler.setFormatter(StructuredLogFormatter(ensure_ascii=ensure_ascii))
        logger.addHandler(handler)  # docstring: 添加 JSON handler

    logger.propagate = False  # docstring: 避免重复向 root 传播
    return logger


def get_logger(name: Optional[str] = None, *, level: Optional[int] = None) -> logging.Logger:
    """
    [职责] 获取项目统一 logger（自动确保 base logger 已配置）。
    [边界] 不强制覆写外部 logging 配置；仅保证本项目 logger 可用。
    [上游关系] services/pipelines/api 调用获取 logger。
    [下游关系] logger 输出 JSON 格式结构化日志。
    """

    configure_logging()  # docstring: 确保 base logger 就绪
    full_name = name or DEFAULT_LOGGER_NAME
    if name and not name.startswith(DEFAULT_LOGGER_NAME):
        full_name = f"{DEFAULT_LOGGER_NAME}.{name}"  # docstring: 统一挂载在项目根 logger 下
    logger = logging.getLogger(full_name)
    if level is not None:
        logger.setLevel(level)  # docstring: 允许调用方自定义级别
    return logger


def build_log_fields(
    *,
    context: Optional[Any] = None,
    trace_id: Optional[str] = None,
    request_id: Optional[str] = None,
    parent_request_id: Optional[str] = None,
    conversation_id: Optional[str] = None,
    message_id: Optional[str] = None,
    retrieval_record_id: Optional[str] = None,
    generation_record_id: Optional[str] = None,
    evaluation_record_id: Optional[str] = None,
    extra: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """
    [职责] 统一构建结构化日志字段（trace/request/record ids 等）。
    [边界] 不生成缺失 trace_id；不校验字段合法性。
    [上游关系] services/pipelines 传入 ctx 或显式 IDs。
    [下游关系] logger.extra 供 StructuredLogFormatter 输出。
    """

    fields: Dict[str, Any] = {}
    if context is not None:
        fields.update(_extract_fields_from_context(context))  # docstring: 从上下文提取 trace 字段

    explicit_fields = {
        "trace_id": trace_id,
        "request_id": request_id,
        "parent_request_id": parent_request_id,
        "conversation_id": conversation_id,
        "message_id": message_id,
        "retrieval_record_id": retrieval_record_id,
        "generation_record_id": generation_record_id,
        "evaluation_record_id": evaluation_record_id,
    }
    for key, value in explicit_fields.items():
        if value is not None:
            fields[key] = str(value)  # docstring: 统一转为字符串便于落盘

    if extra:
        for key, value in extra.items():
            if value is not None:
                fields[key] = value  # docstring: 附加扩展字段

    return fields


def log_event(
    logger: logging.Logger,
    level: int,
    message: str,
    *,
    context: Optional[Any] = None,
    fields: Optional[Mapping[str, Any]] = None,
    exc_info: Optional[Any] = None,
) -> None:
    """
    [职责] 统一记录结构化日志（可自动附加 trace 字段）。
    [边界] 不处理业务语义；不强制 trace_id。
    [上游关系] services/pipelines 在关键节点调用。
    [下游关系] StructuredLogFormatter 输出 JSON。
    """

    extra = build_log_fields(context=context, extra=fields)  # docstring: 合成结构化字段
    logger.log(level, message, extra=extra, exc_info=exc_info)  # docstring: 统一入口写日志


def truncate_text(text: Optional[str], *, max_len: int = DEFAULT_MAX_TEXT_LEN) -> Optional[str]:
    """
    [职责] 截断长文本（避免记录原始输入全文）。
    [边界] 不做敏感识别；仅长度控制。
    [上游关系] services/pipelines 在日志前调用。
    [下游关系] 结构化日志输出安全预览。
    """

    if text is None:
        return None  # docstring: 空值直接返回
    s = str(text)
    if len(s) <= max_len:
        return s  # docstring: 短文本原样返回
    return f"{s[:max_len]}...(truncated)"  # docstring: 长文本截断


def hash_text(text: Optional[str]) -> Optional[str]:
    """
    [职责] 生成文本 sha256 摘要（避免记录原文）。
    [边界] 不提供盐；不作为安全认证。
    [上游关系] services/pipelines 记录用户输入/证据摘要时调用。
    [下游关系] 日志与审计系统用于去重/定位。
    """

    if text is None:
        return None  # docstring: 空值直接返回
    s = str(text)
    if not s:
        return ""  # docstring: 保留空字符串语义
    return hashlib.sha256(s.encode("utf-8")).hexdigest()  # docstring: 生成稳定摘要


def _extract_fields_from_context(context: Any) -> Dict[str, Any]:
    """
    [职责] 从任意上下文对象/映射中提取标准 trace 字段。
    [边界] 仅读取预定义字段；不解析嵌套对象。
    [上游关系] build_log_fields 调用。
    [下游关系] 作为 logger.extra 结构化字段。
    """

    fields: Dict[str, Any] = {}
    for key in TRACE_FIELD_KEYS:
        value = _read_context_value(context, key)
        if value is not None:
            fields[key] = str(value)  # docstring: 统一转为字符串输出
    return fields


def _read_context_value(context: Any, key: str) -> Optional[Any]:
    """
    [职责] 从 context 中安全读取字段值（支持 dict/对象属性）。
    [边界] 不抛异常；读取不到返回 None。
    [上游关系] _extract_fields_from_context 调用。
    [下游关系] 结构化字段提取。
    """

    if isinstance(context, Mapping):
        return context.get(key)  # docstring: dict-like 读取
    return getattr(context, key, None)  # docstring: 对象属性读取


def iter_trace_fields(fields: Mapping[str, Any]) -> Dict[str, Any]:
    """
    [职责] 过滤并输出仅包含 trace 字段的字典。
    [边界] 不校验字段类型；仅做过滤。
    [上游关系] 日志/HTTP 响应需要透传 trace 字段时调用。
    [下游关系] 输出稳定字段集合。
    """

    return {k: fields[k] for k in TRACE_FIELD_KEYS if k in fields}  # docstring: 过滤 trace 字段
