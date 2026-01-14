# src/uae_law_rag/backend/api/errors.py

"""
[职责] API 错误映射：将异常统一转换为 ErrorResponse 与 HTTP status。
[边界] 不记录日志；不负责 trace/request 注入（由 middleware/deps 负责）。
[上游关系] routers 捕获异常后调用本模块。
[下游关系] 返回 ErrorResponse 供前端/UI/审计系统消费。
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

from fastapi.responses import JSONResponse

from uae_law_rag.backend.api.schemas_http._common import ErrorResponse
from uae_law_rag.backend.schemas.ids import new_uuid
from uae_law_rag.backend.utils.errors import to_http_error

_TRACE_HEADER = "x-trace-id"  # docstring: trace header 约定
_REQUEST_HEADER = "x-request-id"  # docstring: request header 约定


def _ensure_trace_id(trace_id: Optional[str]) -> str:
    """
    [职责] 确保 trace_id 存在（空值回退生成）。
    [边界] 仅返回字符串；不写入外部上下文。
    [上游关系] to_error_response / to_json_response 调用。
    [下游关系] ErrorResponse.error.trace_id 必须存在。
    """
    raw = str(trace_id or "").strip()
    if raw:
        return raw  # docstring: 保留上游 trace_id
    return str(new_uuid())  # docstring: 无 trace_id 时生成兜底


def to_error_response(
    error: Exception,
    *,
    trace_id: Optional[str] = None,
) -> Tuple[int, ErrorResponse]:
    """
    [职责] 将异常转换为 (status_code, ErrorResponse)。
    [边界] 不注入 request_id；不写 header；不记录日志。
    [上游关系] routers 捕获异常后调用。
    [下游关系] routers 使用 ErrorResponse 返回 HTTP 错误体。
    """
    resolved_trace_id = _ensure_trace_id(trace_id)  # docstring: 保证 trace_id 存在
    status_code, payload = to_http_error(error, trace_id=resolved_trace_id)  # docstring: 领域错误映射
    response = ErrorResponse.model_validate(payload)  # docstring: 校验 ErrorResponse 结构
    return status_code, response  # docstring: 返回 HTTP status + schema


def to_json_response(
    error: Exception,
    *,
    trace_id: Optional[str] = None,
    request_id: Optional[str] = None,
) -> JSONResponse:
    """
    [职责] 将异常转换为 JSONResponse（含 header 透传）。
    [边界] 不做日志记录；不修改 error 语义。
    [上游关系] routers 捕获异常后调用。
    [下游关系] FastAPI 直接返回该响应对象。
    """
    status_code, response = to_error_response(error, trace_id=trace_id)  # docstring: 构建错误响应
    if hasattr(response, "model_dump"):
        content: Dict[str, Any] = response.model_dump()  # type: ignore[assignment]  # docstring: pydantic v2 序列化
    else:
        content = response.dict()  # type: ignore[assignment]  # docstring: 兼容 pydantic v1

    headers: Dict[str, str] = {}
    if trace_id:
        headers[_TRACE_HEADER] = str(trace_id)  # docstring: 回写 trace_id header
    if request_id:
        headers[_REQUEST_HEADER] = str(request_id)  # docstring: 回写 request_id header

    return JSONResponse(status_code=status_code, content=content, headers=headers)  # docstring: 统一错误响应
