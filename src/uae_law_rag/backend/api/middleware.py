# src/uae_law_rag/backend/api/middleware.py

"""
[职责] API Middleware：注入 trace_id/request_id 与请求耗时统计。
[边界] 不做业务逻辑与异常处理；不重算 pipeline timing。
[上游关系] FastAPI 应用注册本 middleware。
[下游关系] deps/routers 读取 request.state.trace_context 与 timing_ms。
"""

from __future__ import annotations

import time
from typing import Optional

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

from uae_law_rag.backend.schemas.audit import TraceContext
from uae_law_rag.backend.schemas.ids import UUIDStr, new_uuid
from uae_law_rag.backend.utils.constants import TIMING_TOTAL_MS_KEY

_TRACE_HEADER = "x-trace-id"  # docstring: trace header 约定
_REQUEST_HEADER = "x-request-id"  # docstring: request header 约定
_PARENT_HEADER = "x-parent-request-id"  # docstring: parent request header（可选）


def _resolve_header_id(value: Optional[str]) -> Optional[str]:
    """
    [职责] 解析 header 中的 trace/request id。
    [边界] 仅做字符串清理；不做 UUID 校验。
    [上游关系] TraceContextMiddleware 调用。
    [下游关系] 生成 TraceContext 字段。
    """
    raw = str(value or "").strip()
    return raw or None  # docstring: 空值回退 None


class TraceContextMiddleware(BaseHTTPMiddleware):
    """
    [职责] 注入 trace/request id，并记录 request 总耗时。
    [边界] 不捕获异常；不替代 routers/errors.py。
    [上游关系] FastAPI app.add_middleware 注册。
    [下游关系] deps.get_trace_context 使用 request.state.trace_context。
    """

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        """
        [职责] 包装请求生命周期并注入 trace/request/timing。
        [边界] 不修改业务 response body；仅设置 headers 与 state。
        [上游关系] ASGI 调度调用。
        [下游关系] routers/依赖读取 trace_context 与 timing_ms。
        """
        start_ts = time.perf_counter()  # docstring: 记录开始时间

        existing = getattr(request.state, "trace_context", None)
        trace_id = str(getattr(existing, "trace_id", "") or "") if existing else ""
        request_id = str(getattr(existing, "request_id", "") or "") if existing else ""
        parent_request_id = str(getattr(existing, "parent_request_id", "") or "") if existing else ""

        header_trace = _resolve_header_id(request.headers.get(_TRACE_HEADER))  # docstring: 读取 trace header
        header_request = _resolve_header_id(request.headers.get(_REQUEST_HEADER))  # docstring: 读取 request header
        header_parent = _resolve_header_id(request.headers.get(_PARENT_HEADER))  # docstring: 读取 parent header

        if header_trace:
            trace_id = header_trace  # docstring: header 覆盖 trace_id
        if header_request:
            request_id = header_request  # docstring: header 覆盖 request_id
        if header_parent:
            parent_request_id = header_parent  # docstring: header 覆盖 parent_request_id

        if not trace_id:
            trace_id = str(new_uuid())  # docstring: 缺失 trace_id 则生成
        if not request_id:
            request_id = str(new_uuid())  # docstring: 缺失 request_id 则生成

        trace_context = TraceContext(
            trace_id=UUIDStr(trace_id),  # docstring: 统一类型为 UUIDStr
            request_id=UUIDStr(request_id),  # docstring: 统一类型为 UUIDStr
            parent_request_id=UUIDStr(parent_request_id) if parent_request_id else None,  # docstring: 可选父请求ID
            tags={},
        )  # docstring: 构造 TraceContext

        request.state.trace_context = trace_context  # docstring: 注入上下文
        request.state.trace_id = trace_id  # docstring: 便捷字段
        request.state.request_id = request_id  # docstring: 便捷字段
        request.state.parent_request_id = parent_request_id or None  # docstring: 便捷字段

        try:
            response = await call_next(request)  # docstring: 执行下游 handler
        finally:
            total_ms = (time.perf_counter() - start_ts) * 1000.0  # docstring: 计算总耗时
            request.state.timing_ms = {TIMING_TOTAL_MS_KEY: total_ms}  # docstring: 写入 timing_ms

        response.headers[_TRACE_HEADER] = trace_id  # docstring: 回写 trace_id header
        response.headers[_REQUEST_HEADER] = request_id  # docstring: 回写 request_id header
        return response
