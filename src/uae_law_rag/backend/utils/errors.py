# src/uae_law_rag/backend/utils/errors.py

"""
[职责] 统一领域错误合同（error_code/message/detail/cause）与最小 HTTP 映射策略（http_status/retryable）。
[边界] 不依赖 FastAPI/HTTPException；不引入业务语义；仅提供通用错误壳与校验。
[上游关系] services/pipelines 抛出 DomainError 或其子类；调用方负责补充 trace_id 等上下文字段。
[下游关系] api/errors.py 使用本模块将异常映射为 ErrorResponse 与 HTTP status。
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Optional, Tuple


ErrorDetail = Dict[str, Any]  # docstring: 错误细节类型（必须 JSON-safe）

ERROR_CODE_PATTERN_AREA = re.compile(r"^[A-Z][A-Z0-9]*(?:__[A-Z0-9]+)+$")  # docstring: AREA__REASON 规范
ERROR_CODE_PATTERN_DOT = re.compile(r"^[a-z][a-z0-9]*(?:\.[a-z0-9_]+)+$")  # docstring: area.reason 规范

STANDARD_ERROR_CODES = {  # docstring: HTTP 层通用错误码集合
    "bad_request",
    "not_found",
    "conflict",
    "pipeline_error",
    "external_dependency",
    "internal_error",
}

ERROR_HTTP_STATUS_BY_CODE = {  # docstring: 通用错误码 -> HTTP status
    "bad_request": 400,
    "not_found": 404,
    "conflict": 409,
    "pipeline_error": 500,
    "external_dependency": 503,
    "internal_error": 500,
}

ERROR_RETRYABLE_BY_CODE = {  # docstring: 通用错误码 -> retryable 默认值
    "bad_request": False,
    "not_found": False,
    "conflict": False,
    "pipeline_error": False,
    "external_dependency": True,
    "internal_error": False,
}

INTERNAL_ERROR_CODE = "internal_error"  # docstring: 未知异常统一错误码
INTERNAL_ERROR_MESSAGE = "internal error"  # docstring: 未知异常统一消息


def is_valid_error_code(error_code: str) -> bool:
    """
    [职责] 校验错误码是否满足命名规范或通用错误码列表。
    [边界] 仅做格式校验，不保证全局唯一。
    [上游关系] DomainError 初始化可调用校验；也可供 lint/检查脚本使用。
    [下游关系] api/errors.py 可基于该函数做告警或降级逻辑。
    """

    if not error_code:
        return False  # docstring: 空字符串直接视为无效
    if error_code in STANDARD_ERROR_CODES:
        return True  # docstring: 允许通用 HTTP 错误码
    return bool(
        ERROR_CODE_PATTERN_AREA.match(error_code) or ERROR_CODE_PATTERN_DOT.match(error_code)
    )  # docstring: 推荐格式校验（AREA__REASON / area.reason）


def ensure_json_safe_detail(detail: ErrorDetail) -> ErrorDetail:
    """
    [职责] 校验 detail 是否可 JSON 序列化。
    [边界] detail 必须是 dict；不做降级或裁剪；若失败直接抛错提示调用方自行处理。
    [上游关系] DomainError 初始化时调用。
    [下游关系] api/errors.py 可直接将 detail 写入 ErrorResponse.detail。
    """

    if not isinstance(detail, dict):
        raise ValueError("detail must be a dict")  # docstring: 强制 detail 为 dict 结构
    try:
        json.dumps(detail)  # docstring: JSON 序列化校验
    except TypeError as exc:
        raise ValueError("detail must be JSON-serializable") from exc  # docstring: 明确提示调用方降级
    return detail  # docstring: 返回原 detail 便于链式调用


class DomainError(Exception):
    """
    [职责] 领域错误最小合同：统一 error_code/message/detail/cause，并提供 http_status/retryable 提示。
    [边界] 仅表达语义，不承担日志、告警、HTTP 输出。
    [上游关系] services/pipelines 抛出本错误；必要时携带 cause。
    [下游关系] api/errors.py 根据本错误映射 HTTP status 与 ErrorResponse。
    """

    def __init__(
        self,
        *,
        error_code: str,
        message: str,
        detail: Optional[ErrorDetail] = None,
        cause: Optional[Exception] = None,
        http_status: Optional[int] = None,
        retryable: Optional[bool] = None,
        allow_nonstandard_code: bool = False,
    ) -> None:
        """
        [职责] 初始化 DomainError，并对 error_code/detail 做最小校验。
        [边界] 不做 trace_id 注入；不做日志记录。
        [上游关系] 调用方负责提供 error_code/message/detail/cause。
        [下游关系] api/errors.py 读取 error_code/http_status/retryable 字段完成映射。
        """

        if not is_valid_error_code(error_code) and not allow_nonstandard_code:
            raise ValueError(f"invalid error_code: {error_code}")  # docstring: 防止不规范错误码泄露
        normalized_detail = detail or {}  # docstring: 归一化 detail，确保 dict
        ensure_json_safe_detail(normalized_detail)  # docstring: 保证 detail 可序列化

        resolved_http_status = (
            http_status if http_status is not None else ERROR_HTTP_STATUS_BY_CODE.get(error_code, 500)
        )  # docstring: 解析 http_status（优先显式值）
        resolved_retryable = (
            retryable if retryable is not None else ERROR_RETRYABLE_BY_CODE.get(error_code, False)
        )  # docstring: 解析 retryable（优先显式值）

        super().__init__(message)
        self.error_code = error_code  # docstring: 稳定错误码
        self.message = message  # docstring: 用户可读错误信息
        self.detail = normalized_detail  # docstring: JSON-safe 细节
        self.cause = cause  # docstring: 上游异常引用
        self.http_status = resolved_http_status  # docstring: HTTP 映射提示
        self.retryable = resolved_retryable  # docstring: 可重试提示

        if cause is not None:
            self.__cause__ = cause  # docstring: 保留异常链路

    def to_dict(self) -> Dict[str, Any]:
        """
        [职责] 输出 ErrorResponse.error 结构（不包含 trace_id）。
        [边界] 不做字段脱敏；不包含 cause。
        [上游关系] api/errors.py 可直接包装该 dict。
        [下游关系] ErrorResponse.error 可被前端/UI/审计系统消费。
        """

        return {
            "code": self.error_code,
            "message": self.message,
            "detail": self.detail,
        }  # docstring: 返回稳定字段集合


class BadRequestError(DomainError):
    """
    [职责] 表达 400 Bad Request 语义的标准错误。
    [边界] 仅提供默认 error_code/http_status；消息可由调用方覆盖。
    [上游关系] 参数校验失败或用户输入不合法时抛出。
    [下游关系] api/errors.py 映射为 400 + ErrorResponse。
    """

    def __init__(
        self,
        *,
        message: str = "bad request",
        detail: Optional[ErrorDetail] = None,
        cause: Optional[Exception] = None,
        retryable: Optional[bool] = None,
    ) -> None:
        """
        [职责] 初始化 BadRequestError，默认 error_code=bad_request。
        [边界] 不解析业务语义；不记录日志。
        [上游关系] services/api 入参校验失败时抛出。
        [下游关系] api/errors.py 直接映射为 400。
        """

        super().__init__(
            error_code="bad_request",
            message=message,
            detail=detail,
            cause=cause,
            http_status=400,
            retryable=retryable if retryable is not None else False,
        )  # docstring: 使用标准 bad_request 语义


class NotFoundError(DomainError):
    """
    [职责] 表达 404 Not Found 语义的标准错误。
    [边界] 仅提供默认 error_code/http_status。
    [上游关系] 资源缺失或无法定位时抛出。
    [下游关系] api/errors.py 映射为 404 + ErrorResponse。
    """

    def __init__(
        self,
        *,
        message: str = "not found",
        detail: Optional[ErrorDetail] = None,
        cause: Optional[Exception] = None,
        retryable: Optional[bool] = None,
    ) -> None:
        """
        [职责] 初始化 NotFoundError，默认 error_code=not_found。
        [边界] 不补充 trace_id；不记录日志。
        [上游关系] services 或 repos 查询不到资源时抛出。
        [下游关系] api/errors.py 映射为 404。
        """

        super().__init__(
            error_code="not_found",
            message=message,
            detail=detail,
            cause=cause,
            http_status=404,
            retryable=retryable if retryable is not None else False,
        )  # docstring: 使用标准 not_found 语义


class ConflictError(DomainError):
    """
    [职责] 表达 409 Conflict 语义的标准错误。
    [边界] 仅提供默认 error_code/http_status。
    [上游关系] 唯一约束冲突或幂等写入冲突时抛出。
    [下游关系] api/errors.py 映射为 409 + ErrorResponse。
    """

    def __init__(
        self,
        *,
        message: str = "conflict",
        detail: Optional[ErrorDetail] = None,
        cause: Optional[Exception] = None,
        retryable: Optional[bool] = None,
    ) -> None:
        """
        [职责] 初始化 ConflictError，默认 error_code=conflict。
        [边界] 不引入 DB 具体异常类型。
        [上游关系] repos/services 捕获唯一约束冲突后抛出。
        [下游关系] api/errors.py 映射为 409。
        """

        super().__init__(
            error_code="conflict",
            message=message,
            detail=detail,
            cause=cause,
            http_status=409,
            retryable=retryable if retryable is not None else False,
        )  # docstring: 使用标准 conflict 语义


class PipelineError(DomainError):
    """
    [职责] 表达 pipeline 内部失败（500）语义的标准错误。
    [边界] 不包含具体 pipeline 步骤实现；仅表达失败。
    [上游关系] retrieval/generation/evaluator pipeline 抛出或封装。
    [下游关系] api/errors.py 映射为 500 + ErrorResponse。
    """

    def __init__(
        self,
        *,
        message: str = "pipeline error",
        detail: Optional[ErrorDetail] = None,
        cause: Optional[Exception] = None,
        retryable: Optional[bool] = None,
    ) -> None:
        """
        [职责] 初始化 PipelineError，默认 error_code=pipeline_error。
        [边界] 不暴露敏感上下文；detail 需调用方裁剪。
        [上游关系] pipelines 捕获异常后封装抛出。
        [下游关系] api/errors.py 映射为 500。
        """

        super().__init__(
            error_code="pipeline_error",
            message=message,
            detail=detail,
            cause=cause,
            http_status=500,
            retryable=retryable if retryable is not None else False,
        )  # docstring: 使用标准 pipeline_error 语义


class ExternalDependencyError(DomainError):
    """
    [职责] 表达外部依赖故障（503）语义的标准错误。
    [边界] 不绑定具体 provider；仅表达依赖不可用。
    [上游关系] LLM/向量库/外部 API 不可用时抛出。
    [下游关系] api/errors.py 映射为 503 + ErrorResponse。
    """

    def __init__(
        self,
        *,
        message: str = "external dependency error",
        detail: Optional[ErrorDetail] = None,
        cause: Optional[Exception] = None,
        retryable: Optional[bool] = None,
    ) -> None:
        """
        [职责] 初始化 ExternalDependencyError，默认 error_code=external_dependency。
        [边界] 不泄露 provider 密钥或敏感 endpoint。
        [上游关系] adapters/services 捕获第三方异常后抛出。
        [下游关系] api/errors.py 映射为 503。
        """

        super().__init__(
            error_code="external_dependency",
            message=message,
            detail=detail,
            cause=cause,
            http_status=503,
            retryable=retryable if retryable is not None else True,
        )  # docstring: 使用标准 external_dependency 语义


class InternalError(DomainError):
    """
    [职责] 表达未知异常的内部错误（500）语义。
    [边界] 不暴露原始异常堆栈到 message/detail。
    [上游关系] api/errors.py 捕获未知异常时可用该错误包装。
    [下游关系] 前端接收稳定的 internal_error 码。
    """

    def __init__(
        self,
        *,
        message: str = INTERNAL_ERROR_MESSAGE,
        detail: Optional[ErrorDetail] = None,
        cause: Optional[Exception] = None,
        retryable: Optional[bool] = None,
    ) -> None:
        """
        [职责] 初始化 InternalError，默认 error_code=internal_error。
        [边界] 不在此处记录日志或输出 trace_id。
        [上游关系] middleware 或 exception handler 使用。
        [下游关系] api/errors.py 映射为 500。
        """

        super().__init__(
            error_code=INTERNAL_ERROR_CODE,
            message=message,
            detail=detail,
            cause=cause,
            http_status=500,
            retryable=retryable if retryable is not None else False,
        )  # docstring: 使用标准 internal_error 语义


def to_http_error(
    error: Exception,
    *,
    trace_id: Optional[str] = None,
) -> Tuple[int, Dict[str, Any]]:
    """
    [职责] 将异常转换为 HTTP status + ErrorResponse payload（不耦合 FastAPI）。
    [边界] 不注入 request_id；不做日志记录。
    [上游关系] api/errors.py 捕获异常后调用。
    [下游关系] routers 返回统一 ErrorResponse。
    """

    if isinstance(error, DomainError):
        status_code = error.http_status  # docstring: 使用 DomainError 的映射提示
        payload = {"error": error.to_dict()}  # docstring: 复用标准 ErrorResponse.error 结构
    else:
        status_code = ERROR_HTTP_STATUS_BY_CODE[INTERNAL_ERROR_CODE]  # docstring: 未知异常统一 500
        payload = {
            "error": {
                "code": INTERNAL_ERROR_CODE,
                "message": INTERNAL_ERROR_MESSAGE,
                "detail": {},
            }
        }  # docstring: 未知异常降级为 internal_error

    if trace_id:
        payload["error"]["trace_id"] = trace_id  # docstring: API 层注入 trace_id

    return status_code, payload  # docstring: 兼容 FastAPI/Starlette 等调用方式
