//docstring
// 职责: 对齐后端 error HTTP DTO 结构定义。
// 边界: 仅表达错误传输结构，不做本地扩展。
// 上游关系: api/http.ts 的错误捕获层。
// 下游关系: services 与 UI 错误展示。
import type { JsonObject } from '@/types/http/json'
export type ErrorCode =
  | 'bad_request'
  | 'not_found'
  | 'external_dependency'
  | 'pipeline_error'
  | 'internal_error'

export type ErrorInfoDTO = {
  code: ErrorCode
  message: string
  trace_id: string
  detail: JsonObject
}

export type ErrorResponseDTO = {
  error: ErrorInfoDTO
}
