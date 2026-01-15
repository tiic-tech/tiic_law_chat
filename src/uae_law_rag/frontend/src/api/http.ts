// src/api/http.ts
//docstring
// 职责: 提供统一的 JSON 请求封装与基础错误处理（结构化 HttpError + JSON-safe body）。
// 边界: 不做业务语义转换，不依赖具体 endpoint；仅处理传输层协议与最小可解释错误信息。
// 上游关系: src/api/endpoints/*。
// 下游关系: 浏览器 fetch API。
import type { JsonValue } from '@/types/http/json'

export type HttpMethod = 'GET' | 'POST' | 'PUT' | 'PATCH' | 'DELETE'

export type RequestOptions = {
  method?: HttpMethod
  headers?: Record<string, string>
  body?: JsonValue
  signal?: AbortSignal
}

export type HttpErrorInfo = {
  status: number
  url: string
  method: HttpMethod
  message: string
  response_text?: string
  response_json?: JsonValue
}

export class HttpError extends Error {
  public override readonly name = 'HttpError'
  constructor(public readonly info: HttpErrorInfo) {
    super(info.message)
  }
}

const DEFAULT_HEADERS: Record<string, string> = {
  'Content-Type': 'application/json',
}

function safeParseJson(text: string): JsonValue | undefined {
  if (!text) return undefined
  try {
    return JSON.parse(text) as JsonValue
  } catch {
    return undefined
  }
}

export const requestJson = async <T>(url: string, options: RequestOptions = {}): Promise<T> => {
  const { method = 'GET', headers, body, signal } = options

  const response = await fetch(url, {
    method,
    headers: { ...DEFAULT_HEADERS, ...headers },
    body: body === undefined ? undefined : JSON.stringify(body),
    signal,
  })

  // 先读取文本，保证即使不是 JSON 也能给出可解释错误信息
  const text = await response.text()
  const parsed = safeParseJson(text)

  if (!response.ok) {
    throw new HttpError({
      status: response.status,
      url,
      method,
      message: `HTTP ${response.status}`,
      response_text: text || undefined,
      response_json: parsed,
    })
  }

  // 允许空 body（例如 204 或某些后端返回空字符串）
  if (!text) return undefined as T

  // 若返回不是 JSON，则直接抛结构化错误（避免下游吞 parse error）
  if (parsed === undefined) {
    throw new HttpError({
      status: response.status,
      url,
      method,
      message: 'Response is not valid JSON',
      response_text: text,
    })
  }

  return parsed as T
}
