// src/api/http.ts
//docstring
// 职责: 提供统一的 JSON 请求封装与基础错误处理（结构化 HttpError + JSON-safe body）。
// 边界: 不做业务语义转换，不依赖具体 endpoint；仅处理传输层协议与最小可解释错误信息。
// 上游关系: src/api/endpoints/*。
// 下游关系: 浏览器 fetch API。
import { env } from '@/config/env'
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

const API_BASE = env.apiBase

const getOrCreateUserId = (): string => {
  const key = 'uae_law_rag_user_id'
  const existing = localStorage.getItem(key)
  if (existing) return existing

  const created =
    globalThis.crypto?.randomUUID?.() ?? `anon_${Date.now().toString(36)}`
  localStorage.setItem(key, created)
  return created
}

const DEFAULT_HEADERS: Record<string, string> = {
  'Content-Type': 'application/json',
  'x-user-id': 'dev-user', // docstring: M1 本地联调默认用户；后续替换为 auth/session
}

function safeParseJson(text: string): JsonValue | undefined {
  if (!text) return undefined
  try {
    return JSON.parse(text) as JsonValue
  } catch {
    return undefined
  }
}

const normalizeBase = (base: string): string => (base.endsWith('/') ? base.slice(0, -1) : base)
const normalizePath = (path: string): string => (path.startsWith('/') ? path : `/${path}`)
const isAbsoluteUrl = (path: string): boolean => /^https?:\/\//i.test(path)

const buildUrl = (path: string): string => {
  if (isAbsoluteUrl(path)) return path
  const base = normalizeBase(API_BASE)
  const suffix = normalizePath(path)
  return base ? `${base}${suffix}` : suffix
}

const getErrorMessage = (error: unknown): string => {
  if (error instanceof Error) return error.message
  return 'Network error'
}

export const requestJson = async <T>(path: string, options: RequestOptions = {}): Promise<T> => {
  const { method = 'GET', headers, body, signal } = options
  const url = buildUrl(path)

  let response: Response
  try {
    response = await fetch(url, {
      method,
      headers: { ...DEFAULT_HEADERS, 'x-user-id': getOrCreateUserId(), ...headers },
      body: body === undefined ? undefined : JSON.stringify(body),
      signal,
    })
  } catch (error) {
    throw new HttpError({
      status: 0,
      url,
      method,
      message: getErrorMessage(error),
    })
  }

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
