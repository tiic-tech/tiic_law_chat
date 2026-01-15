//docstring
// 职责: 对齐后端 chat HTTP DTO 结构定义。
// 边界: 仅表达传输层字段，不做业务推断。
// 上游关系: api/endpoints/chat.ts。
// 下游关系: services/chat_service.ts。
import type { JsonObject, JsonValue, JsonValueLike } from '@/types/http/json'

export type ChatStatus = 'success' | 'blocked' | 'partial' | 'failed'
export type EvaluatorStatus = 'pass' | 'partial' | 'fail' | 'skipped'

/**
 * Response/Debug 允许扩展字段含 undefined（JsonValueLike）
 * 用途：后端可能透传一些可选扩展字段，或前端 debug 结构需要容忍缺失
 */
export type ChatContextConfigDTO = {
  keyword_top_k?: number
  vector_top_k?: number
  fusion_top_k?: number
  rerank_top_k?: number
  fusion_strategy?: string
  rerank_strategy?: string
  embed_provider?: string
  embed_model?: string
  embed_dim?: number
  model_provider?: string
  model_name?: string
  prompt_name?: string
  prompt_version?: string
  evaluator_config?: Record<string, JsonValue>
  return_records?: boolean
  return_hits?: boolean
  [key: string]: JsonValueLike
}

/**
 * Request 必须是严格 JSON（JsonValue），避免 requestJson body 类型冲突
 * 规则：request body 不应携带 undefined 值；可选字段通过“省略字段”表达
 */
export type ChatContextConfigRequestDTO = {
  keyword_top_k?: number
  vector_top_k?: number
  fusion_top_k?: number
  rerank_top_k?: number
  fusion_strategy?: string
  rerank_strategy?: string
  embed_provider?: string
  embed_model?: string
  embed_dim?: number
  model_provider?: string
  model_name?: string
  prompt_name?: string
  prompt_version?: string
  evaluator_config?: Record<string, JsonValue>
  return_records?: boolean
  return_hits?: boolean
  extra?: Record<string, JsonValue>
}

/**
 * Request DTO：显式成为 JsonObject 子类型，确保可作为 requestJson body
 */
export type ChatRequestDTO = JsonObject & {
  query: string
  conversation_id?: string
  kb_id?: string
  context?: ChatContextConfigRequestDTO
  debug?: boolean
}

export type EvaluatorSummaryDTO = {
  status: EvaluatorStatus
  rule_version: string
  warnings: string[]
}

export type CitationViewDTO = {
  node_id: string
  rank?: number
  quote?: string
  page?: number
  article_id?: string
  section_path?: string
  locator?: Record<string, JsonValue>
  [key: string]: JsonValueLike
}

export type ChatGateSummaryDTO = {
  retrieval?: Record<string, JsonValue>
  generation?: Record<string, JsonValue>
  evaluator?: Record<string, JsonValue>
  extra?: Record<string, JsonValue>
}

export type DebugRecordsDTO = {
  retrieval_record_id?: string
  generation_record_id?: string
  evaluation_record_id?: string
  document_id?: string
  [key: string]: JsonValueLike
}

export type DebugEnvelopeDTO = {
  trace_id: string
  request_id: string
  records: DebugRecordsDTO
  timing_ms?: Record<string, JsonValue>
  extra?: Record<string, JsonValue>
}

export type ChatDebugEnvelopeDTO = DebugEnvelopeDTO & {
  gate?: ChatGateSummaryDTO
}

export type ChatTimingMsDTO = {
  total_ms?: number
  [key: string]: JsonValueLike
}

/**
 * Response DTO：不需要继承 JsonObject（结构型 DTO 不应被强行视为 JsonValue）
 */
export type ChatResponseDTO = {
  conversation_id: string
  message_id: string
  kb_id: string
  status: ChatStatus
  answer: string
  citations: CitationViewDTO[]
  evaluator: EvaluatorSummaryDTO
  timing_ms: ChatTimingMsDTO
  trace_id: string
  request_id: string
  debug?: ChatDebugEnvelopeDTO
}
