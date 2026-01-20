//docstring
// 职责: 对齐后端 chat HTTP DTO 结构定义。
// 边界: 仅表达传输层字段，不做业务推断。
// 上游关系: api/endpoints/chat.ts。
// 下游关系: services/chat_service.ts。
import type { JsonObject, JsonValue, JsonValueLike } from '@/types/http/json'

export type ChatStatus = 'success' | 'blocked' | 'partial' | 'failed'
export type EvaluatorStatus = 'pass' | 'partial' | 'fail' | 'skipped'

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

export type CitationLocatorDTO = {
  page?: number
  start_offset?: number
  end_offset?: number
  article_id?: string
  section_path?: string
  source?: string
  [key: string]: JsonValueLike
}

export type CitationViewDTO = {
  node_id: string
  rank?: number
  quote?: string
  page?: number
  article_id?: string
  section_path?: string
  locator: CitationLocatorDTO
}

export type ChatGateDecisionDTO = {
  passed?: boolean
  status?: string
  reasons?: string[]
}

export type ChatGateSummaryDTO = {
  retrieval?: ChatGateDecisionDTO
  generation?: ChatGateDecisionDTO
  evaluator?: ChatGateDecisionDTO
}

export type DebugRecordsDTO = {
  retrieval_record_id?: string
  generation_record_id?: string
  evaluation_record_id?: string
  document_id?: string
}

export type DebugEnvelopeDTO = {
  trace_id: string
  request_id: string
  records: DebugRecordsDTO
  timing_ms: Record<string, JsonValue>
}

export type DebugEvidenceStatsDTO = {
  dropped_missing_document_id: number
  dropped_missing_node_id: number
  unknown_page_count: number
  deduped_node_count: number
  total_hits_in: number
  total_hits_used: number
  counts_by_source: Record<string, number>
}

export type DebugEvidenceDocumentDTO = {
  file_id: string | null
  pages: Record<string, string[]>
}

export type DebugEvidenceSourceDTO = {
  by_document: Record<string, DebugEvidenceDocumentDTO>
}

export type DebugEvidenceDTO = {
  version: string
  document_ids: string[]
  by_source: Record<string, DebugEvidenceSourceDTO>
  caps: {
    max_documents: number
    max_nodes_per_document: number
    max_pages_per_document: number
  }
  meta: {
    note: string
    stats: DebugEvidenceStatsDTO
  }
}

export type PromptDebugContextItemDTO = {
  node_id: string
  source?: string | null
  used: 'window' | 'original_text' | 'excerpt'
  chars: number
}

export type PromptDebugDTO = {
  version: string
  mode: string
  context_items: PromptDebugContextItemDTO[]
  totals: {
    nodes_used: number
    total_chars: number
  }
}

export type KeywordStatItemDTO = {
  keyword?: string | null
  gt_total?: number | null
  kw_total?: number | null
  overlap?: number | null
  recall?: number | null
  precision?: number | null
  capped?: boolean | null
}

export type KeywordStatsDTO = {
  raw_query: string
  keywords_list: string[]
  items: KeywordStatItemDTO[]
  timing_ms: Record<string, JsonValue>
  meta: Record<string, JsonValue>
}

export type ChatDebugEnvelopeDTO = DebugEnvelopeDTO & {
  gate?: ChatGateSummaryDTO
  provider_snapshot?: Record<string, JsonValue>
  hits_count?: number
  evidence?: DebugEvidenceDTO
  prompt_debug?: PromptDebugDTO
  keyword_stats?: KeywordStatsDTO
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
