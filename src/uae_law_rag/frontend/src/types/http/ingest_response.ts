//docstring
// 职责: 对齐后端 ingest HTTP DTO 结构定义。
// 边界: 仅表达传输层字段，不做业务推断。
// 上游关系: api/endpoints/ingest.ts。
// 下游关系: services/evidence_service.ts。
import type { JsonObject, JsonValue, JsonValueLike } from '@/types/http/json'
export type IngestStatus = 'pending' | 'success' | 'failed'

export type IngestProfileDTO = {
  parser?: string
  parse_version?: string
  segment_version?: string
  [key: string]: JsonValueLike
}

/**
 * Request profile：严格 JSON
 */
export type IngestProfileRequestDTO = {
  parser?: string
  parse_version?: string
  segment_version?: string
  extra?: Record<string, JsonValue>
}

/**
 * Request DTO：显式成为 JsonObject
 */
export type IngestRequestDTO = JsonObject & {
  kb_id: string
  file_name: string
  source_uri: string
  ingest_profile?: IngestProfileRequestDTO
  dry_run?: boolean
}

export type IngestTimingMsDTO = {
  total_ms?: number
  [key: string]: JsonValueLike
}

export type IngestResponseDTO = {
  kb_id: string
  file_id: string
  file_name: string
  document_id?: string
  status: IngestStatus
  node_count: number
  timing_ms: IngestTimingMsDTO
  debug?: {
    trace_id: string
    request_id: string
    records?: Record<string, JsonValue>
    timing_ms?: Record<string, JsonValue>
    [key: string]: JsonValueLike
  }
}
