// src/types/http/records_retrieval_response.ts
//docstring
// 职责: 对齐后端 records/retrieval HTTP DTO 结构定义。
// 边界: 仅表达传输层字段，不做业务推断。
// 上游关系: api/endpoints/records.ts（后续新增）。
// 下游关系: services/evidence_service.ts。
import type { JsonValue, JsonValueLike } from '@/types/http/json'

export type HitSource = 'keyword' | 'vector' | 'fused' | 'reranked'
export type TimingMsDTO = {
  total_ms?: number
  [key: string]: JsonValueLike
}

export type RetrievalStrategySnapshotDTO = {
  keyword_top_k?: number
  vector_top_k?: number
  fusion_top_k?: number
  rerank_top_k?: number
  fusion_strategy?: string
  rerank_strategy?: string
  provider_snapshot: Record<string, JsonValue>
}

export type HitSummaryDTO = {
  node_id: string
  source: HitSource
  rank: number
  score: number
  excerpt?: string
  locator: Record<string, JsonValue>
}

export type RetrievalRecordViewDTO = {
  retrieval_record_id: string
  message_id: string
  kb_id: string
  query_text: string
  strategy_snapshot: RetrievalStrategySnapshotDTO
  timing_ms: TimingMsDTO
  hits: HitSummaryDTO[]
  hits_total?: number
  hits_offset?: number
  hits_limit?: number
  hits_by_source: Record<string, HitSummaryDTO[]>
  hit_counts: Record<string, number>
}
