// src/types/http/records_node_response.ts
//docstring
// 职责: 对齐后端 records/node HTTP DTO 结构定义。
// 边界: 仅表达传输层字段，不做业务推断。
// 上游关系: api/endpoints/records.ts（后续新增）。
// 下游关系: services/evidence_service.ts。
import type { JsonObject } from '@/types/http/json'

export type NodeRecordMetaDTO = JsonObject & {
  window?: string
  original_text?: string
}

export type NodeRecordViewDTO = {
  node_id: string
  kb_id?: string
  document_id: string
  node_index: number
  page?: number
  start_offset?: number
  end_offset?: number
  page_start_offset?: number
  page_end_offset?: number
  article_id?: string
  section_path?: string
  text_excerpt: string
  text_len: number
  meta: NodeRecordMetaDTO
}
