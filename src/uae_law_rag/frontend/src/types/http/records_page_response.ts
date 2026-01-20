// src/types/http/records_page_response.ts
//docstring
// 职责: 对齐后端 records/page HTTP DTO 结构定义。
// 边界: 仅表达传输层字段，不做业务推断。
// 上游关系: api/endpoints/records.ts（后续新增）。
// 下游关系: services/evidence_service.ts。
import type { JsonObject } from '@/types/http/json'

export type PageRecordViewDTO = {
  kb_id: string
  document_id: string
  file_id: string
  page: number
  pages_total?: number
  content: string
  content_len: number
  meta: JsonObject
}
