//docstring
// 职责: ingest 相关 HTTP endpoint 调用。
// 边界: 仅负责请求与响应，不做业务映射。
// 上游关系: services/evidence_service.ts 或未来 ingest_service。
// 下游关系: src/api/http.ts。
import { requestJson } from '@/api/http'
import type { IngestRequestDTO, IngestResponseDTO } from '@/types/http/ingest_response'

export const postIngest = (payload: IngestRequestDTO) => {
  return requestJson<IngestResponseDTO>('/ingest', {
    method: 'POST',
    body: payload,
  })
}
