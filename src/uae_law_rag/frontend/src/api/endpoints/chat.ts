//docstring
// 职责: chat 相关 HTTP endpoint 调用。
// 边界: 仅负责请求与响应，不做业务映射。
// 上游关系: services/chat_service.ts。
// 下游关系: src/api/http.ts。
import { requestJson } from '@/api/http'
import type { ChatRequestDTO, ChatResponseDTO } from '@/types/http/chat_response'

export const postChat = (payload: ChatRequestDTO) => {
  return requestJson<ChatResponseDTO>('/api/chat', {
    method: 'POST',
    body: payload,
  })
}
