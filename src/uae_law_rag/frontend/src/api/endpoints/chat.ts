// src/api/endpoints/chat.ts
//docstring
// 职责: chat 相关 HTTP endpoint 调用。
// 边界: 仅负责请求与响应，不做业务映射；M1 阶段可做最小的 dev header 注入（x-user-id）。
// 上游关系: services/chat_service.ts。
// 下游关系: src/api/http.ts。
import { requestJson } from '@/api/http'
import type { ChatRequestDTO, ChatResponseDTO } from '@/types/http/chat_response'

const DEV_USER_ID = 'dev-user' // docstring: M1 本地联调用固定 user（后续替换为 auth/session）
const DEFAULT_KB_ID = 'default' // docstring: M1 本地联调用固定 kb（后续由 UI 选择）

export const postChat = (payload: ChatRequestDTO) => {
  const next: ChatRequestDTO = {
    ...payload,
    kb_id: payload.kb_id ?? DEFAULT_KB_ID,
  }

  return requestJson<ChatResponseDTO>('/chat', {
    method: 'POST',
    headers: {
      'x-user-id': DEV_USER_ID,
    },
    body: next,
  })
}
