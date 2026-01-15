//docstring
// 职责: Chat 状态容器的最小实现（可替换为专业状态库）。
// 边界: 不直接发起请求，不解析 HTTP DTO。
// 上游关系: services/chat_service.ts。
// 下游关系: pages/chat 读取状态。
import type { ChatResult } from '@/types/domain/message'

type ChatStoreState = {
  results: ChatResult[]
}

const initialState: ChatStoreState = {
  results: [],
}

let state: ChatStoreState = { ...initialState }

export const chatStore = {
  getState: () => state,
  setState: (next: Partial<ChatStoreState>) => {
    state = { ...state, ...next }
  },
}
