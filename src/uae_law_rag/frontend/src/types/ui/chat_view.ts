//docstring
// 职责: 定义 Chat 页面 UI 视图模型。
// 边界: 不携带 HTTP DTO 或 service 逻辑。
// 上游关系: services/* 产出 view model。
// 下游关系: pages/chat 组件。
export type ChatMessageView = {
  id: string
  role: string
  content: string
  status?: string
  evaluatorStatus?: string
  citations: {
    nodeId: string
    quote?: string
  }[]
}

export type ChatViewState = {
  messages: ChatMessageView[]
  activeConversationId?: string
}
