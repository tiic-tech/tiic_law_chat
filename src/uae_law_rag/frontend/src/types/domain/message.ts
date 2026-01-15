//docstring
// 职责: 定义消息与回答的 domain 结构。
// 边界: 不包含 UI 展示细节。
// 上游关系: services/chat_service.ts。
// 下游关系: pages/chat 与 ui 视图模型。
import type { DebugEnvelope } from '@/types/domain/run'

export type ChatStatus = 'success' | 'blocked' | 'partial' | 'failed'
export type EvaluatorStatus = 'pass' | 'partial' | 'fail' | 'skipped'
export type MessageRole = 'user' | 'assistant' | 'system'

export type EvaluatorSummary = {
  status: EvaluatorStatus
  ruleVersion: string
  warnings: string[]
}

export type Citation = {
  nodeId: string
  rank?: number
  quote?: string
  page?: number
  articleId?: string
  sectionPath?: string
  locator?: Record<string, unknown>
}

export type ChatMessage = {
  id: string
  role: MessageRole
  content: string
  status?: ChatStatus
}

export type ChatResult = {
  conversationId: string
  messageId: string
  kbId: string
  status: ChatStatus
  answer: string
  citations: Citation[]
  evaluator: EvaluatorSummary
  timingMs?: Record<string, unknown>
  traceId: string
  requestId: string
  debug?: DebugEnvelope
}
