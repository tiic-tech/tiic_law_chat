// src/uae_law_rag/frontend/src/types/ui/chat_view.tx
//docstring
// 职责: 定义 Chat 页面 UI 视图模型（ViewModel）。
// 边界: 只依赖 domain；不携带 HTTP DTO 或 service 逻辑。
// 上游关系: services/* 产出 view model。
// 下游关系: pages/chat 组件。
import type { EvidenceLocator } from '@/types/domain/evidence'
import type { RunStatus, RunTiming } from '@/types/domain/run'
import type { StepRecord } from '@/types/domain/step'
import type { EvidenceView } from './evidence_view'

export type EvaluatorBadgeView = {
  level: 'pass' | 'partial' | 'fail' | 'skipped' | 'unknown'
  score?: number
  label?: string
}

export type ActiveRunView = {
  runId: string
  status: RunStatus
  answer?: string
  evaluatorBadge?: EvaluatorBadgeView
  steps: StepRecord[]
  timing?: RunTiming
  providerSnapshot?: Record<string, unknown>
}

export type CitationClickRef = {
  nodeId: string
  documentId?: string
  page?: number
  start?: number
  end?: number
}

export type CitationView = {
  nodeId: string
  locator: EvidenceLocator
  onClickRef?: CitationClickRef
}

export type ChatMessageView = {
  id: string
  role: 'user' | 'assistant'
  content: string
  runId?: string
  citations?: CitationView[]
  evaluatorBadge?: EvaluatorBadgeView
}

export type ChatHistoryView = {
  items: ChatMessageView[]
}

export type PromptDebugView = {
  mode: string
  nodesUsed: number
  totalChars: number
  items: Array<{
    nodeId: string
    source?: string
    used: string
    chars: number
  }>
}

export type KeywordStatsView = {
  rawQuery: string
  items: Array<{
    keyword: string
    recall?: number
    precision?: number
    overlap?: number
    counts?: {
      gtTotal?: number
      kwTotal?: number
    }
    capped?: boolean
  }>
  meta?: Record<string, unknown>
}

export type EvidenceSummaryView = {
  totalHits?: number
  sources?: Array<{ name: string; count: number }>
}

export type ChatDebugView = {
  enabled: boolean
  promptDebug?: PromptDebugView
  keywordStats?: KeywordStatsView
  providerSnapshot?: Record<string, unknown>
  evidenceSummary?: EvidenceSummaryView
}

export type ChatView = {
  history: ChatHistoryView
  activeRun?: ActiveRunView
  citations: CitationView[]
  debug?: ChatDebugView
}

export type ChatSessionView = ChatView

export type ChatPageProps = {
  chat: ChatSessionView
  evidence: EvidenceView
  onSend?: (query: string) => void
  onSelectCitation?: (nodeId: string) => void
}
