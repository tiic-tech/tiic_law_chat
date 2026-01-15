//docstring
// 职责: 定义 run 与 debug 记录的 domain 结构。
// 边界: 不引入服务端内部实现字段。
// 上游关系: services/* 的 DTO 映射。
// 下游关系: UI 审计与证据展示。
import type { GateSummary } from '@/types/domain/gate'
import type { StepRecord } from '@/types/domain/step'

export type DebugRecords = {
  retrievalRecordId?: string
  generationRecordId?: string
  evaluationRecordId?: string
  documentId?: string
  [key: string]: unknown
}

export type DebugEnvelope = {
  traceId: string
  requestId: string
  records: DebugRecords
  timingMs?: Record<string, unknown>
  gate?: GateSummary
  [key: string]: unknown
}

export type RunRecord = {
  runId: string
  pipelineName?: string
  status: 'success' | 'degraded' | 'failed'
  startedAt?: string
  finishedAt?: string
  steps: StepRecord[]
}
