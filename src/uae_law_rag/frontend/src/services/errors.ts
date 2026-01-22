import { HttpError } from '@/api/http'
import type { EvaluatorSummary } from '@/types/domain/message'
import type { StepRecord } from '@/types/domain/step'
import type { SystemNoticeView } from '@/types/ui'

type NoticeMeta = SystemNoticeView['meta']

const isNoticeView = (value: unknown): value is SystemNoticeView => {
  if (!value || typeof value !== 'object') return false
  const record = value as Record<string, unknown>
  return typeof record.level === 'string' && typeof record.kind === 'string' && typeof record.title === 'string'
}

const buildHttpMeta = (error: HttpError, endpoint?: string): NoticeMeta => {
  return {
    status: error.info.status,
    requestId: error.info.requestId,
    traceId: error.info.traceId,
    endpoint: endpoint ?? error.info.url,
  }
}

export const toSystemNotice = (error: unknown, context?: { endpoint?: string }): SystemNoticeView => {
  if (isNoticeView(error)) return error

  if (error instanceof HttpError) {
    return {
      level: 'error',
      kind: 'http',
      title: error.info.message || `HTTP ${error.info.status}`,
      detail: error.info.response_text ?? error.message,
      raw: error.info.response_json ?? error.info.response_text ?? error.info,
      meta: buildHttpMeta(error, context?.endpoint),
    }
  }

  if (error instanceof Error) {
    return {
      level: 'error',
      kind: 'unexpected',
      title: error.message || 'Unexpected error',
      detail: error.stack,
      raw: error,
    }
  }

  if (typeof error === 'string') {
    return {
      level: 'error',
      kind: 'unexpected',
      title: error,
    }
  }

  return {
    level: 'error',
    kind: 'unexpected',
    title: 'Unexpected error',
    raw: error,
  }
}

const hasNoEvidenceReason = (reasons: string[]): boolean => {
  return reasons.some((reason) => {
    const normalized = reason.toLowerCase()
    return normalized.includes('noevidence') || normalized.includes('no_evidence') || normalized.includes('empty')
  })
}

export const toGateNotice = (steps: StepRecord[] | undefined): SystemNoticeView | undefined => {
  if (!steps || steps.length === 0) return undefined
  const retrieval = steps.find((step) => step.step === 'retrieval')
  if (!retrieval || retrieval.status === 'success') return undefined

  const reasons = Array.isArray(retrieval.reasons) ? retrieval.reasons : []
  const title = hasNoEvidenceReason(reasons) ? 'No evidence retrieved.' : 'Retrieval gate failed.'
  const detail = reasons.length ? reasons.join(', ') : `status: ${retrieval.status}`
  const level = retrieval.status === 'error' ? 'error' : 'warning'

  return {
    level,
    kind: 'gate',
    title,
    detail,
    raw: retrieval,
  }
}

export const toEvaluatorNotice = (summary?: EvaluatorSummary): SystemNoticeView | undefined => {
  if (!summary) return undefined
  if (summary.status === 'pass') return undefined

  const level = summary.status === 'fail' ? 'error' : summary.status === 'partial' ? 'warning' : 'info'
  const title =
    summary.status === 'partial'
      ? 'Evaluator degraded.'
      : summary.status === 'skipped'
        ? 'Evaluator skipped.'
        : 'Evaluator failed.'

  return {
    level,
    kind: 'evaluator',
    title,
    detail: summary.warnings?.length ? summary.warnings.join(', ') : undefined,
    raw: summary,
  }
}

export const toEvaluatorStepNotice = (steps: StepRecord[] | undefined): SystemNoticeView | undefined => {
  if (!steps || steps.length === 0) return undefined
  const evaluator = steps.find((step) => step.step === 'evaluator')
  if (!evaluator || evaluator.status === 'success') return undefined

  const level = evaluator.status === 'error' ? 'error' : 'warning'
  return {
    level,
    kind: 'evaluator',
    title: 'Evaluator failed.',
    detail: evaluator.reasons?.length ? evaluator.reasons.join(', ') : `status: ${evaluator.status}`,
    raw: evaluator,
  }
}
