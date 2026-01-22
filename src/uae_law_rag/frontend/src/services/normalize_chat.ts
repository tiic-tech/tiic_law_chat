// src/services/normalize_chat.ts
//docstring
// Responsibility: Map ChatResponseDTO into minimal Domain shape for tests and later normalize work.
// Boundary: Pure function only; no IO; no store/UI dependencies.
import type {
  ChatGateDecisionDTO,
  ChatResponseDTO,
  DebugEvidenceDTO,
  KeywordStatsDTO,
  PromptDebugDTO,
} from '@/types/http/chat_response'
import type { EvidenceIndex, EvidenceLocator, EvidenceTreeNode } from '@/types/domain/evidence'
import type { RunRecord, RunStatus, RunTiming } from '@/types/domain/run'
import type { StepName, StepRecord, StepStatus } from '@/types/domain/step'
import type { ChatDebugState, ChatNormalizedResult, KeywordStats, PromptDebug } from '@/types/domain/chat'

const isRecord = (value: unknown): value is Record<string, unknown> => {
  return typeof value === 'object' && value !== null
}

const readNumber = (record: Record<string, unknown>, key: string): number | undefined => {
  const value = record[key]
  return typeof value === 'number' ? value : undefined
}

const readString = (record: Record<string, unknown>, key: string): string | undefined => {
  const value = record[key]
  return typeof value === 'string' ? value : undefined
}

const normalizeDecisionStatus = (decision?: ChatGateDecisionDTO): StepStatus => {
  if (!decision) return 'success'
  if (decision.passed === true) return 'success'
  if (decision.passed === false) return 'degraded'
  const raw = (decision.status ?? '').toLowerCase()
  if (raw === 'pass' || raw === 'success') return 'success'
  if (raw === 'fail' || raw === 'failed' || raw === 'error') return 'error'
  if (raw) return 'degraded'
  return 'degraded'
}

const normalizeStep = (step: StepName, decision?: ChatGateDecisionDTO): StepRecord => {
  return {
    step,
    status: normalizeDecisionStatus(decision),
    reasons: Array.isArray(decision?.reasons) ? decision?.reasons : [],
  }
}

const deriveRunStatus = (chatStatus: ChatResponseDTO['status'], steps: StepRecord[]): RunStatus => {
  if (steps.some((item) => item.status === 'error')) return 'error'
  if (steps.some((item) => item.status === 'degraded')) return 'degraded'
  if (chatStatus === 'failed') return 'error'
  if (chatStatus === 'partial' || chatStatus === 'blocked') return 'degraded'
  return 'success'
}

const buildTiming = (timing: ChatResponseDTO['timing_ms']): RunTiming => {
  const stages: Record<string, number> = {}
  for (const [key, value] of Object.entries(timing)) {
    if (key === 'total_ms') continue
    if (typeof value === 'number') stages[key] = value
  }

  return {
    totalMs: typeof timing.total_ms === 'number' ? timing.total_ms : undefined,
    stages: Object.keys(stages).length ? stages : undefined,
  }
}

const buildLocator = (citation: ChatResponseDTO['citations'][number]): EvidenceLocator => {
  const rawLocator = isRecord(citation.locator) ? citation.locator : {}
  const page = typeof citation.page === 'number' ? citation.page : readNumber(rawLocator, 'page')
  const start = readNumber(rawLocator, 'start_offset') ?? readNumber(rawLocator, 'start')
  const end = readNumber(rawLocator, 'end_offset') ?? readNumber(rawLocator, 'end')
  const documentId = readString(rawLocator, 'document_id')
  const locator: EvidenceLocator = {}

  if (documentId) locator.documentId = documentId
  if (page !== undefined) locator.page = page
  if (start !== undefined) locator.start = start
  if (end !== undefined) locator.end = end

  const source = readString(rawLocator, 'source')
  if (source) locator.source = source

  const articleId = typeof citation.article_id === 'string'
    ? citation.article_id
    : readString(rawLocator, 'article_id')
  if (articleId) locator.articleId = articleId

  const sectionPath = typeof citation.section_path === 'string'
    ? citation.section_path
    : readString(rawLocator, 'section_path')
  if (sectionPath) locator.sectionPath = sectionPath

  return locator
}

const buildEvidenceTree = (evidence?: DebugEvidenceDTO): EvidenceTreeNode[] | undefined => {
  if (!evidence) return undefined
  const nodes: EvidenceTreeNode[] = []

  for (const [sourceKey, sourceValue] of Object.entries(evidence.by_source ?? {})) {
    const sourceNode: EvidenceTreeNode = {
      id: `source:${sourceKey}`,
      label: sourceKey,
      children: [],
    }

    for (const [docId, docValue] of Object.entries(sourceValue.by_document ?? {})) {
      const docNode: EvidenceTreeNode = {
        id: `doc:${docId}`,
        label: docId,
        children: [],
      }

      for (const [pageKey, nodeIds] of Object.entries(docValue.pages ?? {})) {
        const pageNode: EvidenceTreeNode = {
          id: `page:${docId}:${pageKey}`,
          label: `page ${pageKey}`,
          children: (nodeIds ?? []).map((nodeId) => ({
            id: nodeId,
            label: nodeId,
          })),
        }
        docNode.children?.push(pageNode)
      }

      sourceNode.children?.push(docNode)
    }

    nodes.push(sourceNode)
  }

  return nodes.length ? nodes : undefined
}

const buildPromptDebug = (debug?: PromptDebugDTO): PromptDebug | undefined => {
  if (!debug) return undefined

  return {
    mode: debug.mode,
    nodesUsed: debug.totals.nodes_used,
    totalChars: debug.totals.total_chars,
    items: debug.context_items.map((item) => ({
      nodeId: item.node_id,
      source: item.source ?? undefined,
      used: item.used,
      chars: item.chars,
    })),
  }
}

const buildKeywordStats = (stats?: KeywordStatsDTO): KeywordStats | undefined => {
  if (!stats) return undefined

  return {
    rawQuery: stats.raw_query,
    items: stats.items.map((item) => ({
      keyword: item.keyword ?? '',
      recall: item.recall ?? undefined,
      precision: item.precision ?? undefined,
      overlap: item.overlap ?? undefined,
      counts: {
        gtTotal: item.gt_total ?? undefined,
        kwTotal: item.kw_total ?? undefined,
      },
      capped: item.capped ?? undefined,
    })),
    meta: stats.meta as Record<string, unknown>,
  }
}

export const normalizeChatResponse = (response: ChatResponseDTO): ChatNormalizedResult => {
  const gate = response.debug?.gate
  const steps: StepRecord[] = gate
    ? [
      normalizeStep('retrieval', gate.retrieval),
      normalizeStep('generation', gate.generation),
      normalizeStep('evaluator', gate.evaluator),
    ]
    : []

  const run: RunRecord = {
    runId: response.message_id,
    conversationId: response.conversation_id,
    messageId: response.message_id,
    kbId: response.kb_id,
    status: deriveRunStatus(response.status, steps),
    timing: buildTiming(response.timing_ms),
    providerSnapshot: response.debug?.provider_snapshot,
    records: response.debug?.records
      ? {
        retrievalRecordId: response.debug.records.retrieval_record_id,
        generationRecordId: response.debug.records.generation_record_id,
        evaluationRecordId: response.debug.records.evaluation_record_id,
        documentId: response.debug.records.document_id,
      }
      : undefined,
    steps,
  }

  const evidence: EvidenceIndex = {
    citations: response.citations.map((citation) => ({
      nodeId: citation.node_id,
      locator: buildLocator(citation),
    })),
    debugEvidenceTree: buildEvidenceTree(response.debug?.evidence),
  }

  const debug: ChatDebugState = response.debug
    ? {
      available: true,
      promptDebug: buildPromptDebug(response.debug.prompt_debug),
      keywordStats: buildKeywordStats(response.debug.keyword_stats),
    }
    : {
      available: false,
      message: 'Debug disabled or not returned.',
    }

  return {
    run,
    evidence,
    answer: response.answer,
    debug,
    evaluator: response.evaluator
      ? {
          status: response.evaluator.status,
          warnings: response.evaluator.warnings,
          ruleVersion: response.evaluator.rule_version,
        }
      : undefined,
  }
}
