import type { ChatNormalizedResult, ChatDebugState } from '@/types/domain/chat'
import type { ChatMessage } from '@/types/domain/message'
import type { EvidenceCitation, RetrievalHit } from '@/types/domain/evidence'
import type {
  ActiveRunView,
  ChatDebugView,
  ChatSessionView,
  ChatMessageView,
  CitationView,
  EvidenceView,
  EvaluatorBadgeView,
  HitRow,
  RetrievalHitsView,
} from '@/types/ui'
import {
  MESSAGES_NO_DEBUG,
  MESSAGES_OK,
  RUN_EMPTY,
  RUN_NO_DEBUG,
  RUN_OK,
} from '@/fixtures/mock_domain'

export type MockChatMode = 'ok' | 'no_debug' | 'empty' | 'error'

export type ChatServiceSnapshot = {
  chat: ChatSessionView
  evidence: EvidenceView
}

type ChatSendOptions = {
  mode?: MockChatMode
}

const delay = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms))

const mapEvaluatorBadge = (status: ActiveRunView['status']): EvaluatorBadgeView => {
  if (status === 'success') {
    return { level: 'pass', label: 'pass' }
  }
  if (status === 'degraded') {
    return { level: 'partial', label: 'degraded' }
  }
  return { level: 'fail', label: 'error' }
}

const mapActiveRun = (result: ChatNormalizedResult): ActiveRunView => {
  return {
    runId: result.run.runId,
    status: result.run.status,
    answer: result.answer,
    evaluatorBadge: mapEvaluatorBadge(result.run.status),
    steps: result.run.steps,
    timing: result.run.timing,
    providerSnapshot: result.run.providerSnapshot,
  }
}

const mapCitations = (citations: EvidenceCitation[]): CitationView[] => {
  return citations.map((citation) => ({
    nodeId: citation.nodeId,
    locator: citation.locator,
    onClickRef: {
      nodeId: citation.nodeId,
      documentId: citation.locator.documentId,
      page: citation.locator.page,
      start: citation.locator.start,
      end: citation.locator.end,
    },
  }))
}

const mapDebug = (debug: ChatDebugState): ChatDebugView => {
  return {
    enabled: debug.available,
    promptDebug: debug.promptDebug,
    keywordStats: debug.keywordStats,
  }
}

const mapHitRow = (hit: RetrievalHit): HitRow => {
  return {
    nodeId: hit.nodeId,
    source: hit.source,
    rank: hit.rank,
    score: hit.score,
    page: hit.locator?.page,
    articleId: hit.locator?.articleId,
    sectionPath: hit.locator?.sectionPath,
    excerpt: hit.locator?.start !== undefined ? `Offsets ${hit.locator.start}-${hit.locator.end}` : undefined,
  }
}

const buildRetrievalView = (hits: RetrievalHit[], source?: string): RetrievalHitsView => {
  const availableSources = Array.from(
    new Set(hits.map((hit) => hit.source).filter((value): value is string => Boolean(value))),
  )
  return {
    items: hits.map(mapHitRow),
    page: 1,
    pageSize: hits.length,
    total: hits.length,
    source,
    availableSources,
  }
}

const buildEvidenceView = (result?: ChatNormalizedResult): EvidenceView => {
  if (!result) {
    return {
      retrievalHits: { items: [], page: 1, pageSize: 0, total: 0 },
      evidenceTree: undefined,
      nodePreview: undefined,
    }
  }

  const hits = result.evidence.retrievalHitsPaged?.items ?? []
  return {
    retrievalHits: buildRetrievalView(hits, result.evidence.retrievalHitsPaged?.source),
    evidenceTree: result.evidence.debugEvidenceTree,
    nodePreview: undefined,
  }
}

const buildHistory = (
  messages: ChatMessage[],
  result?: ChatNormalizedResult,
  userOverride?: string,
): ChatMessageView[] => {
  return messages.map((message) => {
    const content = message.role === 'user' && userOverride ? userOverride : message.content
    if (message.role === 'assistant' && result) {
      return {
        id: message.id,
        role: 'assistant',
        content: content || result.answer || '',
        runId: result.run.runId,
        citations: mapCitations(result.evidence.citations),
        evaluatorBadge: mapEvaluatorBadge(result.run.status),
      }
    }
    return {
      id: message.id,
      role: message.role === 'assistant' ? 'assistant' : 'user',
      content,
    }
  })
}

const buildSnapshot = (
  result: ChatNormalizedResult | undefined,
  messages: ChatMessage[],
  userOverride?: string,
): ChatServiceSnapshot => {
  const historyItems = buildHistory(messages, result, userOverride)
  const chat: ChatSessionView = {
    history: { items: historyItems },
    activeRun: result ? mapActiveRun(result) : undefined,
    citations: result ? mapCitations(result.evidence.citations) : [],
    debug: result ? mapDebug(result.debug) : { enabled: false },
  }
  return {
    chat,
    evidence: buildEvidenceView(result),
  }
}

export const createMockChatService = (initialMode: MockChatMode = 'ok') => {
  let mode: MockChatMode = initialMode

  const resolveScenario = (nextMode: MockChatMode) => {
    switch (nextMode) {
      case 'no_debug':
        return { result: RUN_NO_DEBUG, messages: MESSAGES_NO_DEBUG }
      case 'empty':
        return { result: RUN_EMPTY, messages: [] }
      case 'ok':
      default:
        return { result: RUN_OK, messages: MESSAGES_OK }
    }
  }

  return {
    getMode: () => mode,
    setMode: (next: MockChatMode) => {
      mode = next
    },
    getSnapshot: async (nextMode?: MockChatMode): Promise<ChatServiceSnapshot> => {
      const activeMode = nextMode ?? mode
      await delay(450)
      if (activeMode === 'error') {
        throw new Error('Mock chat service error')
      }
      const scenario = resolveScenario(activeMode)
      return buildSnapshot(scenario.result, scenario.messages)
    },
    sendMessage: async (text: string, options: ChatSendOptions = {}): Promise<ChatServiceSnapshot> => {
      const activeMode = options.mode ?? mode
      await delay(680)
      if (activeMode === 'error') {
        throw new Error('Mock chat service error')
      }
      const scenario = resolveScenario(activeMode)
      const baseMessages = scenario.messages.length ? scenario.messages : MESSAGES_OK
      return buildSnapshot(scenario.result, baseMessages, text)
    },
  }
}
