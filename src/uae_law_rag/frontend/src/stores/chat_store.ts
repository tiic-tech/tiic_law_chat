// src/stores/chat_store.ts
//docstring
// 职责: Chat 状态容器的最小实现（可替换为专业状态库）；负责持有回放状态并提供 actions。
// 边界: 不直接调用 api/endpoints；不解析 HTTP DTO（只消费 service 输出的 domain 结果）。
// 上游关系: services/chat_service.ts。
// 下游关系: pages/chat（读取状态与触发 actions）。
import { sendChat } from '@/services/chat_service'
import {
  loadNodePreview,
  loadPageReplay,
  loadPageReplayByNode,
  loadRetrievalHits,
} from '@/services/evidence_service'
import type { ChatContextInput, ChatNormalizedResult, ChatSendInput } from '@/types/domain/chat'
import type { NodePreview, PageReplay, RetrievalHitsPaged } from '@/types/domain/evidence'
import type { RetrievalHitsQuery, NodePreviewQuery, PageReplayQuery, PageReplayByNodeQuery } from '@/services/evidence_service'

export type SystemNotice = {
  level: 'info' | 'warning' | 'error'
  message: string
  traceId?: string
  requestId?: string
}

type ChatStoreState = {
  messages: Array<{ id: string; role: 'user' | 'assistant'; text: string; runId?: string }>
  activeRunId?: string
  runsById: Record<string, ChatNormalizedResult>
  ui: {
    debugOpen: boolean
    evidenceOpen: boolean
  }
  evidence: {
    selectedNodeId?: string
    sourceFilter: string[]
    offset: number
    limit: number
  }
  cache: {
    retrievalHitsByKey: Record<string, RetrievalHitsPaged>
    nodePreviewById: Record<string, NodePreview>
    pageReplayByKey: Record<string, PageReplay>
  }
  notice?: SystemNotice
}

const createInitialState = (): ChatStoreState => ({
  messages: [],
  activeRunId: undefined,
  runsById: {},
  ui: {
    debugOpen: false,
    evidenceOpen: false,
  },
  evidence: {
    selectedNodeId: undefined,
    sourceFilter: [],
    offset: 0,
    limit: 10,
  },
  cache: {
    retrievalHitsByKey: {},
    nodePreviewById: {},
    pageReplayByKey: {},
  },
  notice: undefined,
})

let state: ChatStoreState = createInitialState()
const listeners = new Set<() => void>()

const emit = () => {
  for (const listener of listeners) {
    listener()
  }
}

const updateState = (next: ChatStoreState) => {
  state = next
  emit()
}

const mergeState = (next: Partial<ChatStoreState>) => {
  state = { ...state, ...next }
  emit()
}

const buildRetrievalCacheKey = (
  retrievalRecordId: string,
  query: { source?: string[]; offset?: number; limit?: number },
): string => {
  const source = query.source?.join(',') ?? ''
  const offset = query.offset ?? 0
  const limit = query.limit ?? 0
  return `${retrievalRecordId}|${source}|${offset}|${limit}`
}

const buildPageReplayKey = (query: PageReplayQuery): string => {
  const kbId = query.kbId ?? ''
  const maxChars = query.maxChars ?? ''
  return `${query.documentId}|${query.page}|${kbId}|${maxChars}`
}

const createMessageId = (prefix: string, runId: string): string => `${prefix}_${runId}`

export const chatStore = {
  getState: () => state,
  setState: (next: Partial<ChatStoreState>) => {
    mergeState(next)
  },
  subscribe: (listener: () => void) => {
    listeners.add(listener)
    return () => {
      listeners.delete(listener)
    }
  },

  // --- Actions (M1 minimal) ---
  send: async (
    query: string,
    opts: { debug?: boolean; kbId?: string; context?: ChatContextInput } = {},
  ): Promise<ChatNormalizedResult> => {
    const input: ChatSendInput = {
      query,
      kbId: opts.kbId,
      debug: opts.debug,
      context: opts.context,
    }
    const result = await sendChat(input)
    const runId = result.run.runId
    const userMessage = {
      id: createMessageId('user', runId),
      role: 'user' as const,
      text: query,
      runId,
    }
    const assistantMessage = {
      id: createMessageId('assistant', runId),
      role: 'assistant' as const,
      text: result.answer ?? '',
      runId,
    }

    updateState({
      ...state,
      activeRunId: runId,
      runsById: { ...state.runsById, [runId]: result },
      messages: [...state.messages, userMessage, assistantMessage],
    })

    return result
  },

  sendChatAndAppend: async (input: ChatSendInput): Promise<ChatNormalizedResult> => {
    return chatStore.send(input.query, {
      debug: input.debug,
      kbId: input.kbId,
      context: input.context,
    })
  },

  toggleDebug: () => {
    updateState({
      ...state,
      ui: { ...state.ui, debugOpen: !state.ui.debugOpen },
    })
  },

  toggleEvidence: () => {
    updateState({
      ...state,
      ui: { ...state.ui, evidenceOpen: !state.ui.evidenceOpen },
    })
  },

  selectCitation: (nodeId: string) => {
    updateState({
      ...state,
      ui: { ...state.ui, evidenceOpen: true },
      evidence: { ...state.evidence, selectedNodeId: nodeId },
    })
  },

  selectNode: (nodeId: string) => {
    updateState({
      ...state,
      ui: { ...state.ui, evidenceOpen: true },
      evidence: { ...state.evidence, selectedNodeId: nodeId },
    })
  },

  setRetrievalPaging: (next: { sourceFilter?: string[]; offset?: number; limit?: number }) => {
    updateState({
      ...state,
      evidence: {
        ...state.evidence,
        sourceFilter: next.sourceFilter ?? state.evidence.sourceFilter,
        offset: next.offset ?? state.evidence.offset,
        limit: next.limit ?? state.evidence.limit,
      },
    })
  },

  fetchRetrievalHits: async (
    retrievalRecordId: string,
    query: RetrievalHitsQuery = {},
  ): Promise<RetrievalHitsPaged> => {
    const effectiveQuery = {
      source: query.source ?? state.evidence.sourceFilter,
      offset: query.offset ?? state.evidence.offset,
      limit: query.limit ?? state.evidence.limit,
      group: query.group,
    }
    const key = buildRetrievalCacheKey(retrievalRecordId, effectiveQuery)
    const cached = state.cache.retrievalHitsByKey[key]
    if (cached) return cached

    const result = await loadRetrievalHits(retrievalRecordId, effectiveQuery)
    updateState({
      ...state,
      cache: {
        ...state.cache,
        retrievalHitsByKey: {
          ...state.cache.retrievalHitsByKey,
          [key]: result,
        },
      },
    })
    return result
  },

  fetchNodePreview: async (
    nodeId: string,
    query: NodePreviewQuery = {},
  ): Promise<NodePreview> => {
    const cached = state.cache.nodePreviewById[nodeId]
    if (cached) return cached
    const result = await loadNodePreview(nodeId, query)
    updateState({
      ...state,
      cache: {
        ...state.cache,
        nodePreviewById: {
          ...state.cache.nodePreviewById,
          [nodeId]: result,
        },
      },
    })
    return result
  },

  fetchPageReplay: async (query: PageReplayQuery): Promise<PageReplay> => {
    const key = buildPageReplayKey(query)
    const cached = state.cache.pageReplayByKey[key]
    if (cached) return cached
    const result = await loadPageReplay(query)
    updateState({
      ...state,
      cache: {
        ...state.cache,
        pageReplayByKey: {
          ...state.cache.pageReplayByKey,
          [key]: result,
        },
      },
    })
    return result
  },

  fetchPageReplayByNode: async (
    nodeId: string,
    query: PageReplayByNodeQuery = {},
  ): Promise<PageReplay> => {
    const key = `node:${nodeId}|${query.kbId ?? ''}|${query.maxChars ?? ''}`
    const cached = state.cache.pageReplayByKey[key]
    if (cached) return cached
    const result = await loadPageReplayByNode(nodeId, query)
    updateState({
      ...state,
      cache: {
        ...state.cache,
        pageReplayByKey: {
          ...state.cache.pageReplayByKey,
          [key]: result,
        },
      },
    })
    return result
  },

  setNotice: (notice?: SystemNotice) => {
    mergeState({ notice })
  },

  reset: () => {
    updateState(createInitialState())
  },
}
