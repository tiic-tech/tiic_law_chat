// src/stores/chat_store.ts
//docstring
// 职责: Chat 状态容器的最小实现（可替换为专业状态库）；负责持有 UI 视图与证据状态机。
// 边界: 不直接调用 api/endpoints；不解析 HTTP DTO（只消费 service 输出的 view/domain 结果）。
// 上游关系: services/chat_service.ts 或 mock services（由 container 注入）。
// 下游关系: pages/chat（读取状态与触发 actions）。
import type { NodePreview, PageReplay } from '@/types/domain/evidence'
import type { ChatMessageView, ChatSessionView, EvidenceView, SystemNoticeView } from '@/types/ui'
import { toEvaluatorNotice, toEvaluatorStepNotice, toGateNotice, toSystemNotice } from '@/services/errors'

export type EvidenceLoadStatus = 'idle' | 'loading' | 'failed' | 'loaded'

export type ChatScenario = 'ok' | 'no_debug' | 'empty' | 'error'

export type ChatServiceSnapshot = {
  chat: ChatSessionView
  evidence: EvidenceView
}

export type ChatSendOptions = {
  mode?: ChatScenario
  conversationId?: string
  kbId?: string
  debug?: boolean
  history?: ChatMessageView[]
}

export type ChatService = {
  sendMessage: (text: string, options?: ChatSendOptions) => Promise<ChatServiceSnapshot>
  getSnapshot?: (mode?: ChatScenario) => Promise<ChatServiceSnapshot>
  getHistory?: (conversationId: string) => Promise<ChatServiceSnapshot>
}

export type EvidenceService = {
  getNodePreview: (nodeId: string) => Promise<NodePreview>
  getPageReplay: (documentId: string, page: number) => Promise<PageReplay>
  getRetrievalHits: (
    retrievalRecordId: string,
    params: { source?: string; limit: number; offset: number },
  ) => Promise<EvidenceView['retrievalHits']>
}

export type ChatStoreState = {
  chat: ChatSessionView
  evidence: EvidenceView
  ui: {
    drawerOpen: boolean
    notice?: SystemNoticeView
    mockMode: ChatScenario
  }
  evidenceState: {
    selectedNodeId?: string
    nodePreviewStatus: EvidenceLoadStatus
    pageReplayStatus: EvidenceLoadStatus
    pageReplay?: PageReplay
    retrievalHitsStatus: EvidenceLoadStatus
    retrievalRecordId?: string
  }
}

const createEmptyChatView = (): ChatSessionView => ({
  history: { items: [] },
  citations: [],
  debug: { enabled: false },
})

const createEmptyEvidenceView = (): EvidenceView => ({
  retrievalHits: { items: [], page: 1, pageSize: 0, total: 0, availableSources: [] },
  evidenceTree: undefined,
  nodePreview: undefined,
})

const createInitialState = (): ChatStoreState => ({
  chat: createEmptyChatView(),
  evidence: createEmptyEvidenceView(),
  ui: {
    drawerOpen: false,
    notice: undefined,
    mockMode: 'ok',
  },
  evidenceState: {
    selectedNodeId: undefined,
    nodePreviewStatus: 'idle',
    pageReplayStatus: 'idle',
    pageReplay: undefined,
    retrievalHitsStatus: 'idle',
    retrievalRecordId: undefined,
  },
})

export const createChatStore = (services: { chatService: ChatService; evidenceService: EvidenceService }) => {
  let state: ChatStoreState = createInitialState()
  const listeners = new Set<() => void>()

  const emit = () => {
    for (const listener of listeners) {
      listener()
    }
  }

  const setState = (next: ChatStoreState) => {
    state = next
    emit()
  }

  const mergeState = (next: Partial<ChatStoreState>) => {
    state = { ...state, ...next }
    emit()
  }

  const applySnapshot = (snapshot: ChatServiceSnapshot) => {
    const retrievalRecordId = snapshot.chat.activeRun?.records?.retrievalRecordId
    const retrievalHitsStatus: EvidenceLoadStatus = retrievalRecordId
      ? 'loading'
      : snapshot.evidence.retrievalHits.items.length > 0
        ? 'loaded'
        : 'idle'
    const runNotice =
      toEvaluatorNotice(snapshot.chat.activeRun?.evaluatorSummary) ??
      toEvaluatorStepNotice(snapshot.chat.activeRun?.steps) ??
      toGateNotice(snapshot.chat.activeRun?.steps)
    setState({
      ...state,
      chat: snapshot.chat,
      evidence: snapshot.evidence,
      ui: {
        ...state.ui,
        notice: runNotice,
      },
      evidenceState: {
        ...state.evidenceState,
        selectedNodeId: undefined,
        nodePreviewStatus: 'idle',
        pageReplayStatus: 'idle',
        pageReplay: undefined,
        retrievalHitsStatus,
        retrievalRecordId,
      },
    })
    if (retrievalRecordId) {
      void fetchRetrievalHits(retrievalRecordId)
    }
  }

  const setNotice = (notice: SystemNoticeView) => {
    mergeState({
      ui: {
        ...state.ui,
        notice,
      },
    })
  }

  const raiseNotice = (error: unknown) => {
    setNotice(toSystemNotice(error))
  }

  const dismissNotice = () => {
    mergeState({
      ui: {
        ...state.ui,
        notice: undefined,
      },
    })
  }

  const sendUserMessage = async (text: string, options: Omit<ChatSendOptions, 'mode' | 'history'> = {}) => {
    const userMessage: ChatMessageView = {
      id: `user_${Date.now().toString(36)}`,
      role: 'user',
      content: text,
    }
    const nextHistory = [...state.chat.history.items, userMessage]
    const pendingMessage: ChatMessageView = {
      id: `assistant_pending_${Date.now().toString(36)}`,
      role: 'assistant',
      content: 'Waiting for response...',
    }
    mergeState({
      chat: {
        ...state.chat,
        history: { items: [...nextHistory, pendingMessage] },
      },
      ui: {
        ...state.ui,
        notice: undefined,
      },
    })
    try {
      const snapshot = await services.chatService.sendMessage(text, {
        mode: state.ui.mockMode,
        conversationId: options.conversationId,
        kbId: options.kbId,
        debug: options.debug,
        history: nextHistory,
      })
      applySnapshot(snapshot)
    } catch (error) {
      mergeState({
        chat: {
          ...state.chat,
          history: { items: nextHistory },
        },
      })
      raiseNotice(error)
    }
  }

  const triggerBackendError = async (options: { conversationId?: string } = {}) => {
    const conversationId = options.conversationId ?? 'missing-conversation'
    try {
      await services.chatService.sendMessage('__inject_error__', {
        conversationId,
      })
    } catch (error) {
      raiseNotice(error)
    }
  }

  const setMockMode = async (mode: ChatScenario) => {
    mergeState({
      ui: {
        ...state.ui,
        mockMode: mode,
      },
    })

    if (!services.chatService.getSnapshot) return

    try {
      const snapshot = await services.chatService.getSnapshot(mode)
      applySnapshot(snapshot)
    } catch (error) {
      raiseNotice(error)
    }
  }

  const loadConversationHistory = async (conversationId: string) => {
    if (!services.chatService.getHistory) return
    try {
      const snapshot = await services.chatService.getHistory(conversationId)
      applySnapshot(snapshot)
    } catch (error) {
      raiseNotice(error)
    }
  }

  const toggleDrawer = (open?: boolean) => {
    const nextOpen = open ?? !state.ui.drawerOpen
    mergeState({
      ui: {
        ...state.ui,
        drawerOpen: nextOpen,
      },
    })
  }

  const fetchNodePreview = async (nodeId: string) => {
    mergeState({
      evidence: {
        ...state.evidence,
        nodePreview: undefined,
      },
      evidenceState: {
        ...state.evidenceState,
        selectedNodeId: nodeId,
        nodePreviewStatus: 'loading',
      },
    })

    try {
      const preview = await services.evidenceService.getNodePreview(nodeId)
      setState({
        ...state,
        evidence: {
          ...state.evidence,
          nodePreview: preview,
        },
        evidenceState: {
          ...state.evidenceState,
          nodePreviewStatus: 'loaded',
        },
      })
    } catch (error) {
      mergeState({
        evidenceState: {
          ...state.evidenceState,
          nodePreviewStatus: 'failed',
        },
      })
      raiseNotice(error)
    }
  }

  const fetchPageReplay = async (documentId: string, page: number) => {
    mergeState({
      evidenceState: {
        ...state.evidenceState,
        pageReplayStatus: 'loading',
        pageReplay: undefined,
      },
    })

    try {
      const replay = await services.evidenceService.getPageReplay(documentId, page)
      setState({
        ...state,
        evidenceState: {
          ...state.evidenceState,
          pageReplayStatus: 'loaded',
          pageReplay: replay,
        },
      })
    } catch (error) {
      mergeState({
        evidenceState: {
          ...state.evidenceState,
          pageReplayStatus: 'failed',
        },
      })
      raiseNotice(error)
    }
  }

  const fetchRetrievalHits = async (
    retrievalRecordId?: string,
    params: { source?: string; limit?: number; offset?: number } = {},
  ) => {
    const activeRecordId = retrievalRecordId ?? state.evidenceState.retrievalRecordId
    if (!activeRecordId) return
    const limit = params.limit ?? 20
    const offset = params.offset ?? 0
    mergeState({
      evidenceState: {
        ...state.evidenceState,
        retrievalHitsStatus: 'loading',
      },
    })

    try {
      const hits = await services.evidenceService.getRetrievalHits(activeRecordId, {
        source: params.source,
        limit,
        offset,
      })
      setState({
        ...state,
        evidence: {
          ...state.evidence,
          retrievalHits: hits,
        },
        evidenceState: {
          ...state.evidenceState,
          retrievalHitsStatus: 'loaded',
          retrievalRecordId: activeRecordId,
        },
      })
    } catch (error) {
      mergeState({
        evidenceState: {
          ...state.evidenceState,
          retrievalHitsStatus: 'failed',
        },
      })
      raiseNotice(error)
    }
  }

  const selectCitation = (nodeId: string) => {
    mergeState({
      ui: {
        ...state.ui,
        drawerOpen: true,
      },
      evidenceState: {
        ...state.evidenceState,
        selectedNodeId: nodeId,
      },
    })
    void fetchNodePreview(nodeId)
  }

  return {
    getState: () => state,
    setState: mergeState,
    subscribe: (listener: () => void) => {
      listeners.add(listener)
      return () => {
        listeners.delete(listener)
      }
    },
    reset: () => {
      setState(createInitialState())
    },
    sendUserMessage,
    setMockMode,
    toggleDrawer,
    selectCitation,
    fetchNodePreview,
    fetchPageReplay,
    fetchRetrievalHits,
    setNotice,
    raiseNotice,
    dismissNotice,
    triggerBackendError,
    loadConversationHistory,
  }
}

export type ChatStore = ReturnType<typeof createChatStore>
