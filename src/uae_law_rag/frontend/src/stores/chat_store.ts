// src/stores/chat_store.ts
//docstring
// 职责: Chat 状态容器的最小实现（可替换为专业状态库）；负责持有 UI 视图与证据状态机。
// 边界: 不直接调用 api/endpoints；不解析 HTTP DTO（只消费 service 输出的 view/domain 结果）。
// 上游关系: services/chat_service.ts 或 mock services（由 container 注入）。
// 下游关系: pages/chat（读取状态与触发 actions）。
import type { NodePreview, PageReplay } from '@/types/domain/evidence'
import type { ChatSessionView, EvidenceView } from '@/types/ui'

export type SystemNotice = {
  level: 'info' | 'warning' | 'error'
  title: string
  detail?: string
}

export type EvidenceLoadStatus = 'idle' | 'loading' | 'failed' | 'loaded'

export type ChatScenario = 'ok' | 'no_debug' | 'empty' | 'error'

export type ChatServiceSnapshot = {
  chat: ChatSessionView
  evidence: EvidenceView
}

export type ChatService = {
  sendMessage: (text: string, options?: { mode?: ChatScenario }) => Promise<ChatServiceSnapshot>
  getSnapshot?: (mode?: ChatScenario) => Promise<ChatServiceSnapshot>
}

export type EvidenceService = {
  getNodePreview: (nodeId: string) => Promise<NodePreview>
  getPageReplay: (documentId: string, page: number) => Promise<PageReplay>
}

export type ChatStoreState = {
  chat: ChatSessionView
  evidence: EvidenceView
  ui: {
    drawerOpen: boolean
    notice?: SystemNotice
    mockMode: ChatScenario
  }
  evidenceState: {
    selectedNodeId?: string
    nodePreviewStatus: EvidenceLoadStatus
    pageReplayStatus: EvidenceLoadStatus
    pageReplay?: PageReplay
  }
}

const createEmptyChatView = (): ChatSessionView => ({
  history: { items: [] },
  citations: [],
  debug: { enabled: false },
})

const createEmptyEvidenceView = (): EvidenceView => ({
  retrievalHits: { items: [], page: 1, pageSize: 0, total: 0 },
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
  },
})

const buildNotice = (error: unknown): SystemNotice => {
  if (error instanceof Error) {
    return {
      level: 'error',
      title: error.message || 'Unexpected error',
      detail: error.stack,
    }
  }

  if (typeof error === 'string') {
    return {
      level: 'error',
      title: error,
    }
  }

  return {
    level: 'error',
    title: 'Unexpected error',
  }
}

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
    setState({
      ...state,
      chat: snapshot.chat,
      evidence: snapshot.evidence,
      evidenceState: {
        ...state.evidenceState,
        selectedNodeId: undefined,
        nodePreviewStatus: 'idle',
        pageReplayStatus: 'idle',
        pageReplay: undefined,
      },
    })
  }

  const raiseNotice = (error: unknown) => {
    mergeState({
      ui: {
        ...state.ui,
        notice: buildNotice(error),
      },
    })
  }

  const dismissNotice = () => {
    mergeState({
      ui: {
        ...state.ui,
        notice: undefined,
      },
    })
  }

  const sendUserMessage = async (text: string) => {
    try {
      const snapshot = await services.chatService.sendMessage(text, { mode: state.ui.mockMode })
      applySnapshot(snapshot)
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
    raiseNotice,
    dismissNotice,
  }
}

export type ChatStore = ReturnType<typeof createChatStore>
