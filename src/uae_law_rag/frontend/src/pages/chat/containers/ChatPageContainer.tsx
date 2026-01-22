// src/pages/chat/containers/ChatPageContainer.tsx
//docstring
// 职责: Chat 页面容器，读取 store 并向展示层传递 props。
// 边界: 不渲染具体 UI；仅负责数据与交互桥接。
// 上游关系: src/app/layout/AppShell.tsx。
// 下游关系: src/pages/chat/ChatPage.tsx。
import ChatPage from '@/pages/chat/ChatPage'
import { createMockChatService } from '@/pages/chat/mock/mock_chat_service'
import { createMockEvidenceService } from '@/pages/chat/mock/mock_evidence_service'
import { createLiveChatService } from '@/services/chat_service'
import { createLiveEvidenceService } from '@/services/evidence_service'
import { getServiceMode, setServiceMode as setServiceModeOverride, type ServiceMode } from '@/services/service_mode'
import { createChatStore } from '@/stores/chat_store'
import { useChatStore } from '@/stores/use_chat_store'
import { useCallback, useEffect, useMemo, useState } from 'react'

type ChatScenario = 'ok' | 'no_debug' | 'empty' | 'error'
type ChatModeOption = ChatScenario | 'live'

export type ChatTopbarActions = {
  mockMode: ChatModeOption
  drawerOpen: boolean
  onChangeMockMode: (mode: ChatModeOption) => void
  onInjectError: () => void
  onToggleEvidence: () => void
}

type ChatPageContainerProps = {
  conversationId?: string
  onTopbarActionsChange?: (actions: ChatTopbarActions) => void
  onConversationResolved?: (placeholderId: string, conversationId: string) => void
}

const ChatPageContainer = ({ conversationId, onTopbarActionsChange, onConversationResolved }: ChatPageContainerProps) => {
  const [serviceMode, setServiceMode] = useState<ServiceMode>(() => getServiceMode())
  const mockServices = useMemo(
    () => ({
      chatService: createMockChatService('ok'),
      evidenceService: createMockEvidenceService('ok'),
    }),
    [],
  )
  const liveServices = useMemo(
    () => ({
      chatService: createLiveChatService(),
      evidenceService: createLiveEvidenceService(),
    }),
    [],
  )
  const mockStore = useMemo(() => createChatStore(mockServices), [mockServices])
  const liveStore = useMemo(() => createChatStore(liveServices), [liveServices])
  const store = serviceMode === 'live' ? liveStore : mockStore
  const state = useChatStore(store)
  const errorDrawerOpen = state.ui.notice?.level === 'error'
  const activeConversationId = conversationId ?? 'new:default'
  const [unresolvableByConversation, setUnresolvableByConversation] = useState<
    Record<string, string | undefined>
  >({})
  const [hitsSourceByConversation, setHitsSourceByConversation] = useState<Record<string, string>>({})
  const [pageReplayOpenByConversation, setPageReplayOpenByConversation] = useState<Record<string, boolean>>({})

  const conversationMode = useMemo<ChatScenario>(() => {
    if (activeConversationId.startsWith('new:')) return 'empty'
    return 'ok'
  }, [activeConversationId])

  const activeHitsSource =
    hitsSourceByConversation[activeConversationId] ?? state.evidence.retrievalHits.source ?? 'all'
  const activeUnresolvableCitation = unresolvableByConversation[activeConversationId]
  const activePageReplayOpen = pageReplayOpenByConversation[activeConversationId] ?? false

  useEffect(() => {
    store.toggleDrawer(false)
    if (serviceMode === 'mock') {
      void store.setMockMode(conversationMode)
      return
    }
    if (!activeConversationId || activeConversationId.startsWith('new:')) {
      store.reset()
      return
    }
    void store.loadConversationHistory(activeConversationId)
  }, [activeConversationId, conversationMode, serviceMode, store])

  const handleChangeHitsSource = useCallback(
    (source: string) => {
      setHitsSourceByConversation((prev) => ({ ...prev, [activeConversationId]: source }))
      void store.fetchRetrievalHits(undefined, { source: source === 'all' ? undefined : source })
    },
    [activeConversationId, store],
  )

  const handleSelectCitation = useCallback(
    (nodeId: string) => {
      if (!nodeId || nodeId.trim().length === 0) {
        setUnresolvableByConversation((prev) => ({
          ...prev,
          [activeConversationId]: 'Citation locator unavailable.',
        }))
        store.toggleDrawer(true)
        const current = store.getState()
        store.setState({
          evidence: { ...current.evidence, nodePreview: undefined },
          evidenceState: {
            ...current.evidenceState,
            selectedNodeId: undefined,
            nodePreviewStatus: 'idle',
          },
        })
        return
      }
      setUnresolvableByConversation((prev) => ({ ...prev, [activeConversationId]: undefined }))
      store.selectCitation(nodeId)
    },
    [activeConversationId, store],
  )

  const handleChangeMockMode = useCallback(
    async (mode: ChatModeOption) => {
      setHitsSourceByConversation((prev) => ({ ...prev, [activeConversationId]: 'all' }))
      setUnresolvableByConversation((prev) => ({ ...prev, [activeConversationId]: undefined }))
      setPageReplayOpenByConversation((prev) => ({ ...prev, [activeConversationId]: false }))
      if (mode === 'live') {
        setServiceModeOverride('live')
        setServiceMode('live')
        return
      }
      setServiceModeOverride('mock')
      setServiceMode('mock')
      await mockStore.setMockMode(mode)
    },
    [activeConversationId, mockStore],
  )

  const handleToggleEvidenceDrawer = useCallback(
    (open?: boolean) => {
      const nextOpen = open ?? !state.ui.drawerOpen
      if (!nextOpen) {
        setPageReplayOpenByConversation((prev) => ({ ...prev, [activeConversationId]: false }))
      }
      store.toggleDrawer(nextOpen)
    },
    [activeConversationId, state.ui.drawerOpen, store],
  )

  const handleOpenPageReplay = useCallback(() => {
    const preview = state.evidence.nodePreview
    if (!preview || preview.page === undefined) return
    setPageReplayOpenByConversation((prev) => ({ ...prev, [activeConversationId]: true }))
    void store.fetchPageReplay(preview.documentId, preview.page)
  }, [activeConversationId, state.evidence.nodePreview, store])

  const handleClosePageReplay = useCallback(() => {
    setPageReplayOpenByConversation((prev) => ({ ...prev, [activeConversationId]: false }))
  }, [activeConversationId])

  const handleInjectError = useCallback(() => {
    if (serviceMode === 'live') {
      void store.triggerBackendError({ conversationId: 'missing-conversation' })
      return
    }
    store.raiseNotice(new Error('Injected error'))
  }, [serviceMode, store])

  const handleDismissNotice = useCallback(() => {
    store.dismissNotice()
  }, [store])

  const handleSend = useCallback(
    async (query: string) => {
      const isPlaceholder = activeConversationId.startsWith('new:')
      await store.sendUserMessage(query, {
        conversationId: serviceMode === 'live' && isPlaceholder ? undefined : activeConversationId,
        debug: serviceMode === 'live',
      })
      if (serviceMode !== 'live' || !isPlaceholder) return
      const nextConversationId = store.getState().chat.activeRun?.conversationId
      if (!nextConversationId) return
      onConversationResolved?.(activeConversationId, nextConversationId)
    },
    [activeConversationId, onConversationResolved, serviceMode, store],
  )

  const topbarActions = useMemo<ChatTopbarActions>(
    () => ({
      mockMode: serviceMode === 'live' ? 'live' : state.ui.mockMode,
      drawerOpen: state.ui.drawerOpen,
      onChangeMockMode: handleChangeMockMode,
      onInjectError: handleInjectError,
      onToggleEvidence: handleToggleEvidenceDrawer,
    }),
    [
      handleChangeMockMode,
      handleInjectError,
      handleToggleEvidenceDrawer,
      serviceMode,
      state.ui.drawerOpen,
      state.ui.mockMode,
    ],
  )

  useEffect(() => {
    onTopbarActionsChange?.(topbarActions)
  }, [onTopbarActionsChange, topbarActions])

  const resolvedSource = activeHitsSource === 'all' ? undefined : activeHitsSource

  return (
    <ChatPage
      chat={state.chat}
      evidence={state.evidence}
      retrievalHits={{
        items: state.evidence.retrievalHits.items,
        total: state.evidence.retrievalHits.total,
        source: resolvedSource,
        availableSources: state.evidence.retrievalHits.availableSources ?? [],
        status: state.evidenceState.retrievalHitsStatus,
      }}
      ui={{
        drawerOpen: state.ui.drawerOpen,
        errorDrawerOpen,
        notice: state.ui.notice,
        selectedNodeId: state.evidenceState.selectedNodeId,
        nodePreviewStatus: state.evidenceState.nodePreviewStatus,
        pageReplayStatus: state.evidenceState.pageReplayStatus,
        pageReplay: state.evidenceState.pageReplay,
        pageReplayOpen: activePageReplayOpen && state.ui.drawerOpen,
        unresolvableCitation: activeUnresolvableCitation,
      }}
      onSend={handleSend}
      onToggleDrawer={handleToggleEvidenceDrawer}
      onSelectCitation={handleSelectCitation}
      onSelectNode={store.fetchNodePreview}
      onChangeHitsSource={handleChangeHitsSource}
      onOpenPageReplay={handleOpenPageReplay}
      onClosePageReplay={handleClosePageReplay}
      onDismissNotice={handleDismissNotice}
    />
  )
}

export default ChatPageContainer
