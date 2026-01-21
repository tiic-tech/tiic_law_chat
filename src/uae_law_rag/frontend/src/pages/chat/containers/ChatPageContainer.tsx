// src/pages/chat/containers/ChatPageContainer.tsx
//docstring
// 职责: Chat 页面容器，读取 store 并向展示层传递 props。
// 边界: 不渲染具体 UI；仅负责数据与交互桥接。
// 上游关系: src/app/layout/AppShell.tsx。
// 下游关系: src/pages/chat/ChatPage.tsx。
import ChatPage from '@/pages/chat/ChatPage'
import { createMockChatService } from '@/pages/chat/mock/mock_chat_service'
import { createMockEvidenceService } from '@/pages/chat/mock/mock_evidence_service'
import { createChatStore } from '@/stores/chat_store'
import { useChatStore } from '@/stores/use_chat_store'
import { useEffect, useMemo } from 'react'

const USE_MOCK = true

const ChatPageContainer = () => {
  const services = useMemo(() => {
    if (USE_MOCK) {
      return {
        chatService: createMockChatService('ok'),
        evidenceService: createMockEvidenceService('ok'),
      }
    }

    return {
      chatService: createMockChatService('ok'),
      evidenceService: createMockEvidenceService('ok'),
    }
  }, [])

  const store = useMemo(() => createChatStore(services), [services])
  const state = useChatStore(store)

  useEffect(() => {
    void store.setMockMode('ok')
  }, [store])

  return (
    <ChatPage
      chat={state.chat}
      evidence={state.evidence}
      ui={{
        drawerOpen: state.ui.drawerOpen,
        notice: state.ui.notice,
        mockMode: state.ui.mockMode,
        selectedNodeId: state.evidenceState.selectedNodeId,
        nodePreviewStatus: state.evidenceState.nodePreviewStatus,
        pageReplayStatus: state.evidenceState.pageReplayStatus,
        pageReplay: state.evidenceState.pageReplay,
      }}
      onSend={store.sendUserMessage}
      onToggleDrawer={store.toggleDrawer}
      onSelectCitation={store.selectCitation}
      onSelectNode={store.fetchNodePreview}
      onReplayPage={store.fetchPageReplay}
      onChangeMockMode={store.setMockMode}
      onDismissNotice={store.dismissNotice}
      onInjectError={() => store.raiseNotice(new Error('Injected error'))}
    />
  )
}

export default ChatPageContainer
