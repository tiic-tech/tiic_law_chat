// src/pages/chat/ChatPage.tsx
//docstring
// 职责: Chat 页面编排入口，组织历史、证据与输入面板，并作为容器协调 store actions。
// 边界: 不直接调用 api/endpoints；不解析 HTTP DTO；仅渲染 UI 视图并触发上层 actions。
// 上游关系: src/app/layout/AppShell.tsx。
// 下游关系: src/pages/chat/components/*。
import type { PageReplay } from '@/types/domain/evidence'
import type { ChatSessionView, EvidenceView } from '@/types/ui'
import ChatHistoryPanel from '@/pages/chat/components/ChatHistoryPanel/ChatHistoryPanel'
import EvidencePanel from '@/pages/chat/components/EvidencePanel/EvidencePanel'
import InputPanel from '@/pages/chat/components/InputPanel/InputPanel'
import SystemNoticeBar from '@/pages/chat/components/SystemNoticeBar/SystemNoticeBar'
import { Drawer } from '@/ui/components'

type LoadStatus = 'idle' | 'loading' | 'failed' | 'loaded'

type ChatPageProps = {
  chat: ChatSessionView
  evidence: EvidenceView
  ui: {
    drawerOpen: boolean
    notice?: { level: 'info' | 'warning' | 'error'; title: string; detail?: string }
    mockMode: 'ok' | 'no_debug' | 'empty' | 'error'
    selectedNodeId?: string
    nodePreviewStatus: LoadStatus
    pageReplayStatus: LoadStatus
    pageReplay?: PageReplay
  }
  onSend: (query: string) => Promise<void>
  onToggleDrawer: (open?: boolean) => void
  onSelectCitation: (nodeId: string) => void
  onSelectNode: (nodeId: string) => void
  onReplayPage: (documentId: string, page: number) => void
  onChangeMockMode: (mode: 'ok' | 'no_debug' | 'empty' | 'error') => void
  onDismissNotice: () => void
  onInjectError: () => void
}

const ChatPage = ({
  chat,
  evidence,
  ui,
  onSend,
  onToggleDrawer,
  onSelectCitation,
  onSelectNode,
  onReplayPage,
  onChangeMockMode,
  onDismissNotice,
  onInjectError,
}: ChatPageProps) => {
  return (
    <div className="chat-page">
      <SystemNoticeBar notice={ui.notice} onDismiss={onDismissNotice} />
      <div className="chat-page__header">
        <div>
          <div className="chat-page__title">Summary</div>
          <div className="chat-page__subtitle">Mock domain driven UI snapshot</div>
        </div>
        <div className="chat-page__actions">
          <label className="chat-page__mode">
            <span>Mock Mode</span>
            <select
              value={ui.mockMode}
              onChange={(event) => onChangeMockMode(event.target.value as ChatPageProps['ui']['mockMode'])}
            >
              <option value="ok">Loaded</option>
              <option value="no_debug">No debug</option>
              <option value="empty">No run</option>
              <option value="error">Service error</option>
            </select>
          </label>
          <button className="chat-page__action" type="button" onClick={onInjectError}>
            Inject Error
          </button>
          <button
            className="chat-page__action chat-page__action--primary"
            type="button"
            onClick={() => onToggleDrawer(!ui.drawerOpen)}
          >
            {ui.drawerOpen ? 'Close Evidence' : 'Open Evidence'}
          </button>
        </div>
      </div>
      <div className="chat-page__body">
        <div className="chat-page__history">
          <ChatHistoryPanel items={chat.history.items} onClickCitation={onSelectCitation} />
        </div>
        <InputPanel onSend={onSend} />
      </div>
      <Drawer open={ui.drawerOpen} title="Evidence" onClose={() => onToggleDrawer(false)}>
        <EvidencePanel
          activeRun={chat.activeRun}
          evidence={evidence}
          selectedNodeId={ui.selectedNodeId}
          nodePreviewStatus={ui.nodePreviewStatus}
          pageReplayStatus={ui.pageReplayStatus}
          pageReplay={ui.pageReplay}
          onSelectNode={onSelectNode}
          onReplayPage={onReplayPage}
        />
      </Drawer>
    </div>
  )
}

export default ChatPage
