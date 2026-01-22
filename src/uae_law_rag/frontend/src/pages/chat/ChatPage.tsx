// src/pages/chat/ChatPage.tsx
//docstring
// 职责: Chat 页面编排入口，组织历史、证据与输入面板，并作为容器协调 store actions。
// 边界: 不直接调用 api/endpoints；不解析 HTTP DTO；仅渲染 UI 视图并触发上层 actions。
// 上游关系: src/app/layout/AppShell.tsx。
// 下游关系: src/pages/chat/components/*。
import ChatHistoryPanel from '@/pages/chat/components/ChatHistoryPanel/ChatHistoryPanel'
import ErrorDrawer from '@/pages/chat/components/ErrorDrawer'
import EvidencePanel from '@/pages/chat/components/EvidencePanel/EvidencePanel'
import NodePreview from '@/pages/chat/components/EvidencePanel/NodePreview'
import PageReplayDrawer from '@/pages/chat/components/EvidencePanel/PageReplayDrawer'
import RetrievalHitsTable from '@/pages/chat/components/EvidencePanel/RetrievalHitsTable'
import InputPanel from '@/pages/chat/components/InputPanel/InputPanel'
import type { PageReplay } from '@/types/domain/evidence'
import type { ChatSessionView, EvidenceView, SystemNoticeView } from '@/types/ui'
import { Drawer } from '@/ui/components'

type LoadStatus = 'idle' | 'loading' | 'failed' | 'loaded'

type ChatPageProps = {
  chat: ChatSessionView
  evidence: EvidenceView
  retrievalHits: {
    items: EvidenceView['retrievalHits']['items']
    total: number
    source?: string
    availableSources: string[]
    status: LoadStatus
  }
  ui: {
    drawerOpen: boolean
    errorDrawerOpen: boolean
    notice?: SystemNoticeView
    selectedNodeId?: string
    nodePreviewStatus: LoadStatus
    pageReplayStatus: LoadStatus
    pageReplay?: PageReplay
    pageReplayOpen: boolean
    unresolvableCitation?: string
  }
  onSend: (query: string) => Promise<void>
  onToggleDrawer: (open?: boolean) => void
  onSelectCitation: (nodeId: string) => void
  onSelectNode: (nodeId: string) => void
  onChangeHitsSource: (source: string) => void
  onOpenPageReplay: () => void
  onClosePageReplay: () => void
  onDismissNotice: () => void
}

const ChatPage = ({
  chat,
  evidence,
  retrievalHits,
  ui,
  onSend,
  onToggleDrawer,
  onSelectCitation,
  onSelectNode,
  onChangeHitsSource,
  onOpenPageReplay,
  onClosePageReplay,
  onDismissNotice,
}: ChatPageProps) => {
  const canOpenPageReplay =
    ui.nodePreviewStatus === 'loaded' && Boolean(evidence.nodePreview?.page !== undefined)

  return (
    <div className="chat-page">
      <div className="chat-page__body">
        <div className="chat-page__history">
          <ChatHistoryPanel items={chat.history.items} onClickCitation={onSelectCitation} />
        </div>
        <div className="chat-page__input-shell">
          <InputPanel onSend={onSend} />
        </div>
      </div>
      <Drawer open={ui.drawerOpen} title="Evidence" onClose={() => onToggleDrawer(false)}>
        {ui.unresolvableCitation ? (
          <div className="evidence-panel__notice">{ui.unresolvableCitation}</div>
        ) : null}
        <div className="evidence-panel">
          <div className="evidence-panel__section">
            <div className="evidence-panel__heading">Node Preview</div>
            <NodePreview
              status={ui.nodePreviewStatus}
              selectedNodeId={ui.selectedNodeId}
              nodePreview={evidence.nodePreview}
            />
          </div>
          <div className="evidence-panel__section">
            <div className="evidence-panel__heading">Page Replay</div>
            <button
              className="page-replay__trigger"
              type="button"
              onClick={onOpenPageReplay}
              disabled={!canOpenPageReplay}
            >
              Open Page Replay
            </button>
          </div>
          <RetrievalHitsTable
            items={retrievalHits.items}
            total={retrievalHits.total}
            source={retrievalHits.source}
            availableSources={retrievalHits.availableSources}
            status={retrievalHits.status}
            onSelectRow={onSelectNode}
            onChangeSource={onChangeHitsSource}
          />
          <EvidencePanel
            evidence={evidence}
            onSelectNode={onSelectNode}
          />
        </div>
      </Drawer>
      <PageReplayDrawer
        open={ui.pageReplayOpen}
        status={ui.pageReplayStatus}
        replay={ui.pageReplay}
        onClose={onClosePageReplay}
      />
      <ErrorDrawer
        open={ui.errorDrawerOpen}
        stacked={ui.drawerOpen}
        notice={ui.notice}
        onClose={onDismissNotice}
        onDismiss={onDismissNotice}
      />
    </div>
  )
}

export default ChatPage
