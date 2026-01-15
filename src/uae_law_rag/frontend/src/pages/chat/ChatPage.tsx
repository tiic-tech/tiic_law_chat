//docstring
// 职责: Chat 页面编排入口，组织历史、证据与输入面板。
// 边界: 不直接调用 API，不解析原始 DTO。
// 上游关系: src/app/layout/AppShell.tsx。
// 下游关系: src/pages/chat/components/*。
import ChatHistoryPanel from '@/pages/chat/components/ChatHistoryPanel/ChatHistoryPanel'
import EvidencePanel from '@/pages/chat/components/EvidencePanel/EvidencePanel'
import InputPanel from '@/pages/chat/components/InputPanel/InputPanel'
import SystemNoticeBar from '@/pages/chat/components/SystemNoticeBar/SystemNoticeBar'

const ChatPage = () => {
  return (
    <div className="chat-page">
      <SystemNoticeBar />
      <div className="chat-page__body">
        <ChatHistoryPanel />
        <EvidencePanel />
      </div>
      <InputPanel />
    </div>
  )
}

export default ChatPage
