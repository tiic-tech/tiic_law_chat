// src/pages/chat/ChatPage.tsx
//docstring
// 职责: Chat 页面编排入口，组织历史、证据与输入面板，并作为容器协调 store actions。
// 边界: 不直接调用 api/endpoints；不解析 HTTP DTO；仅组装 domain input 并触发 store action。
// 上游关系: src/app/layout/AppShell.tsx。
// 下游关系: src/pages/chat/components/*。
import { HttpError } from '@/api/http'
import ChatHistoryPanel from '@/pages/chat/components/ChatHistoryPanel/ChatHistoryPanel'
import EvidencePanel from '@/pages/chat/components/EvidencePanel/EvidencePanel'
import InputPanel from '@/pages/chat/components/InputPanel/InputPanel'
import SystemNoticeBar from '@/pages/chat/components/SystemNoticeBar/SystemNoticeBar'
import { chatStore } from '@/stores/chat_store'
import { useChatStore } from '@/stores/use_chat_store'

const ChatPage = () => {
  const { ui } = useChatStore()

  const handleSend = async (query: string) => {
    chatStore.setNotice(undefined)
    try {
      await chatStore.send(query, { kbId: 'default', debug: true })
    } catch (error) {
      if (error instanceof HttpError) {
        const info = error.info
        chatStore.setNotice({
          level: 'error',
          message: `HTTP ${info.status}: ${info.message}`,
          traceId: info.traceId,
          requestId: info.requestId,
        })
      } else {
        chatStore.setNotice({
          level: 'error',
          message: 'Unexpected error',
        })
      }
    }
  }

  return (
    <div className="chat-page" data-debug-open={ui.debugOpen}>
      <SystemNoticeBar />
      <div className="chat-page__body">
        <ChatHistoryPanel />
        <EvidencePanel />
      </div>
      <InputPanel onSend={handleSend} />
    </div>
  )
}

export default ChatPage
