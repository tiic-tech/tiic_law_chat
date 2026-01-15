//docstring
// 职责: 渲染对话历史列表的容器。
// 边界: 不负责消息加载与排序规则。
// 上游关系: src/pages/chat/ChatPage.tsx。
// 下游关系: MessageItem。
import MessageItem from '@/pages/chat/components/ChatHistoryPanel/MessageItem/MessageItem'

const ChatHistoryPanel = () => {
  return (
    <section className="chat-history-panel" aria-live="polite">
      <MessageItem />
    </section>
  )
}

export default ChatHistoryPanel
