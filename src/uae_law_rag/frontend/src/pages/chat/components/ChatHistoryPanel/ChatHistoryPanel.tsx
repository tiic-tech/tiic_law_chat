//docstring
// 职责: 渲染对话历史列表的容器。
// 边界: 不负责消息加载与排序规则。
// 上游关系: src/pages/chat/ChatPage.tsx。
// 下游关系: MessageItem。
import MessageItem from '@/pages/chat/components/ChatHistoryPanel/MessageItem/MessageItem'
import { useChatStore } from '@/stores/use_chat_store'

const ChatHistoryPanel = () => {
  const { messages } = useChatStore()

  return (
    <section className="chat-history-panel" aria-live="polite">
      {messages.map((message) => (
        <MessageItem key={message.id} message={message} />
      ))}
    </section>
  )
}

export default ChatHistoryPanel
