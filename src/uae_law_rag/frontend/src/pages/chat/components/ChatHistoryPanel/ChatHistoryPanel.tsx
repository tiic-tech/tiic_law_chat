//docstring
// 职责: 渲染对话历史列表的容器。
// 边界: 不负责消息加载与排序规则。
// 上游关系: src/pages/chat/ChatPage.tsx。
// 下游关系: MessageItem。
import type { ChatMessageView } from '@/types/ui'
import MessageItem from '@/pages/chat/components/ChatHistoryPanel/MessageItem/MessageItem'

type ChatHistoryPanelProps = {
  items: ChatMessageView[]
  onClickCitation: (nodeId: string) => void
}

const ChatHistoryPanel = ({ items, onClickCitation }: ChatHistoryPanelProps) => {
  return (
    <section className="chat-history-panel" aria-live="polite">
      {items.map((message) => (
        <MessageItem key={message.id} message={message} onClickCitation={onClickCitation} />
      ))}
    </section>
  )
}

export default ChatHistoryPanel
