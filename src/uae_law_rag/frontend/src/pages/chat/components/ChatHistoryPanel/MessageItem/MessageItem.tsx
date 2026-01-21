//docstring
// 职责: 单条消息的结构容器，承载用户与助手气泡布局。
// 边界: 不处理消息状态机与业务判断。
// 上游关系: ChatHistoryPanel。
// 下游关系: UserBubble, AssistantBubble。
import type { ChatMessageView } from '@/types/ui'
import AssistantBubble from '@/pages/chat/components/ChatHistoryPanel/MessageItem/AssistantBubble/AssistantBubble'
import UserBubble from '@/pages/chat/components/ChatHistoryPanel/MessageItem/UserBubble'

type MessageItemProps = {
  message: ChatMessageView
  onClickCitation: (nodeId: string) => void
}

const MessageItem = ({ message, onClickCitation }: MessageItemProps) => {
  if (message.role === 'user') {
    return (
      <div className="message-item">
        <UserBubble text={message.content} />
      </div>
    )
  }

  return (
    <div className="message-item">
      <AssistantBubble
        content={message.content}
        citations={message.citations ?? []}
        evaluatorBadge={message.evaluatorBadge}
        onClickCitation={onClickCitation}
      />
    </div>
  )
}

export default MessageItem
