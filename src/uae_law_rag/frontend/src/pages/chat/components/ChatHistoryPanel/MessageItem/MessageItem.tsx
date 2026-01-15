//docstring
// 职责: 单条消息的结构容器，承载用户与助手气泡布局。
// 边界: 不处理消息状态机与业务判断。
// 上游关系: ChatHistoryPanel。
// 下游关系: UserBubble, AssistantBubble。
import AssistantBubble from '@/pages/chat/components/ChatHistoryPanel/MessageItem/AssistantBubble/AssistantBubble'
import UserBubble from '@/pages/chat/components/ChatHistoryPanel/MessageItem/UserBubble'

const MessageItem = () => {
  return (
    <div className="message-item">
      <UserBubble />
      <AssistantBubble />
    </div>
  )
}

export default MessageItem
