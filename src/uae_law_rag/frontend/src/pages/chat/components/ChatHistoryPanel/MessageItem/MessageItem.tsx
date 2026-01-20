//docstring
// 职责: 单条消息的结构容器，承载用户与助手气泡布局。
// 边界: 不处理消息状态机与业务判断。
// 上游关系: ChatHistoryPanel。
// 下游关系: UserBubble, AssistantBubble。
import AssistantBubble from '@/pages/chat/components/ChatHistoryPanel/MessageItem/AssistantBubble/AssistantBubble'
import UserBubble from '@/pages/chat/components/ChatHistoryPanel/MessageItem/UserBubble'
import { useChatStore } from '@/stores/use_chat_store'

type MessageItemProps = {
  message: { id: string; role: 'user' | 'assistant'; text: string; runId?: string }
}

const MessageItem = ({ message }: MessageItemProps) => {
  const { runsById, ui } = useChatStore()

  if (message.role === 'user') {
    return (
      <div className="message-item">
        <UserBubble text={message.text} />
      </div>
    )
  }

  const run = message.runId ? runsById[message.runId] : undefined
  return (
    <div className="message-item">
      <AssistantBubble run={run} debugOpen={ui.debugOpen} />
    </div>
  )
}

export default MessageItem
