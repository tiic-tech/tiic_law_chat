//docstring
// 职责: 助手消息气泡容器，组织回答、评估与引用区块。
// 边界: 不进行 evaluator 或 citation 计算。
// 上游关系: MessageItem。
// 下游关系: AnswerBlock, EvaluatorBadge, CitationList, DebugToggle。
import AnswerBlock from '@/pages/chat/components/ChatHistoryPanel/MessageItem/AssistantBubble/AnswerBlock'
import CitationList from '@/pages/chat/components/ChatHistoryPanel/MessageItem/AssistantBubble/CitationList'
import DebugToggle from '@/pages/chat/components/ChatHistoryPanel/MessageItem/AssistantBubble/DebugToggle'
import EvaluatorBadge from '@/pages/chat/components/ChatHistoryPanel/MessageItem/AssistantBubble/EvaluatorBadge'

const AssistantBubble = () => {
  return (
    <div className="assistant-bubble">
      <AnswerBlock />
      <EvaluatorBadge />
      <CitationList />
      <DebugToggle />
    </div>
  )
}

export default AssistantBubble
