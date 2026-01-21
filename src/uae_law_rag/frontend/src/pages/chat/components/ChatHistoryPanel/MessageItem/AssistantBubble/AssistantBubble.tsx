//docstring
// 职责: 助手消息气泡容器，组织回答、评估与引用区块。
// 边界: 不进行 evaluator 或 citation 计算。
// 上游关系: MessageItem。
// 下游关系: AnswerBlock, EvaluatorBadge, CitationList。
import type { CitationView, EvaluatorBadgeView } from '@/types/ui'
import AnswerBlock from '@/pages/chat/components/ChatHistoryPanel/MessageItem/AssistantBubble/AnswerBlock'
import CitationList from '@/pages/chat/components/ChatHistoryPanel/MessageItem/AssistantBubble/CitationList'
import EvaluatorBadge from '@/pages/chat/components/ChatHistoryPanel/MessageItem/AssistantBubble/EvaluatorBadge'

type AssistantBubbleProps = {
  content: string
  citations: CitationView[]
  evaluatorBadge?: EvaluatorBadgeView
  onClickCitation: (nodeId: string) => void
}

const AssistantBubble = ({ content, citations, evaluatorBadge, onClickCitation }: AssistantBubbleProps) => {
  return (
    <div className="assistant-bubble">
      <AnswerBlock answer={content} />
      {evaluatorBadge && <EvaluatorBadge status={evaluatorBadge.label ?? evaluatorBadge.level} />}
      <CitationList citations={citations} onClickCitation={onClickCitation} />
    </div>
  )
}

export default AssistantBubble
