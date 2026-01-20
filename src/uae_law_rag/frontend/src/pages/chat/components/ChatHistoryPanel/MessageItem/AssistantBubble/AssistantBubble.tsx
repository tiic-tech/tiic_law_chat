//docstring
// 职责: 助手消息气泡容器，组织回答、评估与引用区块。
// 边界: 不进行 evaluator 或 citation 计算。
// 上游关系: MessageItem。
// 下游关系: AnswerBlock, EvaluatorBadge, CitationList, DebugToggle。
import AnswerBlock from '@/pages/chat/components/ChatHistoryPanel/MessageItem/AssistantBubble/AnswerBlock'
import CitationList from '@/pages/chat/components/ChatHistoryPanel/MessageItem/AssistantBubble/CitationList'
import DebugToggle from '@/pages/chat/components/ChatHistoryPanel/MessageItem/AssistantBubble/DebugToggle'
import EvaluatorBadge from '@/pages/chat/components/ChatHistoryPanel/MessageItem/AssistantBubble/EvaluatorBadge'
import { chatStore } from '@/stores/chat_store'
import type { ChatNormalizedResult } from '@/types/domain/chat'
import type { RunStatus } from '@/types/domain/run'

type AssistantBubbleProps = {
  run?: ChatNormalizedResult
  debugOpen: boolean
}

const mapStatusLabel = (status?: RunStatus) => {
  if (!status) return 'unknown'
  return status
}

const AssistantBubble = ({ run, debugOpen }: AssistantBubbleProps) => {
  const citations = run?.evidence.citations ?? []
  const debugMessage = run?.debug.available ? undefined : run?.debug.message

  return (
    <div className="assistant-bubble">
      <AnswerBlock answer={run?.answer ?? ''} />
      <EvaluatorBadge status={mapStatusLabel(run?.run.status)} />
      <CitationList
        citations={citations}
        onSelect={(nodeId) => chatStore.selectCitation(nodeId)}
      />
      <DebugToggle
        open={debugOpen}
        onToggle={() => chatStore.toggleDebug()}
        unavailableMessage={debugOpen ? debugMessage : undefined}
      />
    </div>
  )
}

export default AssistantBubble
