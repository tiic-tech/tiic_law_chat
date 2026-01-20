//docstring
// 职责: 展示回答文本的占位容器。
// 边界: 不做回答合法性判断。
// 上游关系: AssistantBubble。
// 下游关系: 无。
type AnswerBlockProps = {
  answer: string
}

const AnswerBlock = ({ answer }: AnswerBlockProps) => {
  return <div className="answer-block">{answer}</div>
}

export default AnswerBlock
