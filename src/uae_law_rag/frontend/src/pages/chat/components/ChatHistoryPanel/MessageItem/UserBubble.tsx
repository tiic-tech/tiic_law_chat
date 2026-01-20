//docstring
// 职责: 用户消息气泡的展示壳。
// 边界: 不渲染真实内容，等待上层传入。
// 上游关系: MessageItem。
// 下游关系: 无。
type UserBubbleProps = {
  text: string
}

const UserBubble = ({ text }: UserBubbleProps) => {
  return <div className="user-bubble">{text}</div>
}

export default UserBubble
