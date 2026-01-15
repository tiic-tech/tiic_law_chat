//docstring
// 职责: debug 展示开关的占位入口。
// 边界: 不读取 debug 数据，不控制请求参数。
// 上游关系: AssistantBubble。
// 下游关系: EvidencePanel 或 Debug 面板由上层衔接。
const DebugToggle = () => {
  return <div className="debug-toggle" />
}

export default DebugToggle
