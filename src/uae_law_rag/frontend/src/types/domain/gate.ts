//docstring
// 职责: 定义 gate 状态的前端 domain 结构。
// 边界: 不推断 gate 逻辑，仅承载服务端摘要。
// 上游关系: services/chat_service.ts。
// 下游关系: UI 解释层。
export type GateSummary = {
  retrieval?: Record<string, unknown>
  generation?: Record<string, unknown>
  evaluator?: Record<string, unknown>
  [key: string]: unknown
}
