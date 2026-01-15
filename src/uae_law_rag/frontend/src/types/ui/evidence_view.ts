//docstring
// 职责: 定义 EvidencePanel 的 UI 视图模型。
// 边界: 不直接引用服务端原始结构。
// 上游关系: services/evidence_service.ts。
// 下游关系: EvidencePanel 组件。
export type EvidenceHitView = {
  id: string
  rank?: number
  score?: number
  source?: string
  page?: number
  articleId?: string
  sectionPath?: string
  excerpt?: string
}

export type EvidencePanelView = {
  hits: EvidenceHitView[]
  selectedNodeId?: string
  previewText?: string
}
