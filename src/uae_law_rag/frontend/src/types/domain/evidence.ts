//docstring
// 职责: 定义证据实体的 domain 结构。
// 边界: 不包含证据生成逻辑，仅承载数据。
// 上游关系: services/* 的 DTO 映射。
// 下游关系: EvidencePanel/NodePreview。
export type EvidenceType = 'node' | 'chunk' | 'doc' | 'citation'

export type EvidenceSource = 'retrieval' | 'rerank' | 'persist' | 'generation'

export type EvidenceProvenance = {
  docId?: string
  nodeId?: string
  page?: number
  offset?: number
  span?: [number, number]
  articleId?: string
  sectionPath?: string
}

export type Evidence = {
  evidenceId: string
  type: EvidenceType
  source: EvidenceSource
  payload: Record<string, unknown>
  provenance: EvidenceProvenance
}
