// src/uae_law_rag/frontend/src/types/ui/evidence_view.tx
//docstring
// 职责: 定义 EvidencePanel 的 UI 视图模型。
// 边界: 只依赖 domain；不直接引用服务端原始结构。
// 上游关系: services/evidence service (view builder).
// 下游关系: EvidencePanel 组件。
import type { EvidenceTreeNode, NodePreview } from '@/types/domain/evidence'

export type HitRow = {
  nodeId: string
  rank?: number
  score?: number
  source?: string
  page?: number
  articleId?: string
  sectionPath?: string
  excerpt?: string
}

export type EvidenceHitView = HitRow

export type RetrievalHitsView = {
  items: HitRow[]
  page: number
  pageSize: number
  total: number
  source?: string
  availableSources?: string[]
}

export type EvidenceView = {
  retrievalHits: RetrievalHitsView
  nodePreview?: NodePreview
  evidenceTree?: EvidenceTreeNode[]
}

export type EvidencePanelView = {
  hits: EvidenceHitView[]
  selectedNodeId?: string
  previewText?: string
}
