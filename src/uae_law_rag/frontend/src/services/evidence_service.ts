//docstring
// 职责: Evidence 视图模型生成的占位入口。
// 边界: 不进行 HTTP 调用，不做检索逻辑。
// 上游关系: pages/chat 或未来 orchestrator 调用。
// 下游关系: types/ui/evidence_view.ts。
import type { EvidencePanelView } from '@/types/ui/evidence_view'

export const buildEvidencePanelView = (): EvidencePanelView => {
  return {
    hits: [],
  }
}
