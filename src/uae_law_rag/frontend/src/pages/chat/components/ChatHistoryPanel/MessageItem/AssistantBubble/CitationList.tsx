//docstring
// 职责: 引用列表的容器占位。
// 边界: 不合并、不重排 citation。
// 上游关系: AssistantBubble。
// 下游关系: EvidencePanel 联动由上层实现。
import type { EvidenceCitation } from '@/types/domain/evidence'

type CitationListProps = {
  citations: EvidenceCitation[]
  onSelect: (nodeId: string) => void
}

const CitationList = ({ citations, onSelect }: CitationListProps) => {
  if (!citations.length) {
    return <div className="citation-list">No citations</div>
  }

  return (
    <div className="citation-list">
      {citations.map((citation) => (
        <button
          key={citation.nodeId}
          type="button"
          className="citation-list__item"
          onClick={() => onSelect(citation.nodeId)}
        >
          {citation.nodeId}
        </button>
      ))}
    </div>
  )
}

export default CitationList
