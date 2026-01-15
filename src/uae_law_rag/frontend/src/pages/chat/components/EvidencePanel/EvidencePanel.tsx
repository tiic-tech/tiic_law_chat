//docstring
// 职责: 证据区容器，承载检索命中与节点预览。
// 边界: 不请求数据，不推断证据关系。
// 上游关系: ChatPage。
// 下游关系: RetrievalHitsTable, NodePreview。
import NodePreview from '@/pages/chat/components/EvidencePanel/NodePreview'
import RetrievalHitsTable from '@/pages/chat/components/EvidencePanel/RetrievalHitsTable'

const EvidencePanel = () => {
  return (
    <aside className="evidence-panel">
      <RetrievalHitsTable />
      <NodePreview />
    </aside>
  )
}

export default EvidencePanel
