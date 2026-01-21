//docstring
// 职责: 证据区容器，承载检索命中与节点预览。
// 边界: 不请求数据，不推断证据关系。
// 上游关系: ChatPage。
// 下游关系: NodePreview, PageReplay。
import type { ActiveRunView, EvidenceView } from '@/types/ui'
import type { PageReplay as PageReplayRecord, EvidenceTreeNode } from '@/types/domain/evidence'
import NodePreview from '@/pages/chat/components/EvidencePanel/NodePreview'
import PageReplay from '@/pages/chat/components/EvidencePanel/PageReplay'

type LoadStatus = 'idle' | 'loading' | 'failed' | 'loaded'

type EvidencePanelProps = {
  activeRun?: ActiveRunView
  evidence: EvidenceView
  selectedNodeId?: string
  nodePreviewStatus: LoadStatus
  pageReplayStatus: LoadStatus
  pageReplay?: PageReplayRecord
  onSelectNode: (nodeId: string) => void
  onReplayPage: (documentId: string, page: number) => void
}

const renderTree = (nodes: EvidenceTreeNode[], onSelectNode: (nodeId: string) => void) => {
  return (
    <ul className="evidence-tree">
      {nodes.map((node) => {
        const isLeaf = !node.children || node.children.length === 0
        return (
          <li key={node.id} className="evidence-tree__node">
            {isLeaf ? (
              <button
                type="button"
                className="evidence-tree__leaf"
                onClick={() => onSelectNode(node.id)}
              >
                {node.label}
              </button>
            ) : (
              <div className="evidence-tree__branch">{node.label}</div>
            )}
            {node.children && node.children.length > 0 && renderTree(node.children, onSelectNode)}
          </li>
        )
      })}
    </ul>
  )
}

const EvidencePanel = ({
  activeRun,
  evidence,
  selectedNodeId,
  nodePreviewStatus,
  pageReplayStatus,
  pageReplay,
  onSelectNode,
  onReplayPage,
}: EvidencePanelProps) => {
  let statusText = 'No active run'
  if (activeRun) {
    statusText = evidence.evidenceTree ? 'Evidence loaded' : 'Debug evidence unavailable'
  }

  return (
    <aside className="evidence-panel">
      <div className="evidence-panel__status">{statusText}</div>
      <div className="evidence-panel__section">
        <div className="evidence-panel__heading">Evidence Tree</div>
        {evidence.evidenceTree && evidence.evidenceTree.length > 0 ? (
          renderTree(evidence.evidenceTree, onSelectNode)
        ) : (
          <div className="evidence-panel__empty">No evidence tree available.</div>
        )}
      </div>
      <div className="evidence-panel__section">
        <div className="evidence-panel__heading">Node Preview</div>
        <NodePreview
          status={nodePreviewStatus}
          selectedNodeId={selectedNodeId}
          nodePreview={evidence.nodePreview}
          onReplayPage={onReplayPage}
        />
      </div>
      <div className="evidence-panel__section">
        <div className="evidence-panel__heading">Page Replay</div>
        <PageReplay status={pageReplayStatus} replay={pageReplay} />
      </div>
    </aside>
  )
}

export default EvidencePanel
