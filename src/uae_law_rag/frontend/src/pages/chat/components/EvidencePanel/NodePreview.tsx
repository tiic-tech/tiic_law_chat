//docstring
// 职责: 证据节点内容预览的占位容器。
// 边界: 不做高亮逻辑，不做定位计算。
// 上游关系: EvidencePanel。
// 下游关系: 无。
import type { NodePreview as NodePreviewRecord } from '@/types/domain/evidence'

type LoadStatus = 'idle' | 'loading' | 'failed' | 'loaded'

type NodePreviewProps = {
  status: LoadStatus
  selectedNodeId?: string
  nodePreview?: NodePreviewRecord
  onReplayPage: (documentId: string, page: number) => void
}

const NodePreview = ({ status, selectedNodeId, nodePreview, onReplayPage }: NodePreviewProps) => {
  if (!selectedNodeId) {
    return <div className="node-preview">Select a node to preview.</div>
  }

  if (status === 'loading') {
    return <div className="node-preview">Loading node preview...</div>
  }

  if (status === 'failed') {
    return <div className="node-preview">Failed to load node preview.</div>
  }

  if (status === 'loaded' && nodePreview) {
    return (
      <div className="node-preview">
        <div className="node-preview__meta">
          <span>nodeId: {nodePreview.nodeId}</span>
          <span>documentId: {nodePreview.documentId}</span>
          {nodePreview.page !== undefined && <span>page: {nodePreview.page}</span>}
        </div>
        <div className="node-preview__offsets">
          <span>pageStart: {nodePreview.pageStartOffset ?? '-'}</span>
          <span>pageEnd: {nodePreview.pageEndOffset ?? '-'}</span>
          <span>nodeStart: {nodePreview.startOffset ?? '-'}</span>
          <span>nodeEnd: {nodePreview.endOffset ?? '-'}</span>
        </div>
        <div className="node-preview__excerpt">{nodePreview.textExcerpt}</div>
        <button
          className="node-preview__replay"
          type="button"
          onClick={() =>
            nodePreview.page !== undefined && onReplayPage(nodePreview.documentId, nodePreview.page)
          }
          disabled={nodePreview.page === undefined}
        >
          Replay page
        </button>
      </div>
    )
  }

  return <div className="node-preview">Select a node to preview.</div>
}

export default NodePreview
