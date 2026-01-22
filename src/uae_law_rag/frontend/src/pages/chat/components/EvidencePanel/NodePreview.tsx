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
}

const NodePreview = ({ status, selectedNodeId, nodePreview }: NodePreviewProps) => {
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
          <span>pageStart: {nodePreview.startOffset ?? '-'}</span>
          <span>pageEnd: {nodePreview.endOffset ?? '-'}</span>
          <span>nodeStart: {nodePreview.pageStartOffset ?? '-'}</span>
          <span>nodeEnd: {nodePreview.pageEndOffset ?? '-'}</span>
        </div>
        <div className="node-preview__excerpt">{nodePreview.textExcerpt}</div>
      </div>
    )
  }

  return <div className="node-preview">Select a node to preview.</div>
}

export default NodePreview
