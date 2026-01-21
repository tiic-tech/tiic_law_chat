//docstring
// 职责: 页面回放内容展示容器。
// 边界: 不做高亮定位，仅渲染文本截断。
// 上游关系: EvidencePanel。
// 下游关系: 无。
import type { PageReplay as PageReplayRecord } from '@/types/domain/evidence'

type LoadStatus = 'idle' | 'loading' | 'failed' | 'loaded'

type PageReplayProps = {
  status: LoadStatus
  replay?: PageReplayRecord
}

const renderContent = (replay?: PageReplayRecord) => {
  if (!replay) return ''
  const maxChars = 800
  if (replay.content.length <= maxChars) return replay.content
  return `${replay.content.slice(0, maxChars)}...`
}

const PageReplay = ({ status, replay }: PageReplayProps) => {
  if (status === 'loading') return <div className="page-replay">Loading page replay...</div>
  if (status === 'failed') return <div className="page-replay">Failed to replay page.</div>
  if (status === 'loaded' && replay) {
    return (
      <div className="page-replay">
        <div className="page-replay__meta">
          {replay.documentId} - page {replay.page}
        </div>
        <div className="page-replay__content">{renderContent(replay)}</div>
      </div>
    )
  }

  return <div className="page-replay">No page replay loaded.</div>
}

export default PageReplay
