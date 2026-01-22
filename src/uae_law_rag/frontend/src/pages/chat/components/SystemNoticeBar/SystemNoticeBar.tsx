//docstring
// 职责: 系统通知栏占位，展示全局状态或警告。
// 边界: 不生成通知内容。
// 上游关系: ChatPage。
// 下游关系: 无。
import { useState } from 'react'
import type { SystemNoticeView } from '@/types/ui'

type SystemNoticeBarProps = {
  notice?: SystemNoticeView
  onDismiss?: () => void
}

const SystemNoticeBar = ({ notice, onDismiss }: SystemNoticeBarProps) => {
  const noticeKey = notice
    ? `${notice.kind}:${notice.title}:${notice.meta?.requestId ?? notice.meta?.traceId ?? ''}`
    : 'empty'
  const [detailsState, setDetailsState] = useState({ key: '', open: false })
  const isExpanded = detailsState.key === noticeKey && detailsState.open

  const metaParts: string[] = []
  if (notice?.meta?.status !== undefined) metaParts.push(`status ${notice.meta.status}`)
  if (notice?.meta?.requestId) metaParts.push(`request ${notice.meta.requestId}`)
  if (notice?.meta?.traceId) metaParts.push(`trace ${notice.meta.traceId}`)
  if (notice?.meta?.endpoint) metaParts.push(`endpoint ${notice.meta.endpoint}`)
  const metaLine = metaParts.join(' · ')

  const renderRaw = () => {
    if (!notice?.raw) return null
    if (typeof notice.raw === 'string') return notice.raw
    if (notice.raw instanceof Error) return notice.raw.stack ?? notice.raw.message
    try {
      return JSON.stringify(notice.raw, null, 2)
    } catch {
      return String(notice.raw)
    }
  }

  return (
    <div
      className={`system-notice-bar ${notice ? `system-notice-bar--${notice.level}` : 'system-notice-bar--empty'}`}
    >
      {notice ? (
        <>
          <div className="system-notice-bar__message">{notice.title}</div>
          {notice.detail && <div className="system-notice-bar__meta">{notice.detail}</div>}
          {metaLine ? <div className="system-notice-bar__meta">{metaLine}</div> : null}
          {notice.raw && isExpanded ? (
            <pre className="system-notice-bar__detail">{renderRaw()}</pre>
          ) : null}
          <div className="system-notice-bar__actions">
            {notice.raw ? (
              <button
                className="system-notice-bar__toggle"
                type="button"
                onClick={() => setDetailsState({ key: noticeKey, open: !isExpanded })}
              >
                {isExpanded ? 'Hide details' : 'Show details'}
              </button>
            ) : null}
            <button className="system-notice-bar__dismiss" type="button" onClick={() => onDismiss?.()}>
              Dismiss
            </button>
          </div>
        </>
      ) : (
        <span className="system-notice-bar__placeholder" aria-hidden="true" />
      )}
    </div>
  )
}

export default SystemNoticeBar
