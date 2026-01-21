//docstring
// 职责: 系统通知栏占位，展示全局状态或警告。
// 边界: 不生成通知内容。
// 上游关系: ChatPage。
// 下游关系: 无。

type NoticeView = {
  level: 'info' | 'warning' | 'error'
  title: string
  detail?: string
}

type SystemNoticeBarProps = {
  notice?: NoticeView
  onDismiss?: () => void
}

const SystemNoticeBar = ({ notice, onDismiss }: SystemNoticeBarProps) => {
  return (
    <div
      className={`system-notice-bar ${notice ? `system-notice-bar--${notice.level}` : 'system-notice-bar--empty'}`}
    >
      {notice ? (
        <>
          <div className="system-notice-bar__message">{notice.title}</div>
          {notice.detail && <div className="system-notice-bar__meta">{notice.detail}</div>}
          <button className="system-notice-bar__dismiss" type="button" onClick={() => onDismiss?.()}>
            Dismiss
          </button>
        </>
      ) : (
        <span className="system-notice-bar__placeholder" aria-hidden="true" />
      )}
    </div>
  )
}

export default SystemNoticeBar
