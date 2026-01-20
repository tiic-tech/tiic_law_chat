//docstring
// 职责: 系统通知栏占位，展示全局状态或警告。
// 边界: 不生成通知内容。
// 上游关系: ChatPage。
// 下游关系: 无。
import { useChatStore } from '@/stores/use_chat_store'

const SystemNoticeBar = () => {
  const { notice } = useChatStore()

  if (!notice) return <div className="system-notice-bar" />

  return (
    <div className={`system-notice-bar system-notice-bar--${notice.level}`}>
      <div className="system-notice-bar__message">{notice.message}</div>
      {(notice.traceId || notice.requestId) && (
        <div className="system-notice-bar__meta">
          {notice.traceId && <span>traceId: {notice.traceId}</span>}
          {notice.requestId && <span>requestId: {notice.requestId}</span>}
        </div>
      )}
    </div>
  )
}

export default SystemNoticeBar
