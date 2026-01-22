import type { CSSProperties } from 'react'
import type { SystemNoticeView } from '@/types/ui'
import { Drawer } from '@/ui/components'

type ErrorDrawerProps = {
  open: boolean
  stacked?: boolean
  notice?: SystemNoticeView
  onClose: () => void
  onDismiss: () => void
}

const ErrorDrawer = ({ open, stacked = false, notice, onClose, onDismiss }: ErrorDrawerProps) => {
  const hasError = notice?.level === 'error'
  const wrapperStyle: CSSProperties = {
    ['--drawer-offset' as string]: stacked ? '420px' : '0px',
  }
  const wrapperClass = `error-drawer__wrapper ${open ? 'error-drawer__wrapper--open' : ''}`

  return (
    <div className={wrapperClass} style={wrapperStyle}>
      <Drawer open={open} title="Errors" onClose={onClose}>
        <div className="error-drawer">
          {hasError ? (
            <>
              <div className="error-drawer__title">{notice?.title}</div>
              {notice?.detail ? <pre className="error-drawer__detail">{notice.detail}</pre> : null}
              <div className="error-drawer__actions">
                <button className="error-drawer__action" type="button" onClick={onDismiss}>
                  Dismiss
                </button>
              </div>
            </>
          ) : (
            <div className="error-drawer__empty">No errors.</div>
          )}
        </div>
      </Drawer>
    </div>
  )
}

export default ErrorDrawer
