import type { ReactNode } from 'react'
import { useEffect } from 'react'

type DrawerProps = {
  open: boolean
  title: string
  onClose: () => void
  children: ReactNode
}

const Drawer = ({ open, title, onClose, children }: DrawerProps) => {
  useEffect(() => {
    if (!open) return

    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        onClose()
      }
    }

    document.body.classList.add('drawer-open')
    window.addEventListener('keydown', handleKeyDown)

    return () => {
      document.body.classList.remove('drawer-open')
      window.removeEventListener('keydown', handleKeyDown)
    }
  }, [open, onClose])

  return (
    <div className={`drawer ${open ? 'drawer--open' : ''}`} aria-hidden={!open}>
      <div className="drawer__overlay" onClick={onClose} />
      <aside className="drawer__panel" role="dialog" aria-modal="true" aria-label={title}>
        <header className="drawer__header">
          <h2 className="drawer__title">{title}</h2>
          <button className="drawer__close" type="button" onClick={onClose}>
            Close
          </button>
        </header>
        <div className="drawer__content">{children}</div>
      </aside>
    </div>
  )
}

export default Drawer
