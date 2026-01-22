//docstring
// 职责: 应用布局骨架，承载页面编排入口。
// 边界: 不包含业务请求与状态逻辑。
// 上游关系: src/app/App.tsx。
// 下游关系: 无（F0 工程壳占位）。

import ChatPageContainer, { type ChatTopbarActions } from '@/pages/chat/containers/ChatPageContainer'
import { listConversations } from '@/services/conversation_service'
import { useCallback, useMemo, useRef, useState } from 'react'

type NavItem = {
  id: string
  label: string
  group: 'Conversations'
}

const FALLBACK_NAV_ITEM: NavItem = { id: 'fallback', label: 'Conversation', group: 'Conversations' }

const buildPlaceholder = (index: number): NavItem => ({
  id: `new:${Date.now()}`,
  label: `New chat ${index}`,
  group: 'Conversations',
})

const AppShell = () => {
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false)
  const [navItems, setNavItems] = useState<NavItem[]>(() => [buildPlaceholder(1)])
  const [selectedNavId, setSelectedNavId] = useState(navItems[0]?.id ?? '')
  const [topbarActions, setTopbarActions] = useState<ChatTopbarActions | null>(null)
  const serviceMode = topbarActions?.mockMode === 'live' ? 'live' : 'mock'
  const lastServiceModeRef = useRef<'mock' | 'live'>(serviceMode)
  const selectedNavItem = useMemo(
    () => navItems.find((item) => item.id === selectedNavId) ?? navItems[0] ?? FALLBACK_NAV_ITEM,
    [navItems, selectedNavId],
  )
  const conversationItems = navItems

  const handleNewChat = () => {
    const nextIndex = navItems.length + 1
    const nextItem = buildPlaceholder(nextIndex)
    setNavItems((prev) => [nextItem, ...prev])
    setSelectedNavId(nextItem.id)
  }

  const refreshConversations = useCallback(async () => {
    const items = await listConversations()
    setNavItems((prev) => {
      const pending = prev.filter((item) => item.id.startsWith('new:'))
      const liveItems: NavItem[] = items.map((item) => ({
        id: item.conversationId,
        label: item.label,
        group: 'Conversations',
      }))
      const merged = [...pending, ...liveItems]
      setSelectedNavId((current) => {
        if (merged.some((item) => item.id === current)) return current
        return merged[0]?.id ?? ''
      })
      return merged
    })
  }, [])

  const handleConversationResolved = useCallback((placeholderId: string, conversationId: string) => {
    setNavItems((prev) => {
      const existing = prev.find((item) => item.id === conversationId)
      const nextItems = prev.filter((item) => item.id !== placeholderId)
      if (existing) return nextItems
      return [
        {
          id: conversationId,
          label: conversationId,
          group: 'Conversations',
        },
        ...nextItems,
      ]
    })
    setSelectedNavId(conversationId)
  }, [])

  const handleTopbarActionsChange = useCallback(
    (actions: ChatTopbarActions) => {
      setTopbarActions(actions)
      const nextServiceMode = actions.mockMode === 'live' ? 'live' : 'mock'
      if (lastServiceModeRef.current === nextServiceMode) return
      lastServiceModeRef.current = nextServiceMode
      if (nextServiceMode === 'live') {
        void refreshConversations()
        return
      }
      setNavItems((prev) => {
        const pending = prev.filter((item) => item.id.startsWith('new:'))
        const nextItems = pending.length > 0 ? pending : [buildPlaceholder(1)]
        if (
          pending.length === prev.length &&
          pending.every((item, index) => item.id === prev[index]?.id)
        ) {
          return prev
        }
        setSelectedNavId((current) => {
          if (nextItems.some((item) => item.id === current)) return current
          return nextItems[0]?.id ?? ''
        })
        return nextItems
      })
    },
    [refreshConversations],
  )

  return (
    <div className="app-shell" data-collapsed={sidebarCollapsed}>
      <aside className="app-shell__sidebar">
        <div className="sidebar__header">
          <button
            className="sidebar__toggle"
            type="button"
            aria-label="Toggle sidebar"
            onClick={() => setSidebarCollapsed((prev) => !prev)}
          >
            {sidebarCollapsed ? '>' : '<'}
          </button>
          <span className="sidebar__logo">UAE LAW RAG</span>
        </div>
        <div className="sidebar__search">
          <input className="sidebar__search-input" placeholder="Search" />
        </div>
        <button className="sidebar__new-chat" type="button" onClick={handleNewChat}>
          + New Chat
        </button>
        {conversationItems.length > 0 ? (
          <div className="sidebar__section">
            <div className="sidebar__label">Conversations</div>
            {conversationItems.map((item) => (
              <button
                key={item.id}
                className={`sidebar__item ${selectedNavId === item.id ? 'sidebar__item--active' : ''}`}
                type="button"
                onClick={() => setSelectedNavId(item.id)}
              >
                {item.label}
              </button>
            ))}
          </div>
        ) : null}
      </aside>
      <main className="app-shell__main">
        <header className="app-shell__topbar">
          <div>
            <div className="app-shell__title">{selectedNavItem.label}</div>
            <div className="app-shell__subtitle">{selectedNavItem.group}</div>
          </div>
          {topbarActions ? (
            <div className="app-shell__topbar-actions">
              <label className="chat-page__mode">
                <span>Mock Mode</span>
                <select
                  value={topbarActions.mockMode}
                  onChange={(event) =>
                    topbarActions.onChangeMockMode(
                      event.target.value as ChatTopbarActions['mockMode'],
                    )
                  }
                >
                  <option value="ok">Loaded</option>
                  <option value="no_debug">No debug</option>
                  <option value="empty">No run</option>
                  <option value="error">Service error</option>
                  <option value="live">Live</option>
                </select>
              </label>
              <button className="chat-page__action" type="button" onClick={topbarActions.onInjectError}>
                Inject Error
              </button>
              <button
                className="chat-page__action chat-page__action--primary"
                type="button"
                onClick={topbarActions.onToggleEvidence}
              >
                {topbarActions.drawerOpen ? 'Close Evidence' : 'Open Evidence'}
              </button>
            </div>
          ) : null}
        </header>
        <section className="app-shell__content">
          <div className="app-shell__window">
            <ChatPageContainer
              conversationId={selectedNavId}
              onTopbarActionsChange={handleTopbarActionsChange}
              onConversationResolved={handleConversationResolved}
            />
          </div>
        </section>
      </main>
    </div>
  )
}

export default AppShell
