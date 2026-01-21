//docstring
// 职责: 应用布局骨架，承载页面编排入口。
// 边界: 不包含业务请求与状态逻辑。
// 上游关系: src/app/App.tsx。
// 下游关系: 无（F0 工程壳占位）。

import ChatPageContainer from '@/pages/chat/containers/ChatPageContainer'
import { useState } from 'react'

const AppShell = () => {
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false)

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
        <div className="sidebar__section">
          <div className="sidebar__label">Projects</div>
          <button className="sidebar__item sidebar__item--active" type="button">
            tiic_law_chat
          </button>
          <button className="sidebar__item" type="button">
            tiic-rag
          </button>
        </div>
        <div className="sidebar__section">
          <div className="sidebar__label">Workspace</div>
          <button className="sidebar__item" type="button">
            Git version guide
          </button>
          <button className="sidebar__item" type="button">
            System innovation notes
          </button>
        </div>
      </aside>
      <main className="app-shell__main">
        <header className="app-shell__topbar">
          <div>
            <div className="app-shell__title">M1 Frontend Strategy</div>
            <div className="app-shell__subtitle">tiic_law_chat</div>
          </div>
        </header>
        <section className="app-shell__content">
          <div className="app-shell__window">
            <ChatPageContainer />
          </div>
        </section>
      </main>
    </div>
  )
}

export default AppShell
