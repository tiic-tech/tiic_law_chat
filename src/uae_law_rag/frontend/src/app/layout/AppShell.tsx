//docstring
// 职责: 应用布局骨架，承载页面编排入口。
// 边界: 不包含业务请求与状态逻辑。
// 上游关系: src/app/App.tsx。
// 下游关系: src/pages/chat/ChatPage.tsx。
import ChatPage from '@/pages/chat/ChatPage'

const AppShell = () => {
  return (
    <div className="app-shell">
      <ChatPage />
    </div>
  )
}

export default AppShell
