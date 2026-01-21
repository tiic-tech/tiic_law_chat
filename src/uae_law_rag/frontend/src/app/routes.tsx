//docstring
// 职责: 声明应用路由表的结构化配置。
// 边界: 不直接创建路由实例，不执行导航逻辑。
// 上游关系: AppShell 或未来 Router 层消费。
// 下游关系: src/pages/* 页面组件。
import type { ReactElement } from 'react'
import ChatPageContainer from '@/pages/chat/containers/ChatPageContainer'

type AppRoute = {
  id: string
  path: string
  element: ReactElement
}

export const appRoutes: AppRoute[] = [
  {
    id: 'chat',
    path: '/chat',
    element: <ChatPageContainer />,
  },
]
