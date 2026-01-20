//docstring
// 职责: 应用挂载入口，绑定根节点与运行时模式。
// 边界: 不负责路由/业务/状态管理。
// 上游关系: index.html 提供 #root。
// 下游关系: src/app/App.tsx。
import { env } from '@/config/env'
import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import App from './app/App'
import './styles/globals.css'

if (import.meta.env.DEV) {
  console.log('[env] VITE_API_BASE =', env.apiBase)
  console.log('[env] VITE_BACKEND_TARGET =', env.backendTarget)
}

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
  </StrictMode>,
)
