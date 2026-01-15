//docstring
// 职责: 应用挂载入口，绑定根节点与运行时模式。
// 边界: 不负责路由/业务/状态管理。
// 上游关系: index.html 提供 #root。
// 下游关系: src/app/App.tsx。
import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './styles/globals.css'
import App from './app/App'

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <App />
  </StrictMode>,
)
