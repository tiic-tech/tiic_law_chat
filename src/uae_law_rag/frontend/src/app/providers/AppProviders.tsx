//docstring
// 职责: 组合应用级 Provider 的占位层（Router/Store/Theme）。
// 边界: 不实现具体 Provider 逻辑，仅提供挂载点。
// 上游关系: src/app/App.tsx。
// 下游关系: 应用内所有页面与组件树。
import type { ReactNode } from 'react'

type AppProvidersProps = {
  children: ReactNode
}

const AppProviders = ({ children }: AppProvidersProps) => {
  return <>{children}</>
}

export default AppProviders
