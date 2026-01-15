//docstring
// 职责: 应用根组件，拼装全局 Providers 与应用壳。
// 边界: 不包含业务逻辑与 API 调用。
// 上游关系: src/main.tsx。
// 下游关系: src/app/providers/AppProviders.tsx, src/app/layout/AppShell.tsx。
import AppProviders from '@/app/providers/AppProviders'
import AppShell from '@/app/layout/AppShell'

const App = () => {
  return (
    <AppProviders>
      <AppShell />
    </AppProviders>
  )
}

export default App
