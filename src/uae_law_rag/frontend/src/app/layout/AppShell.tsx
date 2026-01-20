//docstring
// 职责: 应用布局骨架，承载页面编排入口。
// 边界: 不包含业务请求与状态逻辑。
// 上游关系: src/app/App.tsx。
// 下游关系: 无（F0 工程壳占位）。

const AppShell = () => {
  return (
    <div className="app-shell">
      <header className="app-shell__header">
        <span className="app-shell__eyebrow">UAE Law RAG</span>
        <h1 className="app-shell__title">M1 Scaffold</h1>
        <p className="app-shell__subtitle">Frontend F0 base is locked and ready.</p>
      </header>
      <section className="app-shell__panel">
        <h2 className="app-shell__panel-title">Empty Shell</h2>
        <p className="app-shell__panel-caption">Business UI starts at F1-F7.</p>
      </section>
    </div>
  )
}

export default AppShell
