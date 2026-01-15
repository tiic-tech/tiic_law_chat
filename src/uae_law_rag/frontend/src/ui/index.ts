// src/ui/index.ts
//docstring
// 职责: ui 根出口；聚合并重导出通用 UI 层（components/tokens 等），提供稳定导入面（@/ui）。
// 边界: 仅做 re-export；不得引入业务语义组件（pages/chat/...）。
// 上游关系: src/ui/components/*（与后续 tokens/*）。
// 下游关系: pages/components（复用通用 UI 原子组件）。
export * from './components';
