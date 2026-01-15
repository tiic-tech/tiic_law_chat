// src/ui/components/index.ts
//docstring
// 职责: ui/components 子模块出口；聚合并重导出通用 UI 原子组件（无业务语义）。
// 边界: 仅做 re-export；不得引入 pages/* 业务组件，不得依赖 services/stores/api/types/http。
// 上游关系: src/ui/components/*（Button/Badge/Table/CodeBlock 等将逐步加入）。
// 下游关系: pages/*, components/*（通过 @/ui/components 复用基础 UI 组件）。
export { };
