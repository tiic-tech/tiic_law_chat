// src/types/ui/index.ts
//docstring
// 职责: UI View Types 出口；聚合并重导出面向展示层的视图模型（chat_view/evidence_view）。
// 边界: 仅做 re-export；ui types 只依赖 domain（投影），不得依赖 http DTO，不包含后端字段名泄漏。
// 上游关系: src/types/ui/{chat_view,evidence_view}.ts。
// 下游关系: pages/*, components/*（props/视图模型），避免组件直接耦合 domain 细节。
export * from './chat_view'
export * from './evidence_view'
