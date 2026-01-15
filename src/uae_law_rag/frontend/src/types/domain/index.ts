// src/types/domain/index.ts
//docstring
// 职责: Domain Types 出口；聚合并重导出前端领域模型（run/step/evidence/message/gate）。
// 边界: 仅做 re-export；domain 层类型不得依赖 http DTO，不包含任何传输层字段命名约束。
// 上游关系: src/types/domain/{run,step,evidence,message,gate}.ts。
// 下游关系: services/*（normalize 输出）、stores/*（持久化 Domain state）、ui/types（视图投影）。
export * from './evidence'
export * from './gate'
export * from './message'
export * from './run'
export * from './step'
