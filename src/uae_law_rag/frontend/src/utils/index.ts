// src/utils/index.ts
//docstring
// 职责: utils 根出口；聚合并重导出基础工具函数（assert/format/logger），提供稳定导入面（@/utils）。
// 边界: 仅做 re-export；不得引入业务模块（services/stores/pages/components/api）。
// 上游关系: src/utils/{assert,format,logger}.ts。
// 下游关系: 全前端工程（以 @/utils 作为唯一入口）。
export * from './assert'
export * from './format'
export * from './logger'
