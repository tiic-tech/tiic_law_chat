/// <reference types="vite/client" />

//docstring
// 职责: 为 Vite 的 import.meta.env 提供 TypeScript 类型声明（含自定义 VITE_* 变量）。
// 边界: 仅类型声明；不包含运行时代码。
// 上游关系: Vite 注入的环境变量。
// 下游关系: src/config/env.ts 等读取 env 的模块。

interface ImportMetaEnv {
    readonly VITE_API_BASE?: string
    readonly VITE_BACKEND_TARGET?: string
}

interface ImportMeta {
    readonly env: ImportMetaEnv
}
