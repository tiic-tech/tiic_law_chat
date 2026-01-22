// src/config/env.ts
//docstring
// 职责: 读取 Vite 环境变量并提供运行时可见的调试信息（仅开发期使用）。
// 边界: 不包含业务逻辑；不依赖 stores/services/api。
// 上游关系: Vite import.meta.env。
// 下游关系: app 启动时可选打印，辅助联调定位。
const metaEnv = (import.meta.env ?? {}) as Partial<ImportMetaEnv>

export const env = {
  apiBase: metaEnv.VITE_API_BASE ?? '/api',
  backendTarget: metaEnv.VITE_BACKEND_TARGET,
  serviceMode: metaEnv.VITE_SERVICE_MODE ?? 'mock',
}
