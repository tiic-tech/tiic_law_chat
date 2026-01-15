// src/utils/logger.ts
//docstring
// 职责: 提供统一的前端日志接口与最小实现（可在 M2+ 替换为结构化日志/telemetry）。
// 边界: 默认仅封装 console；不得直接耦合具体 UI，不得包含业务决策逻辑；不得吞掉异常。
// 上游关系: 无（基础工具层）。
// 下游关系: services/*（请求/normalize 调试）、pages/components（DebugToggle）、api/http（错误观测）。
export type LogLevel = 'debug' | 'info' | 'warn' | 'error'

export type LogMeta = Record<string, unknown>

export interface Logger {
    debug(message: string, meta?: LogMeta): void
    info(message: string, meta?: LogMeta): void
    warn(message: string, meta?: LogMeta): void
    error(message: string, meta?: LogMeta): void
}

function emit(level: LogLevel, message: string, meta?: LogMeta) {
    const payload = meta ? [message, meta] : [message]
    console[level](...payload)
}

export const logger: Logger = {
    debug: (m, meta) => emit('debug', m, meta),
    info: (m, meta) => emit('info', m, meta),
    warn: (m, meta) => emit('warn', m, meta),
    error: (m, meta) => emit('error', m, meta),
}
