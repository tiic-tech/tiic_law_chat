// src/utils/format.ts
//docstring
// 职责: 提供纯格式化函数（时间/大小/截断/snippet 等），统一前端展示层的基础文案与格式。
// 边界: 纯函数集合；不做 IO，不读写 store，不依赖 services/api，不包含业务规则（如 gate 判定）。
// 上游关系: 无（基础工具层）。
// 下游关系: pages/components（展示）、services（日志/调试摘要）。
export function truncate(text: string, maxLen: number): string {
    if (maxLen <= 0) return ''
    if (text.length <= maxLen) return text
    return text.slice(0, Math.max(0, maxLen - 1)) + '…'
}

export function formatMs(ms?: number): string {
    if (ms === undefined || ms === null || Number.isNaN(ms)) return '—'
    if (ms < 1000) return `${Math.round(ms)} ms`
    return `${(ms / 1000).toFixed(2)} s`
}

export function safeJsonStringify(value: unknown, space = 2): string {
    try {
        return JSON.stringify(value, null, space)
    } catch {
        return '"[Unserializable]"'
    }
}
