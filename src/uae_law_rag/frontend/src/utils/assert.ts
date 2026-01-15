// src/utils/assert.ts
//docstring
// 职责: 提供运行时断言与类型收窄工具（invariant/assertNever 等），用于提升可解释错误与类型安全。
// 边界: 仅包含无副作用的纯函数；不得依赖浏览器 API、不得依赖 services/stores/pages。
// 上游关系: 无（基础工具层）。
// 下游关系: services/*（normalize 断言）、stores/*（状态不变量）、pages/components（输入校验与防御式渲染）。
export class InvariantError extends Error {
    public override readonly name = 'InvariantError'
    constructor(message: string, public readonly meta?: Record<string, unknown>) {
        super(message)
    }
}

export function invariant(condition: unknown, message: string, meta?: Record<string, unknown>): asserts condition {
    if (!condition) throw new InvariantError(message, meta)
}

export function assertNever(x: never, message = 'Unreachable case'): never {
    throw new InvariantError(message, { value: x })
}
