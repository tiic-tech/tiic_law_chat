// src/types/domain/chat.ts
//docstring
// 职责: 定义 Chat 用例的 domain 输入类型（UI/store 使用的 camelCase 结构），与传输层 DTO 解耦。
// 边界: 仅表达领域输入字段，不包含 HTTP snake_case，不依赖 api/http/types/http。
// 上游关系: pages/chat（用户输入、上下文设置）。
// 下游关系: services/chat_service.ts（负责映射为 HTTP DTO 并发起请求）。
import type { EvidenceIndex } from '@/types/domain/evidence'
import type { EvaluatorSummary } from '@/types/domain/message'
import type { RunRecord } from '@/types/domain/run'
export type ChatContextInput = {
    keywordTopK?: number
    vectorTopK?: number
    fusionTopK?: number
    rerankTopK?: number
    fusionStrategy?: string
    rerankStrategy?: string

    embedProvider?: string
    embedModel?: string
    embedDim?: number

    modelProvider?: string
    modelName?: string

    promptName?: string
    promptVersion?: string

    evaluatorConfig?: Record<string, unknown>
    returnRecords?: boolean
    returnHits?: boolean

    extra?: Record<string, unknown>
}

export type ChatSendInput = {
    query: string
    conversationId?: string
    kbId?: string
    context?: ChatContextInput
    debug?: boolean
}

export type PromptDebugItem = {
    nodeId: string
    source?: string
    used: string
    chars: number
}

export type PromptDebug = {
    mode: string
    nodesUsed: number
    totalChars: number
    items: PromptDebugItem[]
}

export type KeywordStatsItem = {
    keyword: string
    recall?: number
    precision?: number
    overlap?: number
    counts?: {
        gtTotal?: number
        kwTotal?: number
    }
    capped?: boolean
}

export type KeywordStats = {
    rawQuery: string
    items: KeywordStatsItem[]
    meta?: Record<string, unknown>
}

export type ChatDebugState = {
    available: boolean
    message?: string
    promptDebug?: PromptDebug
    keywordStats?: KeywordStats
}

export type ChatNormalizedResult = {
    run: RunRecord
    evidence: EvidenceIndex
    answer?: string
    debug: ChatDebugState
    evaluator?: EvaluatorSummary
}
