// src/types/index.ts
//docstring
// 职责: types 根出口；提供稳定导入面（@/types），并在此处消解 domain/http/ui 的同名导出冲突。
// 边界: 只做类型 re-export 与必要的别名（alias）；不引入运行时代码、不做任何类型推断/转换。
// 上游关系: src/types/domain/*, src/types/http/*, src/types/ui/*。
// 下游关系: services/*, stores/*, pages/*, components/*（通过 @/types 统一导入）。

// 1) domain：保持星号导出（若后续与 ui 冲突，应在此处显式消解）
export * from './domain'

// 2) ui：通常不应与 domain 冲突；若冲突，按同样方式显式重导出
export * from './ui'

// 3) http：避免 export * 触发“同名导出”冲突 —— 显式列出并在必要处 alias
export type {
    // ---- DTO：传输层结构（按字母序/语义分组维护） ----
    ChatContextConfigDTO,
    ChatDebugEnvelopeDTO,
    ChatGateSummaryDTO,
    ChatRequestDTO,
    ChatResponseDTO,
    ChatTimingMsDTO,
    CitationViewDTO,
    DebugEnvelopeDTO,
    DebugRecordsDTO,
    ErrorInfoDTO,
    ErrorResponseDTO,
    EvaluatorSummaryDTO,
    // ---- 冲突项：用 alias 消歧（仅对同名导出做 alias） ----
    ChatStatus as HttpChatStatus,
    EvaluatorStatus as HttpEvaluatorStatus,
    IngestProfileDTO, IngestRequestDTO,
    IngestResponseDTO, IngestStatus, IngestTimingMsDTO
} from './http'
