// src/types/http/index.ts
//docstring
// 职责: HTTP DTO 出口；聚合并重导出与后端接口对齐的传输层类型（chat/ingest/error 等）。
// 边界: 仅做 re-export；http 层类型只表达字段结构，不做业务推断，不依赖 domain/ui。
// 上游关系: src/types/http/{chat_response,ingest_response,error_response}.ts。
// 下游关系: api/http.ts, api/endpoints/*（返回 DTO）；services/*（作为 normalize 输入）。
export * from './chat_response'
export * from './error_response'
export * from './chat_history'
export * from './ingest_response'
export * from './records_node_response'
export * from './records_page_response'
export * from './records_retrieval_response'

export type {
  JsonArray,
  JsonObject,
  JsonPrimitive,
  JsonValue,
  JsonValueLike,
} from './json'
