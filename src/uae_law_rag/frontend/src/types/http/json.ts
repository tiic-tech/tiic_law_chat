// src/types/http/json.ts
//docstring
// 职责: 定义 HTTP 传输层可序列化 JSON 类型（JsonValue/JsonObject/JsonArray），供 DTO 的扩展字段约束与 request body 使用。
// 边界: 仅类型定义；不得包含运行时代码；不得依赖 domain/ui。
// 上游关系: 无。
// 下游关系: types/http/* DTO 定义、api/http.ts（request body 类型约束）。
export type JsonPrimitive = string | number | boolean | null
export type JsonValue = JsonPrimitive | JsonObject | JsonArray
export type JsonObject = { [key: string]: JsonValue }
export type JsonArray = JsonValue[]
// 关键：为 TypeScript 可选字段/省略字段场景提供“JSON-like”值类型（允许 undefined）
// - 用途：DTO 的 index signature、debug.records 等允许出现 undefined（序列化时会被忽略或由实现决定）
export type JsonValueLike = JsonValue | undefined
