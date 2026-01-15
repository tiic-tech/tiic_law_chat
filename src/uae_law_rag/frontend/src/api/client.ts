//docstring
// 职责: 聚合 API endpoint 暴露统一调用入口。
// 边界: 不处理业务状态，不做 DTO -> Domain 映射。
// 上游关系: services/*。
// 下游关系: src/api/endpoints/*。
import { postChat } from '@/api/endpoints/chat'
import { postIngest } from '@/api/endpoints/ingest'

export const apiClient = {
  postChat,
  postIngest,
}
