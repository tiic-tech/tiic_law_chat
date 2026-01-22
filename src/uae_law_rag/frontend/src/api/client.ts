//docstring
// 职责: 聚合 API endpoint 暴露统一调用入口。
// 边界: 不处理业务状态，不做 DTO -> Domain 映射。
// 上游关系: services/*。
// 下游关系: src/api/endpoints/*。
import {
  getChatConversations,
  getChatMessages,
  getNodeRecord,
  getPageRecord,
  getPageRecordByNode,
  getRetrievalRecord,
  postChat,
  postIngest,
} from '@/api/endpoints'

export const apiClient = {
  getChatConversations,
  getChatMessages,
  getNode: getNodeRecord,
  getPageReplay: getPageRecord,
  getPageReplayByNode: getPageRecordByNode,
  getRetrievalRecord,
  postChat,
  postIngest,
}
