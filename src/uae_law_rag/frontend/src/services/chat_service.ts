//docstring
// 职责: Chat 用例层，负责调用 API 并映射为 domain 结果。
// 边界: 不直接渲染 UI，不在组件内调用。
// 上游关系: pages/chat 通过交互触发。
// 下游关系: api/endpoints/chat.ts, stores/chat_store.ts。
import { postChat } from '@/api/endpoints/chat'
import { chatStore } from '@/stores/chat_store'
import type {
  ChatRequestDTO,
  ChatResponseDTO,
  CitationViewDTO,
  EvaluatorSummaryDTO,
} from '@/types/http/chat_response'
import type { ChatResult, Citation, EvaluatorSummary } from '@/types/domain/message'
import type { DebugEnvelope } from '@/types/domain/run'

const mapEvaluatorSummary = (
  evaluator: EvaluatorSummaryDTO,
): EvaluatorSummary => {
  return {
    status: evaluator.status,
    ruleVersion: evaluator.rule_version,
    warnings: evaluator.warnings,
  }
}

const mapCitation = (citation: CitationViewDTO): Citation => {
  return {
    nodeId: citation.node_id,
    rank: citation.rank,
    quote: citation.quote,
    page: citation.page,
    articleId: citation.article_id,
    sectionPath: citation.section_path,
    locator: citation.locator,
  }
}

const mapDebugEnvelope = (
  debug: ChatResponseDTO['debug'],
): DebugEnvelope | undefined => {
  if (!debug) {
    return undefined
  }

  const records = debug.records

  return {
    traceId: debug.trace_id,
    requestId: debug.request_id,
    records: {
      retrievalRecordId: records.retrieval_record_id,
      generationRecordId: records.generation_record_id,
      evaluationRecordId: records.evaluation_record_id,
      documentId: records.document_id,
    },
    timingMs: debug.timing_ms,
    gate: debug.gate,
  }
}

const mapChatResponse = (response: ChatResponseDTO): ChatResult => {
  return {
    conversationId: response.conversation_id,
    messageId: response.message_id,
    kbId: response.kb_id,
    status: response.status,
    answer: response.answer,
    citations: response.citations.map(mapCitation),
    evaluator: mapEvaluatorSummary(response.evaluator),
    timingMs: response.timing_ms,
    traceId: response.trace_id,
    requestId: response.request_id,
    debug: mapDebugEnvelope(response.debug),
  }
}

export const sendChat = async (payload: ChatRequestDTO): Promise<ChatResult> => {
  const response = await postChat(payload)
  const result = mapChatResponse(response)
  const current = chatStore.getState()

  chatStore.setState({
    results: [...current.results, result],
  })

  return result
}
