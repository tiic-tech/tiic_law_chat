import type { JsonValue } from '@/types/http/json'

export type ChatHistoryMessageDTO = {
  conversation_id: string
  message_id: string
  query: string
  answer: string
  status: string
  created_at?: string | null
}

export type ConversationSummaryDTO = {
  conversation_id: string
  name?: string | null
  created_at?: string | null
  updated_at?: string | null
  meta?: JsonValue
}
