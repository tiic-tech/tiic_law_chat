import { apiClient } from '@/api/client'
import type { ConversationItemView } from '@/types/ui'

export const listConversations = async (): Promise<ConversationItemView[]> => {
  const items = await apiClient.getChatConversations()
  return items.map((item) => ({
    conversationId: item.conversation_id,
    label: item.conversation_id,
    createdAt: item.created_at ?? undefined,
    updatedAt: item.updated_at ?? undefined,
  }))
}
