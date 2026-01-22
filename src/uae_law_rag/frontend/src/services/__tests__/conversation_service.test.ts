import { afterEach, describe, expect, it, vi } from 'vitest'
import { apiClient } from '@/api/client'
import { listConversations } from '@/services/conversation_service'

describe('conversation service', () => {
  afterEach(() => {
    vi.restoreAllMocks()
  })

  it('maps conversation summaries to UI view', async () => {
    vi.spyOn(apiClient, 'getChatConversations').mockResolvedValue([
      {
        conversation_id: 'conv-1',
        name: 'First',
        created_at: '2024-01-01T00:00:00Z',
        updated_at: '2024-01-01T00:00:00Z',
      },
    ])

    const items = await listConversations()

    expect(items).toEqual([
      {
        conversationId: 'conv-1',
        label: 'conv-1',
        createdAt: '2024-01-01T00:00:00Z',
        updatedAt: '2024-01-01T00:00:00Z',
      },
    ])
  })
})
