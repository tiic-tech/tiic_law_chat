import { afterEach, describe, expect, it, vi } from 'vitest'
import chatDebugFixture from '@/fixtures/chat_debug.json'
import { apiClient } from '@/api/client'
import { createLiveChatService } from '@/services/chat_service'
import type { ChatResponseDTO } from '@/types/http/chat_response'

describe('live smoke contract', () => {
  afterEach(() => {
    vi.restoreAllMocks()
  })

  it('sends a query and surfaces the answer', async () => {
    const fixture = chatDebugFixture as ChatResponseDTO
    vi.spyOn(apiClient, 'postChat').mockResolvedValue(fixture)

    const chatService = createLiveChatService()
    const snapshot = await chatService.sendMessage('hello', { conversationId: 'conv-1', history: [] })

    expect(snapshot.chat.history.items.length).toBeGreaterThan(0)
    const lastMessage = snapshot.chat.history.items[snapshot.chat.history.items.length - 1]!
    expect(lastMessage.role).toBe('assistant')
    expect(lastMessage.content).toContain(fixture.answer)
    expect(snapshot.chat.citations.length).toBeGreaterThan(0)
  })
})
