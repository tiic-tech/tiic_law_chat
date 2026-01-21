import { describe, expect, it, vi } from 'vitest'
import { createChatStore } from '@/stores/chat_store'
import type { ChatServiceSnapshot } from '@/stores/chat_store'
import type { ChatSessionView, EvidenceView } from '@/types/ui'
import type { NodePreview, PageReplay } from '@/types/domain/evidence'

const buildSnapshot = (): ChatServiceSnapshot => {
  const chat: ChatSessionView = {
    history: { items: [] },
    citations: [],
    debug: { enabled: false },
  }
  const evidence: EvidenceView = {
    retrievalHits: { items: [], page: 1, pageSize: 0, total: 0 },
    evidenceTree: undefined,
    nodePreview: undefined,
  }
  return { chat, evidence }
}

describe('chatStore (mocked services)', () => {
  it('loads snapshot when switching mock mode', async () => {
    const chatService = {
      sendMessage: vi.fn().mockResolvedValue(buildSnapshot()),
      getSnapshot: vi.fn().mockResolvedValue(buildSnapshot()),
    }
    const evidenceService = {
      getNodePreview: vi.fn(),
      getPageReplay: vi.fn(),
    }

    const store = createChatStore({ chatService, evidenceService })
    await store.setMockMode('ok')

    expect(chatService.getSnapshot).toHaveBeenCalledWith('ok')
    expect(store.getState().chat.history.items).toHaveLength(0)
  })

  it('fetches node preview and updates status', async () => {
    const preview: NodePreview = {
      nodeId: 'node-1',
      documentId: 'doc-1',
      meta: {},
      textExcerpt: 'excerpt',
    }
    const chatService = {
      sendMessage: vi.fn().mockResolvedValue(buildSnapshot()),
      getSnapshot: vi.fn().mockResolvedValue(buildSnapshot()),
    }
    const evidenceService = {
      getNodePreview: vi.fn().mockResolvedValue(preview),
      getPageReplay: vi.fn(),
    }

    const store = createChatStore({ chatService, evidenceService })
    await store.fetchNodePreview('node-1')

    expect(store.getState().evidence.nodePreview).toEqual(preview)
    expect(store.getState().evidenceState.nodePreviewStatus).toBe('loaded')
  })

  it('fetches page replay and updates status', async () => {
    const replay: PageReplay = {
      documentId: 'doc-1',
      page: 2,
      content: 'page content',
    }
    const chatService = {
      sendMessage: vi.fn().mockResolvedValue(buildSnapshot()),
      getSnapshot: vi.fn().mockResolvedValue(buildSnapshot()),
    }
    const evidenceService = {
      getNodePreview: vi.fn(),
      getPageReplay: vi.fn().mockResolvedValue(replay),
    }

    const store = createChatStore({ chatService, evidenceService })
    await store.fetchPageReplay('doc-1', 2)

    expect(store.getState().evidenceState.pageReplay).toEqual(replay)
    expect(store.getState().evidenceState.pageReplayStatus).toBe('loaded')
  })
})
