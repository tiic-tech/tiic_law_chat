import { describe, expect, it, vi } from 'vitest'
import { createChatStore } from '@/stores/chat_store'
import type { SystemNoticeView } from '@/types/ui'

describe('chatStore notice state', () => {
  it('sets and dismisses notice', () => {
    const chatService = {
      sendMessage: vi.fn(),
      getSnapshot: vi.fn(),
    }
    const evidenceService = {
      getNodePreview: vi.fn(),
      getPageReplay: vi.fn(),
      getRetrievalHits: vi.fn(),
    }

    const store = createChatStore({ chatService, evidenceService })
    const notice: SystemNoticeView = {
      level: 'warning',
      kind: 'unexpected',
      title: 'Heads up',
      detail: 'Something happened.',
    }

    store.setNotice(notice)
    expect(store.getState().ui.notice).toEqual(notice)

    store.dismissNotice()
    expect(store.getState().ui.notice).toBeUndefined()
  })
})
