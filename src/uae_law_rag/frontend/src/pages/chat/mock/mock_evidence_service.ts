import type { NodePreview, PageReplay, RetrievalHitsPaged } from '@/types/domain/evidence'
import { NODE_PREVIEW_FAIL_ID, NODE_PREVIEW_OK, PAGE_REPLAY_FAIL_KEY, PAGE_REPLAY_OK, RUN_OK } from '@/fixtures/mock_domain'

export type MockEvidenceMode = 'ok' | 'error'

const delay = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms))

export const createMockEvidenceService = (initialMode: MockEvidenceMode = 'ok') => {
  let mode: MockEvidenceMode = initialMode

  return {
    getMode: () => mode,
    setMode: (next: MockEvidenceMode) => {
      mode = next
    },
    getNodePreview: async (nodeId: string): Promise<NodePreview> => {
      await delay(420)
      if (mode === 'error' || nodeId === NODE_PREVIEW_FAIL_ID) {
        throw new Error('Mock node preview error')
      }
      return NODE_PREVIEW_OK
    },
    getPageReplay: async (documentId: string, page: number): Promise<PageReplay> => {
      await delay(520)
      const key = `${documentId}|${page}`
      if (mode === 'error' || key === PAGE_REPLAY_FAIL_KEY) {
        throw new Error('Mock page replay error')
      }
      return PAGE_REPLAY_OK
    },
    getRetrievalHits: async (): Promise<RetrievalHitsPaged> => {
      await delay(320)
      if (mode === 'error') {
        throw new Error('Mock retrieval hits error')
      }
      return RUN_OK.evidence.retrievalHitsPaged ?? { items: [], page: 1, pageSize: 0, total: 0 }
    },
  }
}
