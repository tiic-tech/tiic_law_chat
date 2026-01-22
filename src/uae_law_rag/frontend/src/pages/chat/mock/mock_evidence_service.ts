import type { NodePreview, PageReplay, RetrievalHit } from '@/types/domain/evidence'
import type { HitRow, RetrievalHitsView } from '@/types/ui'
import {
  NODE_PREVIEW_FAIL_ID,
  NODE_PREVIEW_OK,
  PAGE_REPLAY_FAIL_KEY,
  PAGE_REPLAY_OK,
  RUN_OK,
} from '@/fixtures/mock_domain'

export type MockEvidenceMode = 'ok' | 'error'

const delay = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms))

const mapHitRow = (hit: RetrievalHit): HitRow => {
  return {
    nodeId: hit.nodeId,
    source: hit.source,
    rank: hit.rank,
    score: hit.score,
    page: hit.locator?.page,
    articleId: hit.locator?.articleId,
    sectionPath: hit.locator?.sectionPath,
    excerpt: hit.locator?.start !== undefined ? `Offsets ${hit.locator.start}-${hit.locator.end}` : undefined,
  }
}

const getAvailableSources = (hits: RetrievalHit[]) => {
  return Array.from(new Set(hits.map((hit) => hit.source).filter((value): value is string => Boolean(value))))
}

export const createMockEvidenceService = (initialMode: MockEvidenceMode = 'ok') => {
  let mode: MockEvidenceMode = initialMode
  const allHits = RUN_OK.evidence.retrievalHitsPaged?.items ?? []

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
    getRetrievalHits: async (
      _retrievalRecordId: string,
      params: {
        source?: string
        limit: number
        offset: number
      },
    ): Promise<RetrievalHitsView> => {
      await delay(320)
      if (mode === 'error') {
        throw new Error('Mock retrieval hits error')
      }
      const { source, limit, offset } = params
      const filtered = source ? allHits.filter((hit) => hit.source === source) : allHits
      const slice = filtered.slice(offset, offset + limit)
      return {
        items: slice.map(mapHitRow),
        total: filtered.length,
        page: limit ? Math.floor(offset / limit) + 1 : 1,
        pageSize: limit,
        source,
        availableSources: getAvailableSources(allHits),
      }
    },
  }
}
