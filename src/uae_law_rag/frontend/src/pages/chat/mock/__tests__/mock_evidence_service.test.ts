import { describe, expect, it } from 'vitest'
import { createMockEvidenceService } from '@/pages/chat/mock/mock_evidence_service'
import { NODE_PREVIEW_FAIL_ID, PAGE_REPLAY_FAIL_KEY } from '@/fixtures/mock_domain'

describe('mock evidence service', () => {
  it('filters and paginates retrieval hits', async () => {
    const service = createMockEvidenceService('ok')
    const firstPage = await service.getRetrievalHits('mock', { source: 'keyword', limit: 5, offset: 0 })

    expect(firstPage.items).toHaveLength(5)
    expect(firstPage.total).toBeGreaterThan(5)
    expect(firstPage.source).toBe('keyword')
    expect(firstPage.items.every((item) => item.source === 'keyword')).toBe(true)
    expect(firstPage.page).toBe(1)
    expect(firstPage.pageSize).toBe(5)
    expect(firstPage.availableSources).toEqual(expect.arrayContaining(['keyword', 'vector']))

    const secondPage = await service.getRetrievalHits('mock', { source: 'keyword', limit: 5, offset: 5 })
    expect(secondPage.items).toHaveLength(5)
    expect(secondPage.total).toBe(firstPage.total)
    expect(secondPage.page).toBe(2)
    expect(secondPage.items[0]?.nodeId).not.toBe(firstPage.items[0]?.nodeId)
  })

  it('surfaces node preview and page replay failures', async () => {
    const service = createMockEvidenceService('ok')
    await expect(service.getNodePreview(NODE_PREVIEW_FAIL_ID)).rejects.toThrow('Mock node preview error')

    const [documentId, page] = PAGE_REPLAY_FAIL_KEY.split('|') as [string, string]
    await expect(service.getPageReplay(documentId, Number(page))).rejects.toThrow('Mock page replay error')
  })
})
