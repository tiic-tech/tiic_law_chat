//docstring
// 职责: Evidence 相关 records API 调用与 DTO -> Domain 映射。
// 边界: 不返回 DTO；仅输出 domain 结构。
// 上游关系: stores 或 services orchestrator。
// 下游关系: api/client.ts。
import { apiClient } from '@/api/client'
import type {
  EvidenceLocator,
  NodePreview,
  PageReplay,
  RetrievalHit,
  RetrievalHitsPaged,
} from '@/types/domain/evidence'
import type { HitRow, RetrievalHitsView } from '@/types/ui'
import type { RetrievalRecordViewDTO, HitSummaryDTO } from '@/types/http/records_retrieval_response'
import type { NodeRecordViewDTO } from '@/types/http/records_node_response'
import type { PageRecordViewDTO } from '@/types/http/records_page_response'

export type RetrievalHitsQuery = {
  source?: string[]
  limit?: number
  offset?: number
  group?: boolean
}

export type NodePreviewQuery = {
  kbId?: string
  maxChars?: number
}

export type PageReplayQuery = {
  documentId: string
  page: number
  kbId?: string
  maxChars?: number
}

export type PageReplayByNodeQuery = {
  kbId?: string
  maxChars?: number
}

const isRecord = (value: unknown): value is Record<string, unknown> => {
  return typeof value === 'object' && value !== null
}

const readNumber = (record: Record<string, unknown>, key: string): number | undefined => {
  const value = record[key]
  return typeof value === 'number' ? value : undefined
}

const readString = (record: Record<string, unknown>, key: string): string | undefined => {
  const value = record[key]
  return typeof value === 'string' ? value : undefined
}

const buildLocatorFromRecord = (input: unknown): EvidenceLocator => {
  const record = isRecord(input) ? input : {}
  const locator: EvidenceLocator = {}

  const documentId = readString(record, 'document_id')
  if (documentId) locator.documentId = documentId

  const page = readNumber(record, 'page')
  if (page !== undefined) locator.page = page

  const start = readNumber(record, 'start_offset') ?? readNumber(record, 'start')
  if (start !== undefined) locator.start = start

  const end = readNumber(record, 'end_offset') ?? readNumber(record, 'end')
  if (end !== undefined) locator.end = end

  const source = readString(record, 'source')
  if (source) locator.source = source

  const articleId = readString(record, 'article_id')
  if (articleId) locator.articleId = articleId

  const sectionPath = readString(record, 'section_path')
  if (sectionPath) locator.sectionPath = sectionPath

  return locator
}

const mapRetrievalHit = (hit: HitSummaryDTO): RetrievalHit => {
  return {
    nodeId: hit.node_id,
    source: hit.source,
    rank: hit.rank,
    score: hit.score,
    locator: buildLocatorFromRecord(hit.locator),
  }
}

const mapRetrievalRecord = (
  record: RetrievalRecordViewDTO,
  sources: string[] | undefined,
): RetrievalHitsPaged => {
  const pageSize = record.hits_limit ?? record.hits.length
  const offset = record.hits_offset ?? 0
  const page = pageSize ? Math.floor(offset / pageSize) + 1 : 1
  const total = record.hits_total ?? record.hits.length

  return {
    items: record.hits.map(mapRetrievalHit),
    page,
    pageSize: pageSize || record.hits.length || 0,
    total,
    source: sources && sources.length === 1 ? sources[0] : undefined,
  }
}

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

const readAvailableSources = (record: RetrievalRecordViewDTO, hits: RetrievalHit[]): string[] => {
  const counts = record.hit_counts
  if (counts && Object.keys(counts).length) {
    return Object.keys(counts)
  }
  return Array.from(new Set(hits.map((hit) => hit.source).filter((value): value is string => Boolean(value))))
}

const mapNodePreview = (record: NodeRecordViewDTO): NodePreview => {
  return {
    nodeId: record.node_id,
    documentId: record.document_id,
    page: record.page,
    startOffset: record.start_offset,
    endOffset: record.end_offset,
    pageStartOffset: record.page_start_offset,
    pageEndOffset: record.page_end_offset,
    meta: {
      window: record.meta?.window,
      originalText: record.meta?.original_text,
    },
    textExcerpt: record.text_excerpt,
  }
}

const mapPageReplay = (record: PageRecordViewDTO): PageReplay => {
  return {
    documentId: record.document_id,
    page: record.page,
    kbId: record.kb_id,
    content: record.content,
  }
}

export const loadRetrievalHits = async (
  retrievalRecordId: string,
  query: RetrievalHitsQuery = {},
): Promise<RetrievalHitsPaged> => {
  const record = await apiClient.getRetrievalRecord(retrievalRecordId, query)
  return mapRetrievalRecord(record, query.source)
}

export const loadRetrievalHitsView = async (
  retrievalRecordId: string,
  query: RetrievalHitsQuery = {},
): Promise<RetrievalHitsView> => {
  const record = await apiClient.getRetrievalRecord(retrievalRecordId, query)
  const paged = mapRetrievalRecord(record, query.source)
  const availableSources = readAvailableSources(record, paged.items)
  return {
    items: paged.items.map(mapHitRow),
    page: paged.page,
    pageSize: paged.pageSize,
    total: paged.total,
    source: paged.source,
    availableSources,
  }
}

export const loadNodePreview = async (
  nodeId: string,
  query: NodePreviewQuery = {},
): Promise<NodePreview> => {
  const record = await apiClient.getNode(nodeId, query)
  return mapNodePreview(record)
}

export const loadPageReplay = async (query: PageReplayQuery): Promise<PageReplay> => {
  const record = await apiClient.getPageReplay(query)
  return mapPageReplay(record)
}

export const loadPageReplayByNode = async (
  nodeId: string,
  query: PageReplayByNodeQuery = {},
): Promise<PageReplay> => {
  const record = await apiClient.getPageReplayByNode(nodeId, query)
  return mapPageReplay(record)
}

export const createLiveEvidenceService = () => {
  return {
    getNodePreview: (nodeId: string) => loadNodePreview(nodeId),
    getPageReplay: (documentId: string, page: number) => loadPageReplay({ documentId, page }),
    getRetrievalHits: (retrievalRecordId: string, params: { source?: string; limit: number; offset: number }) =>
      loadRetrievalHitsView(retrievalRecordId, {
        source: params.source ? [params.source] : undefined,
        limit: params.limit,
        offset: params.offset,
      }),
  }
}
