import type { ChatNormalizedResult, ChatDebugState } from '@/types/domain/chat'
import type { ChatMessage } from '@/types/domain/message'
import type {
  EvidenceCitation,
  EvidenceTreeNode,
  NodePreview,
  PageReplay,
  RetrievalHitsPaged,
} from '@/types/domain/evidence'
import type { RunRecord } from '@/types/domain/run'
import type { StepRecord } from '@/types/domain/step'

const stepsBase: StepRecord[] = [
  { step: 'retrieval', status: 'success', reasons: [] },
  { step: 'generation', status: 'success', reasons: [] },
  { step: 'evaluator', status: 'success', reasons: [] },
]

const runOk: RunRecord = {
  runId: 'run_ok_001',
  conversationId: 'conv_ok_001',
  kbId: 'kb_uae_law',
  queryText: 'What are the penalties for late salary payments?',
  status: 'success',
  timing: { totalMs: 1840, stages: { retrieval: 320, generation: 900, evaluator: 620 } },
  records: {
    retrievalRecordId: 'mock_retrieval_ok',
    documentId: 'doc001',
  },
  steps: stepsBase,
}

const runNoDebug: RunRecord = {
  runId: 'run_no_debug_001',
  conversationId: 'conv_no_debug_001',
  kbId: 'kb_uae_law',
  queryText: 'Explain probation period limits in UAE law.',
  status: 'degraded',
  timing: { totalMs: 1420, stages: { retrieval: 240, generation: 760, evaluator: 420 } },
  records: {
    retrievalRecordId: 'mock_retrieval_no_debug',
    documentId: 'doc002',
  },
  steps: [
    { step: 'retrieval', status: 'success', reasons: [] },
    { step: 'generation', status: 'degraded', reasons: ['noEvidence'] },
    { step: 'evaluator', status: 'success', reasons: [] },
  ],
}

const citationsOk: EvidenceCitation[] = [
  {
    nodeId: 'nodeA',
    locator: {
      documentId: 'doc001',
      page: 12,
      start: 120,
      end: 248,
      source: 'keyword',
    },
  },
  {
    nodeId: 'nodeB',
    locator: {
      documentId: 'doc002',
      page: 4,
      start: 44,
      end: 132,
      source: 'vector',
    },
  },
  {
    nodeId: '',
    locator: {},
  },
]

const evidenceTreeOk: EvidenceTreeNode[] = [
  {
    id: 'source:keyword',
    label: 'keyword',
    children: [
      {
        id: 'doc:doc001',
        label: 'doc001',
        children: [
          {
            id: 'page:12',
            label: 'page 12',
            children: [
              {
                id: 'nodeA',
                label: 'nodeA',
                locator: {
                  documentId: 'doc001',
                  page: 12,
                  start: 120,
                  end: 248,
                  source: 'keyword',
                },
              },
            ],
          },
        ],
      },
    ],
  },
  {
    id: 'source:vector',
    label: 'vector',
    children: [
      {
        id: 'doc:doc002',
        label: 'doc002',
        children: [
          {
            id: 'page:4',
            label: 'page 4',
            children: [
              {
                id: 'nodeB',
                label: 'nodeB',
                locator: {
                  documentId: 'doc002',
                  page: 4,
                  start: 44,
                  end: 132,
                  source: 'vector',
                },
              },
            ],
          },
        ],
      },
    ],
  },
]

const retrievalHitsOk: RetrievalHitsPaged = {
  items: [
    {
      nodeId: 'nodeA',
      source: 'keyword',
      rank: 1,
      score: 0.93,
      locator: {
        documentId: 'doc001',
        page: 12,
        start: 120,
        end: 248,
        source: 'keyword',
      },
    },
    {
      nodeId: 'nodeB',
      source: 'vector',
      rank: 2,
      score: 0.88,
      locator: {
        documentId: 'doc002',
        page: 4,
        start: 44,
        end: 132,
        source: 'vector',
      },
    },
    {
      nodeId: 'nodeC',
      source: 'keyword',
      rank: 3,
      score: 0.86,
      locator: {
        documentId: 'doc003',
        page: 7,
        start: 30,
        end: 98,
        source: 'keyword',
      },
    },
    {
      nodeId: 'nodeD',
      source: 'vector',
      rank: 4,
      score: 0.84,
      locator: {
        documentId: 'doc004',
        page: 10,
        start: 52,
        end: 140,
        source: 'vector',
      },
    },
    {
      nodeId: 'nodeE',
      source: 'keyword',
      rank: 5,
      score: 0.82,
      locator: {
        documentId: 'doc005',
        page: 3,
        start: 18,
        end: 76,
        source: 'keyword',
      },
    },
    {
      nodeId: 'nodeF',
      source: 'vector',
      rank: 6,
      score: 0.81,
      locator: {
        documentId: 'doc006',
        page: 15,
        start: 90,
        end: 166,
        source: 'vector',
      },
    },
    {
      nodeId: 'nodeG',
      source: 'keyword',
      rank: 7,
      score: 0.79,
      locator: {
        documentId: 'doc007',
        page: 6,
        start: 40,
        end: 122,
        source: 'keyword',
      },
    },
    {
      nodeId: 'nodeH',
      source: 'vector',
      rank: 8,
      score: 0.78,
      locator: {
        documentId: 'doc008',
        page: 9,
        start: 24,
        end: 112,
        source: 'vector',
      },
    },
    {
      nodeId: 'nodeI',
      source: 'keyword',
      rank: 9,
      score: 0.76,
      locator: {
        documentId: 'doc009',
        page: 11,
        start: 48,
        end: 138,
        source: 'keyword',
      },
    },
    {
      nodeId: 'nodeJ',
      source: 'vector',
      rank: 10,
      score: 0.75,
      locator: {
        documentId: 'doc010',
        page: 5,
        start: 34,
        end: 118,
        source: 'vector',
      },
    },
    {
      nodeId: 'nodeK',
      source: 'keyword',
      rank: 11,
      score: 0.73,
      locator: {
        documentId: 'doc011',
        page: 2,
        start: 12,
        end: 70,
        source: 'keyword',
      },
    },
    {
      nodeId: 'nodeL',
      source: 'vector',
      rank: 12,
      score: 0.71,
      locator: {
        documentId: 'doc012',
        page: 16,
        start: 92,
        end: 176,
        source: 'vector',
      },
    },
    {
      nodeId: 'nodeM',
      source: 'keyword',
      rank: 13,
      score: 0.69,
      locator: {
        documentId: 'doc013',
        page: 13,
        start: 60,
        end: 146,
        source: 'keyword',
      },
    },
    {
      nodeId: 'nodeN',
      source: 'vector',
      rank: 14,
      score: 0.67,
      locator: {
        documentId: 'doc014',
        page: 8,
        start: 26,
        end: 116,
        source: 'vector',
      },
    },
    {
      nodeId: 'nodeO',
      source: 'keyword',
      rank: 15,
      score: 0.65,
      locator: {
        documentId: 'doc015',
        page: 14,
        start: 72,
        end: 158,
        source: 'keyword',
      },
    },
    {
      nodeId: 'nodeP',
      source: 'vector',
      rank: 16,
      score: 0.63,
      locator: {
        documentId: 'doc016',
        page: 1,
        start: 8,
        end: 64,
        source: 'vector',
      },
    },
    {
      nodeId: 'nodeQ',
      source: 'keyword',
      rank: 17,
      score: 0.61,
      locator: {
        documentId: 'doc017',
        page: 17,
        start: 100,
        end: 190,
        source: 'keyword',
      },
    },
    {
      nodeId: 'nodeR',
      source: 'vector',
      rank: 18,
      score: 0.59,
      locator: {
        documentId: 'doc018',
        page: 18,
        start: 110,
        end: 198,
        source: 'vector',
      },
    },
    {
      nodeId: 'nodeS',
      source: 'keyword',
      rank: 19,
      score: 0.57,
      locator: {
        documentId: 'doc019',
        page: 19,
        start: 116,
        end: 206,
        source: 'keyword',
      },
    },
    {
      nodeId: 'nodeT',
      source: 'vector',
      rank: 20,
      score: 0.55,
      locator: {
        documentId: 'doc020',
        page: 20,
        start: 120,
        end: 216,
        source: 'vector',
      },
    },
  ],
  page: 1,
  pageSize: 10,
  total: 20,
  source: undefined,
}

const debugOk: ChatDebugState = {
  available: true,
  promptDebug: {
    mode: 'windowPreferred',
    nodesUsed: 2,
    totalChars: 420,
    items: [
      { nodeId: 'nodeA', source: 'keyword', used: 'window', chars: 220 },
      { nodeId: 'nodeB', source: 'vector', used: 'snippet', chars: 200 },
    ],
  },
  keywordStats: {
    rawQuery: 'late salary payments',
    items: [{ keyword: 'salary', recall: 0.61 }],
  },
}

const debugUnavailable: ChatDebugState = {
  available: false,
  message: 'Debug evidence unavailable.',
}

export const RUN_OK: ChatNormalizedResult = {
  run: runOk,
  answer:
    'Late salary payments can trigger penalties under UAE labor law, including fines and enforcement actions.',
  debug: debugOk,
  evidence: {
    citations: citationsOk,
    debugEvidenceTree: evidenceTreeOk,
    retrievalHitsPaged: retrievalHitsOk,
  },
}

export const RUN_NO_DEBUG: ChatNormalizedResult = {
  run: runNoDebug,
  answer:
    'Probation period limits are typically capped, and termination rights are more flexible within that window.',
  debug: debugUnavailable,
  evidence: {
    citations: citationsOk,
    retrievalHitsPaged: retrievalHitsOk,
  },
}

export const RUN_EMPTY: ChatNormalizedResult | undefined = undefined

export const MESSAGES_OK: ChatMessage[] = [
  {
    id: 'msg_ok_user_001',
    role: 'user',
    content: 'What are the penalties for late salary payments?',
  },
  {
    id: 'msg_ok_assistant_001',
    role: 'assistant',
    content:
      'Late salary payments can trigger penalties under UAE labor law, including fines and enforcement actions.',
  },
]

export const MESSAGES_NO_DEBUG: ChatMessage[] = [
  {
    id: 'msg_no_debug_user_001',
    role: 'user',
    content: 'Explain probation period limits in UAE law.',
  },
  {
    id: 'msg_no_debug_assistant_001',
    role: 'assistant',
    content:
      'Probation period limits are typically capped, and termination rights are more flexible within that window.',
  },
]

export const NODE_PREVIEW_OK: NodePreview = {
  nodeId: 'nodeA',
  documentId: 'doc001',
  page: 12,
  startOffset: 120,
  endOffset: 248,
  pageStartOffset: 90,
  pageEndOffset: 300,
  meta: {
    window: 'Within 14 days of delay, penalties may apply depending on employer size.',
    originalText:
      'Article 43 states that delays may incur penalties, and repeat delays can trigger escalated enforcement.',
  },
  textExcerpt:
    'Article 43 states that delays may incur penalties, and repeat delays can trigger escalated enforcement.',
}

export const PAGE_REPLAY_OK: PageReplay = {
  documentId: 'doc001',
  page: 12,
  kbId: 'kb_uae_law',
  content:
    'Article 43: Employers must pay salaries on time. Delays beyond 14 days may result in penalties. Continued non-compliance can lead to escalating enforcement measures and potential suspension of permits.',
}

export const NODE_PREVIEW_FAIL_ID = 'nodeZ'
export const PAGE_REPLAY_FAIL_KEY = 'doc999|1'
