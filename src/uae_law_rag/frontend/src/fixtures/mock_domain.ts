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
  steps: stepsBase,
}

const runNoDebug: RunRecord = {
  runId: 'run_no_debug_001',
  conversationId: 'conv_no_debug_001',
  kbId: 'kb_uae_law',
  queryText: 'Explain probation period limits in UAE law.',
  status: 'degraded',
  timing: { totalMs: 1420, stages: { retrieval: 240, generation: 760, evaluator: 420 } },
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
  ],
  page: 1,
  pageSize: 10,
  total: 2,
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

export const NODE_PREVIEW_FAIL_ID = 'nodeB'
export const PAGE_REPLAY_FAIL_KEY = 'doc002|4'
