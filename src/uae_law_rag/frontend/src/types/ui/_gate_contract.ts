// src/types/ui/gate contract
//docstring
// 职责: F1-3 UI Gate 结构自检（仅用于 typecheck，禁止运行时依赖）。
// 边界: 只依赖 ui/domain；不导入 http DTO。
// 上游关系: pnpm typecheck 执行。
// 下游关系: 无（仅用于类型约束）。
import type { EvidenceTreeNode, NodePreview } from '@/types/domain/evidence'
import type { StepRecord } from '@/types/domain/step'
import type { ChatPageProps, ChatView, EvidenceView } from '@/types/ui'

const stepsSample: StepRecord[] = [
  { step: 'retrieval', status: 'success', reasons: [] },
  { step: 'generation', status: 'degraded', reasons: ['noEvidence'] },
  { step: 'evaluator', status: 'success', reasons: [] },
]

const evidenceTreeSample: EvidenceTreeNode[] = [
  {
    id: 'source:keyword',
    label: 'keyword',
    children: [
      {
        id: 'doc:doc001',
        label: 'doc001',
        children: [
          {
            id: 'page:1',
            label: 'page 1',
          },
        ],
      },
    ],
  },
]

const nodePreviewSample: NodePreview = {
  nodeId: 'node001',
  documentId: 'doc001',
  page: 1,
  pageStartOffset: 10,
  pageEndOffset: 120,
  meta: { window: 'window', originalText: 'original' },
  textExcerpt: 'excerpt',
}

const chatSample: ChatView = {
  history: {
    items: [
      { id: 'msg-user-1', role: 'user', content: 'What is the penalty?' },
      {
        id: 'msg-assistant-1',
        role: 'assistant',
        content: 'Penalties may apply depending on enforcement.',
        citations: [
          {
            nodeId: 'node001',
            locator: { documentId: 'doc001', page: 1, start: 10, end: 120 },
          },
        ],
      },
    ],
  },
  activeRun: {
    runId: 'run001',
    status: 'degraded',
    answer: undefined,
    evaluatorBadge: { level: 'partial', label: 'degraded' },
    steps: stepsSample,
  },
  citations: [
    {
      nodeId: 'node001',
      locator: { documentId: 'doc001', page: 1, start: 10, end: 120 },
      onClickRef: { nodeId: 'node001', documentId: 'doc001', page: 1, start: 10, end: 120 },
    },
  ],
  debug: {
    enabled: true,
    promptDebug: {
      mode: 'windowPreferred',
      nodesUsed: 1,
      totalChars: 120,
      items: [{ nodeId: 'node001', used: 'window', chars: 120 }],
    },
    keywordStats: {
      rawQuery: 'example query',
      items: [{ keyword: 'law', recall: 0.5 }],
    },
    evidenceSummary: {
      totalHits: 1,
      sources: [{ name: 'keyword', count: 1 }],
    },
  },
}

const evidenceSample: EvidenceView = {
  retrievalHits: {
    items: [
      {
        nodeId: 'node001',
        source: 'keyword',
        rank: 1,
        score: 0.95,
        page: 1,
      },
    ],
    page: 1,
    pageSize: 10,
    total: 1,
    source: 'keyword',
    availableSources: ['keyword'],
  },
  nodePreview: nodePreviewSample,
  evidenceTree: evidenceTreeSample,
}

export const gateChatPageProps: ChatPageProps = {
  chat: chatSample,
  evidence: evidenceSample,
  onSend: () => {},
  onSelectCitation: () => {},
}
