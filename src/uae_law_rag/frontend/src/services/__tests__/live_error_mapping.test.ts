import { describe, expect, it } from 'vitest'
import { HttpError } from '@/api/http'
import { toSystemNotice } from '@/services/errors'

describe('toSystemNotice (http errors)', () => {
  it('maps HttpError into a http notice with meta', () => {
    const error = new HttpError({
      status: 500,
      url: 'https://example.com/api/chat',
      method: 'POST',
      message: 'HTTP 500',
      traceId: 'trace-1',
      requestId: 'req-1',
      response_text: 'boom',
    })

    const notice = toSystemNotice(error)
    expect(notice.kind).toBe('http')
    expect(notice.level).toBe('error')
    expect(notice.meta?.status).toBe(500)
    expect(notice.meta?.requestId).toBe('req-1')
    expect(notice.meta?.traceId).toBe('trace-1')
    expect(notice.meta?.endpoint).toContain('/api/chat')
  })
})
