import { describe, expect, it, vi } from 'vitest'

import { jsonResponse } from '../test/test-utils'
import { ApiError, apiPost } from './client'

describe('API client', () => {
  it('normalizes FastAPI validation details', async () => {
    vi.stubGlobal('fetch', vi.fn(() => Promise.resolve(jsonResponse({ detail: [{ loc: ['body', 'spot'], msg: 'Input should be greater than 0', type: 'greater_than' }] }, 422))))

    const request = apiPost('/api/pricing/black-scholes', {})

    await expect(request).rejects.toBeInstanceOf(ApiError)
    await expect(request).rejects.toMatchObject({ status: 422, issues: [{ location: 'body.spot', message: 'Input should be greater than 0', type: 'greater_than' }] })
  })
})
