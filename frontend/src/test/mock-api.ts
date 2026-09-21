import { vi } from 'vitest'

import { jsonResponse } from './test-utils'

export function requestUrl(input: string | URL | Request): string {
  return typeof input === 'string' ? input : input instanceof URL ? input.href : input.url
}

export function installApiMock(
  handler: (url: string, init?: RequestInit) => Response | Promise<Response> | undefined,
) {
  const mock = vi.fn((input: string | URL | Request, init?: RequestInit) => {
    const url = requestUrl(input)
    if (url.endsWith('/health')) return Promise.resolve(jsonResponse({ status: 'ok', service: 'monte-carlo-options' }))
    if (url.endsWith('/version')) return Promise.resolve(jsonResponse({ service: 'monte-carlo-options', version: '0.22.0' }))
    const response = handler(url, init)
    return Promise.resolve(response ?? jsonResponse({ items: [], page: 1, page_size: 10, total: 0 }))
  })
  vi.stubGlobal('fetch', mock)
  return mock
}
