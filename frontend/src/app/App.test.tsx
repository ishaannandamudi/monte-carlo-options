import { screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import { jsonResponse, renderApp } from '../test/test-utils'

function mockSystemApi(healthStatus = 200): void {
  vi.stubGlobal(
    'fetch',
    vi.fn((input: string | URL | Request) => {
      const url =
        typeof input === 'string' ? input : input instanceof URL ? input.href : input.url
      if (url.endsWith('/health')) {
        return Promise.resolve(
          jsonResponse(
            healthStatus === 200
              ? { status: 'ok', service: 'monte-carlo-options' }
              : { detail: 'Service unavailable' },
            healthStatus,
          ),
        )
      }
      if (url.endsWith('/version')) {
        return Promise.resolve(
          jsonResponse({ service: 'monte-carlo-options', version: '0.22.0' }),
        )
      }
      return Promise.resolve(jsonResponse({ detail: 'Not found' }, 404))
    }),
  )
}

describe('application shell', () => {
  beforeEach(() => {
    mockSystemApi()
  })

  it('renders persistent navigation and the analyzer landing page', async () => {
    renderApp()

    expect(screen.getByRole('navigation', { name: 'Primary navigation' })).toBeInTheDocument()
    expect(
      await screen.findByRole('heading', { name: 'American Options Analyzer' }),
    ).toBeInTheDocument()
    expect(screen.getByText('American Options Research')).toBeInTheDocument()
  })

  it('navigates to a production Phase 22 route', async () => {
    const user = userEvent.setup()
    renderApp()

    const navigation = screen.getByRole('navigation', { name: 'Primary navigation' })
    await user.click(within(navigation).getByRole('link', { name: /Pricing/ }))

    expect(await screen.findByRole('heading', { name: 'Pricing workbench' })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Price option' })).toBeInTheDocument()
  })

  it('shows successful health and version data', async () => {
    renderApp('/research')

    expect(await screen.findByText('Connected')).toBeInTheDocument()
    expect(screen.getByText('0.22.0')).toBeInTheDocument()
    expect(screen.getByText('OK')).toBeInTheDocument()
  })

  it('keeps the application usable when backend health fails', async () => {
    mockSystemApi(503)
    renderApp()

    expect(await screen.findByRole('alert')).toHaveTextContent('Backend disconnected')
    expect(screen.getByRole('heading', { name: 'American Options Analyzer' })).toBeInTheDocument()
    expect(screen.getByRole('navigation', { name: 'Primary navigation' })).toBeInTheDocument()
  })
})
