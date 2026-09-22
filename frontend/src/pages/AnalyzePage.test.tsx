import { screen, within } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { beforeEach, describe, expect, it, vi } from 'vitest'

import type { OptionsAnalysisResult } from '../api/optionsAnalyzer'
import { installApiMock, requestUrl } from '../test/mock-api'
import { jsonResponse, renderApp } from '../test/test-utils'

const callContract: OptionsAnalysisResult['contracts'][number] = {
  provider_symbol: 'O:AAPL-C-100', option_type: 'CALL', strike: 100, expiration: '2026-11-06', exercise_style: 'AMERICAN', provider_exercise_style: 'AMERICAN', bid: 5.1, ask: 5.3, market_midpoint: 5.2, market_price_source: 'MIDPOINT', market_reference_price: 5.2, comparison_price: 5.2, comparison_price_source: 'BID_ASK_MIDPOINT', last: 5.15, bid_size: 12, ask_size: 10, volume: 840, open_interest: 5420, quote_timestamp: '2026-09-21T14:30:00Z', quote_age_seconds: 30, days_to_expiration: 46, time_to_expiration_years: 0.126, implied_volatility: 0.24, implied_volatility_status: 'CONVERGED', implied_volatility_method: 'BSM_EQUIVALENT', provider_implied_volatility: 0.245, fair_value_volatility: 0.23, fair_value_volatility_source: 'SMILE_INTERPOLATED', fair_value_volatility_independent: true, fair_value: 5.45, fair_value_gap: 0.25, fair_value_gap_pct: 0.0480769, delta: 0.56, gamma: 0.031, theta: -8.2, theta_daily: -0.02247, vega: 13.5, vega_per_vol_point: 0.135, rho: 6.2, intrinsic_value: 2, time_value: 3.2, moneyness: 0.9804, log_moneyness: -0.02, spread: 0.2, spread_pct: 0.03846, european_reference_value: 5.41, early_exercise_premium: 0.04, tree_steps: 400, crr_pricing_runtime_ms: 0.5, greeks_runtime_ms: 2, model_runtime_ms: 2.5, analysis_quality: 'HIGH_DATA_QUALITY', quality_flags: [], warning_flags: ['BSM_EQUIVALENT_IMPLIED_VOLATILITY'], market_data_mode: 'SNAPSHOT', eod_open: null, eod_high: null, eod_low: null, eod_close: null, eod_vwap: null, data_date: null,
}

const putContract: OptionsAnalysisResult['contracts'][number] = {
  ...callContract,
  provider_symbol: 'O:AAPL-P-100',
  option_type: 'PUT',
  bid: 3.8,
  ask: 4,
  market_midpoint: 3.9,
  market_reference_price: 3.9,
  comparison_price: 3.9,
  fair_value: 3.75,
  fair_value_gap: -0.15,
  fair_value_gap_pct: -0.03846,
  delta: -0.44,
  implied_volatility: 0.25,
  fair_value_volatility: 0.235,
  volume: 500,
  open_interest: 2800,
}

function analysisResult(overrides: Partial<OptionsAnalysisResult> = {}): OptionsAnalysisResult {
  return {
    symbol: 'AAPL', data_source: 'MASSIVE', data_mode: 'SNAPSHOT', price_as_of: '2026-09-21T14:30:30Z', underlying_price_source: 'SNAPSHOT_QUOTE', option_price_source: 'BID_ASK_MIDPOINT', provider_capabilities: { snapshot_available: true, end_of_day_available: true, basic_contract_limit: null }, underlying: { price: 102, bid: 101.99, ask: 102.01, last: 102, timestamp: '2026-09-21T14:30:00Z', data_age_seconds: 30, source: 'MASSIVE', price_source: 'SNAPSHOT_QUOTE', open: null, high: null, low: null, close: null, vwap: null, volume: 1000, data_date: null }, selected_expiration: '2026-11-06', available_expirations: ['2026-11-06', '2026-11-13'], assumptions: { risk_free_rate: 0.04, risk_free_rate_source: 'CONFIGURED_DEFAULT', dividend_yield: 0, dividend_yield_source: 'CONFIGURED_DEFAULT', valuation_time: '2026-09-21T14:30:30Z', pricing_model: 'CRR_AMERICAN', implied_volatility_method: 'BSM_EQUIVALENT', tree_steps: 400, target_expiration_days: 45, selected_expiration_days: 46, strike_window_pct: 0.15, max_contracts: 20, max_quote_age_seconds: 900, allow_last_price_fallback: false, maximum_relative_spread: 0.5 }, summary: { contracts_analyzed: 2, calls: 1, puts: 1, atm_strike: 100, source_contracts_considered: 6, contracts_with_market_iv: 2, contracts_with_independent_fair_vol: 2 }, contracts: [callContract, putContract], smile: [{ provider_symbol: callContract.provider_symbol, option_type: 'CALL', strike: 100, log_moneyness: -0.02, market_implied_volatility: 0.24, fair_value_volatility: 0.23, fair_value_volatility_source: 'SMILE_INTERPOLATED' }, { provider_symbol: putContract.provider_symbol, option_type: 'PUT', strike: 100, log_moneyness: -0.02, market_implied_volatility: 0.25, fair_value_volatility: 0.235, fair_value_volatility_source: 'SMILE_INTERPOLATED' }], warnings: [], timings: { market_data_ms: 12, implied_volatility_ms: 4, smile_construction_ms: 2, crr_pricing_ms: 1, greeks_ms: 7, pricing_and_greeks_ms: 8, total_ms: 26 }, ...overrides,
  }
}

describe('American Options Analyzer', () => {
  beforeEach(() => {
    installApiMock((url) => url.endsWith('/api/options/analyze') ? jsonResponse(analysisResult()) : undefined)
  })

  it('starts with a simple ticker-only landing page and hidden advanced controls', async () => {
    renderApp()
    expect(await screen.findByRole('heading', { name: 'American Options Analyzer' })).toBeInTheDocument()
    expect(screen.getByLabelText('Ticker symbol')).toBeInTheDocument()
    expect(screen.getByRole('button', { name: 'Analyze' })).toBeInTheDocument()
    expect(screen.queryByLabelText('CRR tree steps')).not.toBeVisible()
  })

  it('normalizes ticker input, shows loading, and sends visible assumptions', async () => {
    const user = userEvent.setup()
    let resolveAnalysis: ((response: Response) => void) | undefined
    const pending = new Promise<Response>((resolve) => { resolveAnalysis = resolve })
    const fetchMock = installApiMock((url) => url.endsWith('/api/options/analyze') ? pending : undefined)
    renderApp()
    await user.type(await screen.findByLabelText('Ticker symbol'), ' aapl ')
    await user.click(screen.getByRole('button', { name: 'Analyze' }))
    expect(screen.getByText('Fetching underlying quote…')).toBeInTheDocument()
    const request = fetchMock.mock.calls.find(([input]) => requestUrl(input).endsWith('/api/options/analyze'))
    const body = request?.[1]?.body
    expect(typeof body).toBe('string')
    const payload: unknown = typeof body === 'string' ? JSON.parse(body) : null
    expect(payload).toMatchObject({ symbol: 'AAPL', tree_steps: 400, target_expiration_days: 45 })
    resolveAnalysis?.(jsonResponse(analysisResult()))
    expect(await screen.findByRole('heading', { name: 'AAPL' })).toBeInTheDocument()
    expect(screen.getByText('Snapshot data')).toBeInTheDocument()
  })

  it('renders the result header, sortable chain, gaps, and expiration refresh', async () => {
    const user = userEvent.setup()
    const fetchMock = vi.mocked(fetch)
    renderApp()
    await user.type(await screen.findByLabelText('Ticker symbol'), 'AAPL')
    await user.click(screen.getByRole('button', { name: 'Analyze' }))
    expect(await screen.findByRole('heading', { name: 'AAPL' })).toBeInTheDocument()
    const table = screen.getByRole('table', { name: /American options model comparison/ })
    expect(within(table).getByText('4.81%')).toHaveClass('gap-positive')
    expect(within(table).getByText('-3.85%')).toHaveClass('gap-negative')
    await user.click(within(table).getByRole('button', { name: 'Sort by Gap %' }))
    await user.selectOptions(screen.getByLabelText('Selected expiration'), '2026-11-13')
    await screen.findByRole('heading', { name: 'AAPL' })
    const analysisCalls = fetchMock.mock.calls.filter(([input]) => requestUrl(input).endsWith('/api/options/analyze'))
    const latestBody = analysisCalls.at(-1)?.[1]?.body
    expect(typeof latestBody).toBe('string')
    const latestPayload: unknown = typeof latestBody === 'string' ? JSON.parse(latestBody) : null
    expect(latestPayload).toMatchObject({ expiration: '2026-11-13' })
  })

  it('filters calls and puts and opens keyboard-accessible contract details with Greeks', async () => {
    const user = userEvent.setup()
    renderApp()
    await user.type(await screen.findByLabelText('Ticker symbol'), 'AAPL')
    await user.click(screen.getByRole('button', { name: 'Analyze' }))
    const table = await screen.findByRole('table', { name: /American options model comparison/ })
    await user.click(screen.getByRole('button', { name: 'Calls' }))
    expect(within(table).getByText('CALL')).toBeInTheDocument()
    expect(within(table).queryByText('PUT')).not.toBeInTheDocument()
    await user.click(within(table).getByRole('button', { name: /View details for O:AAPL-C-100/ }))
    expect(screen.getByRole('heading', { name: /CALL.*strike/ })).toBeInTheDocument()
    expect(screen.getByRole('heading', { name: 'Greeks' })).toBeInTheDocument()
    expect(screen.getByText('SMILE INTERPOLATED')).toBeInTheDocument()
  })

  it('draws market and fair-value volatility series and renders stale warnings', async () => {
    installApiMock((url) => url.endsWith('/api/options/analyze') ? jsonResponse(analysisResult({ warnings: ['Delayed/stale market data is present.'] })) : undefined)
    const user = userEvent.setup()
    renderApp()
    await user.type(await screen.findByLabelText('Ticker symbol'), 'AAPL')
    await user.click(screen.getByRole('button', { name: 'Analyze' }))
    expect(await screen.findByText('Delayed/stale market data is present.')).toBeInTheDocument()
    expect(screen.getByRole('img', { name: /Selected-expiration volatility smile/ })).toBeInTheDocument()
    expect(screen.getAllByText('Market IV').length).toBeGreaterThan(0)
    expect(screen.getAllByText('Fair-value vol').length).toBeGreaterThan(0)
  })

  it('adapts badges, chain columns, and contract details for end-of-day data', async () => {
    const eodContract: OptionsAnalysisResult['contracts'][number] = {
      ...callContract,
      bid: null,
      ask: null,
      market_midpoint: null,
      market_price_source: 'END_OF_DAY_CLOSE',
      market_reference_price: 5.05,
      comparison_price: 5.05,
      comparison_price_source: 'END_OF_DAY_CLOSE',
      last: null,
      bid_size: null,
      ask_size: null,
      open_interest: null,
      implied_volatility_method: 'EOD_BSM_EQUIVALENT_IV',
      market_data_mode: 'END_OF_DAY',
      eod_open: 4.9,
      eod_high: 5.2,
      eod_low: 4.8,
      eod_close: 5.05,
      eod_vwap: 5,
      data_date: '2026-09-21',
      quality_flags: ['END_OF_DAY_DATA', 'NO_LIVE_BID_ASK'],
    }
    installApiMock((url) => url.endsWith('/api/options/analyze') ? jsonResponse(analysisResult({
      data_mode: 'END_OF_DAY',
      price_as_of: '2026-09-21T20:00:00Z',
      underlying_price_source: 'END_OF_DAY_CLOSE',
      option_price_source: 'END_OF_DAY_CLOSE',
      provider_capabilities: { snapshot_available: false, end_of_day_available: true, basic_contract_limit: 2 },
      underlying: { price: 102, bid: null, ask: null, last: null, timestamp: '2026-09-21T20:00:00Z', data_age_seconds: 0, source: 'MASSIVE', price_source: 'END_OF_DAY_CLOSE', open: 101, high: 103, low: 100, close: 102, vwap: 101.8, volume: 1000, data_date: '2026-09-21' },
      assumptions: { ...analysisResult().assumptions, valuation_time: '2026-09-21T20:00:00Z', implied_volatility_method: 'EOD_BSM_EQUIVALENT_IV' },
      summary: { ...analysisResult().summary, contracts_analyzed: 1, puts: 0 },
      contracts: [eodContract],
      warnings: ['Using Massive end-of-day data because snapshot access is not included in the current provider plan.'],
    })) : undefined)
    const user = userEvent.setup()
    renderApp()
    await user.type(await screen.findByLabelText('Ticker symbol'), 'AAPL')
    await user.click(screen.getByRole('button', { name: 'Analyze' }))
    expect(await screen.findByText('End-of-day data')).toBeInTheDocument()
    const table = screen.getByRole('table', { name: /American options model comparison/ })
    expect(within(table).getByRole('button', { name: 'Sort by EOD Price' })).toBeInTheDocument()
    expect(within(table).getByText('Data Date')).toBeInTheDocument()
    expect(within(table).queryByRole('button', { name: 'Sort by Bid' })).not.toBeInTheDocument()
    await user.click(within(table).getByRole('button', { name: /View details/ }))
    expect(screen.getByText('EOD close / reference')).toBeInTheDocument()
    expect(screen.getByText('Unavailable on this data mode')).toBeInTheDocument()
    expect(screen.getByText('EOD BSM-equivalent')).toBeInTheDocument()
  })

  it('validates tickers, exposes advanced assumptions, and presents provider errors', async () => {
    const user = userEvent.setup()
    const first = renderApp()
    await user.type(await screen.findByLabelText('Ticker symbol'), '$bad')
    await user.click(screen.getByRole('button', { name: 'Analyze' }))
    expect(screen.getByText(/Enter a valid ticker/)).toBeInTheDocument()
    await user.click(screen.getByText('Advanced assumptions'))
    expect(screen.getByLabelText('CRR tree steps')).toBeVisible()

    first.unmount()
    installApiMock((url) => url.endsWith('/api/options/analyze') ? jsonResponse({ detail: 'Massive rate limit reached' }, 429) : undefined)
    renderApp()
    await user.type(await screen.findByLabelText('Ticker symbol'), 'AAPL')
    await user.click(screen.getByRole('button', { name: 'Analyze' }))
    expect(await screen.findByText('Massive rate limit reached')).toBeInTheDocument()
  })

  it('shows a precise provider entitlement error', async () => {
    installApiMock((url) => url.endsWith('/api/options/analyze') ? jsonResponse({ detail: 'Massive authenticated successfully, but the current plan does not include this endpoint.' }, 403) : undefined)
    const user = userEvent.setup()
    renderApp()
    await user.type(await screen.findByLabelText('Ticker symbol'), 'AAPL')
    await user.click(screen.getByRole('button', { name: 'Analyze' }))
    expect(await screen.findByText(/authenticated successfully.*current plan/)).toBeInTheDocument()
  })

  it('keeps the complete Research terminal reachable', async () => {
    const user = userEvent.setup()
    renderApp()
    await user.click(screen.getByRole('link', { name: /Research$/ }))
    expect(await screen.findByRole('heading', { name: 'Quantitative Derivatives Workbench' })).toBeInTheDocument()
    const navigation = screen.getByRole('navigation', { name: 'Primary navigation' })
    await user.click(within(navigation).getByRole('link', { name: /Pricing/ }))
    expect(await screen.findByRole('heading', { name: 'Pricing workbench' })).toBeInTheDocument()
  })
})
