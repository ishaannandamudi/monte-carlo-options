import { screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it } from 'vitest'

import { installApiMock } from '../test/mock-api'
import { jsonResponse, renderApp } from '../test/test-utils'

describe('production research workflows', () => {
  it('prices a vanilla option and renders price and Greeks', async () => {
    installApiMock((url) => {
      if (url.endsWith('/api/pricing/black-scholes')) return jsonResponse({ price: 10.450583, method: 'BLACK_SCHOLES' })
      if (url.endsWith('/api/greeks/black-scholes')) return jsonResponse({ delta: 0.6368, gamma: 0.0188, vega: 37.52, theta: -6.41, rho: 53.23 })
      return undefined
    })
    const user = userEvent.setup()
    renderApp('/pricing')

    await user.click(await screen.findByRole('button', { name: 'Price option' }))

    expect(await screen.findByText('$10.45')).toBeInTheDocument()
    expect(screen.getByText('DELTA')).toBeInTheDocument()
    expect(screen.getByText('0.6368')).toBeInTheDocument()
  })

  it('keeps model comparison failures visible beside successful prices', async () => {
    installApiMock((url) => url.endsWith('/api/analytics/models/compare') ? jsonResponse({ successful_models: ['BLACK_SCHOLES'], failed_models: ['CRR_BINOMIAL'], excluded_models: ['HESTON_MONTE_CARLO'], outputs: [{ model: 'BLACK_SCHOLES', status: 'SUCCESS', price: 10.45, runtime_ms: 0.2 }, { model: 'CRR_BINOMIAL', status: 'FAILED', price: null, runtime_ms: null }] }) : undefined)
    const user = userEvent.setup()
    renderApp('/models')

    await user.click(await screen.findByRole('button', { name: 'Run workflow' }))

    expect(await screen.findByText('CRR_BINOMIAL failed')).toBeInTheDocument()
    expect(screen.getByText('HESTON_MONTE_CARLO excluded')).toBeInTheDocument()
    expect(screen.getByText('SUCCESS')).toBeInTheDocument()
    expect(screen.getByText('FAILED')).toBeInTheDocument()
  })

  it('adds, removes, and values portfolio positions', async () => {
    installApiMock((url) => url.endsWith('/api/portfolio/value') ? jsonResponse({ portfolio_id: 'frontend-portfolio', total_value: 21.5, gross_long_value: 21.5, gross_short_value: 0, position_count: 1, successful_position_count: 1, failed_position_count: 0, completeness: 'COMPLETE', positions: [{ position_id: 'call-1', status: 'SUCCESS', unit_price: 10.75, position_value: 21.5 }], runtime_ms: 1.2 }) : undefined)
    const user = userEvent.setup()
    renderApp('/portfolio')

    await user.click(await screen.findByRole('button', { name: 'Add position' }))
    expect(screen.getByText('2 positions · positive is long, negative is short')).toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: 'Remove position position-2' }))
    expect(screen.getByText('1 position · positive is long, negative is short')).toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: 'Run Portfolio value' }))
    expect((await screen.findAllByText('$21.50')).length).toBeGreaterThanOrEqual(1)
  })

  it('renders a portfolio scenario result and approximation residual', async () => {
    installApiMock((url) => url.endsWith('/api/portfolio/scenario') ? jsonResponse({ base_value: 10, scenario_value: 7, full_repricing_pnl: -3, greek_approximation_pnl: -2.8, approximation_residual: -0.2, warnings: ['Large nonlinear move'] }) : undefined)
    const user = userEvent.setup()
    renderApp('/portfolio')
    await user.selectOptions(await screen.findByLabelText('Analysis'), 'scenario')
    await user.click(screen.getByRole('button', { name: 'Run Scenario analysis' }))

    expect(await screen.findByText('Large nonlinear move')).toBeInTheDocument()
    expect(screen.getByText('-$3.00')).toBeInTheDocument()
  })

  it('renders VaR and Expected Shortfall with warnings', async () => {
    installApiMock((url) => url.endsWith('/api/risk/historical') ? jsonResponse({ portfolio_id: 'risk', base_value: 20, value_at_risk: 3.2, expected_shortfall: 4.7, confidence_level: 0.99, method: 'HISTORICAL', valuation_mode: 'DELTA', scenario_count: 6, distribution_summary: {}, warnings: ['Short history'], runtime_ms: 2.5 }) : undefined)
    const user = userEvent.setup()
    renderApp('/risk')
    await user.click(await screen.findByRole('button', { name: 'Run workflow' }))

    expect(await screen.findByText('$3.20')).toBeInTheDocument()
    expect(screen.getByText('$4.70')).toBeInTheDocument()
    expect(screen.getByText('Short history')).toBeInTheDocument()
  })

  it('renders HMM fit diagnostics', async () => {
    installApiMock((url) => url.endsWith('/api/regime/hmm/fit') ? jsonResponse({ parameters: {}, converged: true, iterations: 14, log_likelihood: 31.25, aic: -42, bic: -35, state_summaries: [], transition_diagnostics: {}, probability_series: [], runtime_ms: 8.4 }) : undefined)
    const user = userEvent.setup()
    renderApp('/regimes')
    await user.click(await screen.findByRole('button', { name: 'Run workflow' }))

    expect(await screen.findByText('31.25')).toBeInTheDocument()
    expect(screen.getByText('8.40 ms')).toBeInTheDocument()
  })

  it('shows deterministic Sobol uncertainty as unavailable', async () => {
    installApiMock((url) => url.endsWith('/api/pricing/qmc/european') ? jsonResponse({ product: 'EUROPEAN', price: 10.45, standard_error: null, confidence_interval_low: null, confidence_interval_high: null, sampling_method: 'SOBOL', requested_samples: 1024, effective_samples: 1024, runtime_ms: 1.1 }) : undefined)
    const user = userEvent.setup()
    renderApp('/qmc')
    await user.click(await screen.findByRole('button', { name: 'Run workflow' }))

    expect(await screen.findByText('Unavailable')).toBeInTheDocument()
  })

  it('shows randomized QMC replicate standard error', async () => {
    installApiMock((url) => url.endsWith('/api/pricing/qmc/european') ? jsonResponse({ product: 'EUROPEAN', price: 10.45, standard_error: 0.0125, confidence_interval_low: 10.42, confidence_interval_high: 10.48, sampling_method: 'SCRAMBLED_SOBOL', requested_samples: 1024, effective_samples: 1024, runtime_ms: 2.1 }) : undefined)
    const user = userEvent.setup()
    renderApp('/qmc')
    await user.selectOptions(await screen.findByLabelText('Workflow'), 'rqmc')
    await user.click(screen.getByRole('button', { name: 'Run workflow' }))

    expect(await screen.findByText('0.0125')).toBeInTheDocument()
  })
})
