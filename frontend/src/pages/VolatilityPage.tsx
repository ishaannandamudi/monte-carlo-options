import type { ResearchOperation } from '../api/research'
import { hestonDataset, hestonParameters, market, mertonParameters, vanillaContract } from '../api/templates'
import type { JsonObject } from '../api/types'
import { OperationWorkbench } from '../components/research/OperationWorkbench'
import { Alert, HelpTip, PageHeader } from '../components/ui'

function syntheticSmile(maturity: number, expiration: string): JsonObject {
  const forward = 100 * Math.exp(0.02 * maturity)
  return {
    underlying_symbol: 'SYNTHETIC', valuation_timestamp: '2026-01-01T00:00:00Z', expiration,
    maturity_years: maturity, spot: 100, forward, risk_free_rate: 0.02, dividend_yield: 0,
    coordinate_convention: 'LOG_FORWARD_MONEYNESS',
    points: [-0.1, 0, 0.1].map((logMoneyness) => ({
      strike: forward * Math.exp(logMoneyness), maturity_years: maturity,
      implied_volatility: 0.2 - 0.08 * logMoneyness, option_type: 'CALL',
      forward_price: forward, moneyness: (forward * Math.exp(logMoneyness)) / 100,
      forward_moneyness: Math.exp(logMoneyness), log_moneyness: logMoneyness,
      source: 'SYNTHETIC', expiration,
    })),
    construction_metadata: { selection_policy: 'ALL_VALID', minimum_points: 1, accepted_points: 3, excluded_points: 0, atm_policy: 'TIGHTEST_RELATIVE_SPREAD_THEN_CALL', duplicate_policy: 'PREFERRED_SIDE_SPREAD_FRESHNESS_SYMBOL', source: 'SYNTHETIC' },
  }
}

const base = { contract: vanillaContract(), market: market() }
const hestonBase = { ...base, parameters: hestonParameters() }
const mertonBase = { ...base, parameters: mertonParameters() }
const calibration = {
  dataset: hestonDataset(),
  config: { target: 'PRICE', weighting: 'VEGA', expiry_balancing: true, selection_policy: 'OTM_ONLY', optimizer: 'LEAST_SQUARES', initial_parameters: hestonParameters(), multi_start_count: 1, max_iterations: 25, robust_loss: 'SOFT_L1', feller_penalty_weight: 1 },
  parameters: hestonParameters(),
}

const operations: readonly ResearchOperation[] = [
  { id: 'implied-vol', label: 'Implied volatility', description: 'Invert one European option price with bounded, safeguarded solvers.', path: '/api/volatility/implied', template: { contract: vanillaContract(), risk_free_rate: 0.05, dividend_yield: 0.01, observed_price: 9.826297782739111, solver_config: { method: 'AUTO' } } },
  { id: 'market-smile', label: 'Price-implied smile', description: 'Invert a strike slice and preserve per-point failures.', path: '/api/analytics/volatility/smile', template: { spot: 100, maturity_years: 1, risk_free_rate: 0.05, dividend_yield: 0.01, option_type: 'CALL', strikes: [80, 90, 100, 110, 120], option_prices: [22.3, 15.1, 9.826297782739111, 5.8, 3.1] } },
  { id: 'surface', label: 'Volatility surface', description: 'Build a total-variance surface from multiple validated smiles.', path: '/api/analytics/volatility/surface/build', template: { smiles: [syntheticSmile(0.5, '2026-07-02T12:00:00Z'), syntheticSmile(1, '2027-01-01T00:00:00Z')] } },
  { id: 'heston-smile', label: 'Heston smile', description: 'Generate model-implied prices and volatilities across strikes.', path: '/api/analytics/heston/smile', template: { ...hestonBase, strikes: [70, 80, 90, 100, 110, 120, 130] } },
  { id: 'heston-fitted', label: 'Heston fitted surface', description: 'Compare calibrated Heston prices/IVs with market observations and residuals.', path: '/api/analytics/heston/calibration/fitted-surface', template: calibration, timeoutMs: 180_000, note: 'Parameters are explicit so this view diagnoses a known fit; run calibration from Models to estimate them.' },
  { id: 'merton-smile', label: 'Merton smile', description: 'Generate the jump-diffusion smile across strikes.', path: '/api/analytics/merton/smile', template: { ...mertonBase, strikes: [70, 80, 90, 100, 110, 120, 130] } },
]

export default function VolatilityPage() {
  return <div className="page-stack"><PageHeader eyebrow="Volatility" title="Smiles and surfaces" description="Invert market prices, build total-variance surfaces, and compare Heston and Merton model-implied volatility." /><Alert title="Interpretation" variant="info"><HelpTip term="Implied volatility">The Black-Scholes volatility reproducing an observed price; it is a representation, not a forecast.</HelpTip> Surface values and residuals use the backend coordinate and interpolation policies.</Alert><OperationWorkbench operations={operations} title="Volatility workflow" /></div>
}
