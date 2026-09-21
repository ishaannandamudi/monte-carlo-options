import type { ResearchOperation } from '../api/research'
import { hestonDataset, hestonParameters, market, mertonParameters, simulation, vanillaContract } from '../api/templates'
import { OperationWorkbench } from '../components/research/OperationWorkbench'
import { Alert, HelpTip, PageHeader } from '../components/ui'

const contract = vanillaContract()
const marketInput = market()
const heston = { contract, market: marketInput, parameters: hestonParameters() }
const merton = { contract, market: marketInput, parameters: mertonParameters() }
const comparison = {
  contract, market: marketInput,
  config: { selected_models: ['BLACK_SCHOLES', 'MONTE_CARLO_GBM', 'CRR_BINOMIAL', 'HESTON_ANALYTICAL', 'MERTON_ANALYTICAL'], reference_model: 'BLACK_SCHOLES', deterministic_seed: 71 },
  gbm_monte_carlo: simulation(5000), tree: { steps: 250 }, heston_parameters: hestonParameters(), merton_parameters: mertonParameters(),
  heston_monte_carlo: { num_simulations: 2000, num_time_steps: 20, seed: 4 },
}
const calibration = {
  dataset: hestonDataset(),
  config: {
    target: 'PRICE', weighting: 'VEGA', expiry_balancing: true,
    selection_policy: 'OTM_ONLY', optimizer: 'HYBRID', initial_parameters: hestonParameters(),
    bounds: { v0: { lower: 0.001, upper: 0.5 }, kappa: { lower: 0.05, upper: 10 }, theta: { lower: 0.001, upper: 0.5 }, sigma_v: { lower: 0.01, upper: 3 }, rho: { lower: -0.999, upper: 0.999 } },
    multi_start_count: 2, random_seed: 19, max_iterations: 40, robust_loss: 'SOFT_L1', feller_penalty_weight: 1, identifiability_diagnostics: true,
  },
}

const operations: readonly ResearchOperation[] = [
  { id: 'compare', label: 'Multi-model comparison', description: 'Compare compatible engines with explicit success, excluded, and failed statuses.', path: '/api/analytics/models/compare', template: comparison, timeoutMs: 180_000 },
  { id: 'market-fit', label: 'Market fit', description: 'Measure price/IV/spread errors by expiry and moneyness for selected models.', path: '/api/analytics/models/market-fit', template: { dataset: hestonDataset(), selected_models: ['BLACK_SCHOLES', 'HESTON_ANALYTICAL', 'MERTON_ANALYTICAL'], heston_parameters: hestonParameters(), merton_parameters: mertonParameters(), use_observed_iv_for_black_scholes: true, preprocessing: { target: 'PRICE', selection_policy: 'ALL' } }, timeoutMs: 180_000 },
  { id: 'model-risk', label: 'Model-risk report', description: 'Combine dispersion, consistency, fit, and stability diagnostics without assigning probabilities to models.', path: '/api/analytics/models/risk', template: { comparison }, timeoutMs: 180_000 },
  { id: 'heston-analytical', label: 'Heston analytical price', description: 'Little Heston Trap Fourier valuation and integration diagnostics.', path: '/api/pricing/heston/analytical', template: heston },
  { id: 'heston-mc', label: 'Heston Monte Carlo', description: 'QE-M simulation with antithetic cluster uncertainty.', path: '/api/pricing/heston/monte-carlo', template: { ...heston, simulation: { num_simulations: 10_000, num_time_steps: 50, seed: 42, scheme: 'QUADRATIC_EXPONENTIAL', antithetic: true } }, timeoutMs: 180_000 },
  { id: 'heston-sensitivity', label: 'Heston sensitivity', description: 'Sweep one risk-neutral parameter and inspect price response.', path: '/api/analytics/heston/sensitivity', template: { ...heston, parameter: 'rho', values: [-0.9, -0.7, -0.5, -0.3, 0] } },
  { id: 'heston-calibration', label: 'Heston calibration', description: 'Bounded calibration with selectable target, weights, robust loss, starts, and Feller penalty.', path: '/api/calibration/heston', template: calibration, timeoutMs: 300_000 },
  { id: 'heston-diagnostics', label: 'Heston fit diagnostics', description: 'Residual, expiry, moneyness, ATM, skew, active-bound, and identifiability diagnostics.', path: '/api/analytics/heston/calibration/diagnostics', template: { ...calibration, parameters: hestonParameters() }, timeoutMs: 180_000 },
  { id: 'merton-analytical', label: 'Merton analytical price', description: 'Conditional-lognormal Poisson-mixture valuation.', path: '/api/pricing/merton/analytical', template: merton },
  { id: 'merton-mc', label: 'Merton Monte Carlo', description: 'Exact aggregate-jump terminal simulation and uncertainty.', path: '/api/pricing/merton/monte-carlo', template: { ...merton, simulation: { num_simulations: 20_000, seed: 42 } }, timeoutMs: 180_000 },
  { id: 'merton-compare', label: 'Merton analytical vs MC', description: 'Compare price, error, uncertainty, and jump diagnostics.', path: '/api/analytics/merton/compare', template: { ...merton, simulation: { num_simulations: 20_000, seed: 42 } }, timeoutMs: 180_000 },
  { id: 'merton-greeks', label: 'Merton Greeks', description: 'Standard diffusion Greeks under the jump-diffusion price.', path: '/api/analytics/merton/greeks', template: merton },
  { id: 'merton-jumps', label: 'Merton jump diagnostics', description: 'Inspect compensator, expected counts, realized count moments, and zero-jump frequency.', path: '/api/analytics/merton/jump-diagnostics', template: { ...merton, simulation: { num_simulations: 20_000, seed: 42 } }, timeoutMs: 180_000 },
]

export default function ModelsPage() {
  return <div className="page-stack"><PageHeader eyebrow="Models" title="Model research" description="Compare established engines, calibrate Heston, and inspect stochastic-volatility and jump-diffusion diagnostics." /><Alert title="Model risk" variant="warning"><HelpTip term="Model dispersion">Cross-model price spread for the explicitly selected engines; it is descriptive and not a probability distribution.</HelpTip> Weak Heston identifiability and Feller violations remain visible in backend diagnostics.</Alert><OperationWorkbench operations={operations} title="Model workflow" /></div>
}
