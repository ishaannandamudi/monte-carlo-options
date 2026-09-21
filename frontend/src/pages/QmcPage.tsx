import type { ResearchOperation } from '../api/research'
import { market, qmcConfig, vanillaContract } from '../api/templates'
import { OperationWorkbench } from '../components/research/OperationWorkbench'
import { Alert, HelpTip, PageHeader } from '../components/ui'

const vanilla = { contract: vanillaContract(), market: market() }
const operations: readonly ResearchOperation[] = [
  { id: 'sobol', label: 'European deterministic Sobol', description: 'Price with a deterministic Sobol integration rule. Classical standard error is intentionally unavailable.', path: '/api/pricing/qmc/european', template: { ...vanilla, qmc: qmcConfig('SOBOL') } },
  { id: 'rqmc', label: 'European randomized QMC', description: 'Use independent Sobol scrambles; uncertainty comes only from replicate estimates.', path: '/api/pricing/qmc/european', template: { ...vanilla, qmc: qmcConfig('SCRAMBLED_SOBOL') } },
  { id: 'asian', label: 'Asian RQMC', description: 'Price a discrete Asian with standard, Brownian-bridge, or PCA path construction.', path: '/api/pricing/qmc/asian', template: { contract: { option_type: 'CALL', spot: 100, strike: 100, maturity_years: 1, averaging_type: 'GEOMETRIC', averaging_schedule: { monitoring_count: 12 } }, market: market(), qmc: { ...qmcConfig('SCRAMBLED_SOBOL', 12), path_construction: 'BROWNIAN_BRIDGE' } }, timeoutMs: 180_000 },
  { id: 'basket', label: 'Basket RQMC', description: 'Use Sobol coordinates through the validated Phase 12 correlation factor.', path: '/api/pricing/qmc/basket', template: { contract: { option_type: 'CALL', strike: 100, maturity_years: 1, weights: [0.5, 0.5], basket_type: 'GEOMETRIC' }, market: { asset_ids: ['A', 'B'], spots: [100, 105], volatilities: [0.2, 0.25], risk_free_rate: 0.03, dividend_yields: [0.01, 0.02], correlation_matrix: [[1, 0.4], [0.4, 1]] }, qmc: qmcConfig('SCRAMBLED_SOBOL', 2) }, timeoutMs: 180_000 },
  { id: 'sampling', label: 'Sampling diagnostics', description: 'Inspect bounded point previews, discrepancy, power-of-two behavior, and seed provenance.', path: '/api/analytics/qmc/sampling', template: { config: qmcConfig('SCRAMBLED_SOBOL', 2), include_points: true, max_output_points: 16 } },
  { id: 'compare', label: 'MC / Sobol / RQMC comparison', description: 'Compare matched sample budgets, reference errors, runtime, and uncertainty availability.', path: '/api/analytics/qmc/compare', template: { ...vanilla, num_samples: 2048, randomized_replicates: 8, root_seed: 5 }, timeoutMs: 180_000 },
  { id: 'convergence', label: 'QMC convergence', description: 'Measure error over powers of two and estimate empirical log-log slopes.', path: '/api/analytics/qmc/convergence', template: { ...vanilla, minimum_power: 7, maximum_power: 12, randomized_replicates: 8, root_seed: 5 }, timeoutMs: 240_000 },
]

export default function QmcPage() {
  return <div className="page-stack"><PageHeader eyebrow="Numerical integration" title="Quasi-Monte Carlo" description="Compare PCG64, deterministic Sobol, and randomized Sobol for European, Asian, and basket valuation." /><div className="help-strip"><HelpTip term="Sobol">A low-discrepancy deterministic integration sequence; deterministic estimates do not have a classical Monte Carlo SE.</HelpTip><HelpTip term="Scrambling">A randomized, unbiased transformation of Sobol points.</HelpTip><HelpTip term="RQMC SE">The sample standard deviation of independent scramble estimates divided by √replicates.</HelpTip></div><Alert title="Uncertainty policy" variant="info">Deterministic Sobol displays standard error and confidence interval as unavailable. Zero would incorrectly claim certainty.</Alert><OperationWorkbench operations={operations} title="QMC workflow" /></div>
}
