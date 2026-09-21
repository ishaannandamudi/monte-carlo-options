import type { ResearchOperation } from '../api/research'
import { covariance, historicalDataset, portfolio, riskMappings } from '../api/templates'
import { OperationWorkbench } from '../components/research/OperationWorkbench'
import { Alert, HelpTip, PageHeader } from '../components/ui'

const common = { portfolio: portfolio(), factor_mappings: riskMappings() }
const config = { confidence_level: 0.99, horizon: { trading_days: 1, trading_days_per_year: 252 }, valuation_mode: 'DELTA', include_contributions: true, include_incremental_var: true, include_tail_scenarios: true }
const backtestObservations = Array.from({ length: 20 }, (_, index) => ({
  date: `2025-01-${String(index + 2).padStart(2, '0')}T00:00:00Z`,
  predicted_var: 4,
  realized_pnl: index === 7 ? -5 : (index % 5) - 2,
}))

const operations: readonly ResearchOperation[] = [
  { id: 'historical', label: 'Historical VaR / ES', description: 'Replay dated joint factor shocks without independent resampling.', path: '/api/risk/historical', template: { ...common, dataset: historicalDataset(), config: { ...config, method: 'HISTORICAL', valuation_mode: 'DELTA_GAMMA_DIAGONAL' } }, timeoutMs: 240_000 },
  { id: 'parametric', label: 'Parametric-normal VaR / ES', description: 'Apply mapped local exposures to a supplied covariance and IID horizon assumption.', path: '/api/risk/parametric', template: { ...common, covariance: covariance(), config: { ...config, method: 'PARAMETRIC_NORMAL' } }, timeoutMs: 180_000 },
  { id: 'monte-carlo', label: 'Monte Carlo VaR / ES', description: 'Simulate correlated normal or standardized Student-t factors with reproducible seed control.', path: '/api/risk/monte-carlo', template: { ...common, covariance: covariance(), config: { ...config, method: 'MONTE_CARLO', simulations: 20_000, root_seed: 42, distribution: 'MULTIVARIATE_NORMAL', antithetic: true } }, timeoutMs: 300_000 },
  { id: 'contributions', label: 'Risk contributions', description: 'Calculate component, marginal, incremental, and tail ES contributions where supported.', path: '/api/risk/contributions', template: { ...common, covariance: covariance(), config: { ...config, method: 'PARAMETRIC_NORMAL' } }, timeoutMs: 240_000 },
  { id: 'report', label: 'Risk and stress report', description: 'Keep probabilistic risk and named deterministic stresses visible as separate concepts.', path: '/api/risk/report', template: { ...common, covariance: covariance(), config: { ...config, method: 'PARAMETRIC_NORMAL' }, stress_scenarios: [{ name: 'Equity selloff', spot_relative_shift: -0.2, volatility_absolute_shift: 0.1 }] }, timeoutMs: 300_000 },
  { id: 'backtest', label: 'VaR backtest', description: 'Test strict exceptions and report Kupiec, Christoffersen, and conditional-coverage diagnostics.', path: '/api/risk/backtest', template: { confidence_level: 0.99, observations: backtestObservations } },
]

export default function RiskPage() {
  return <div className="page-stack"><PageHeader eyebrow="Market risk" title="VaR, Expected Shortfall, and backtesting" description="Run historical, parametric, or Monte Carlo loss analysis with explicit confidence, horizon, valuation mode, factor mappings, and contribution policies." /><div className="help-strip"><HelpTip term="VaR">The selected loss quantile at the requested confidence and horizon.</HelpTip><HelpTip term="Expected Shortfall">The mean loss in the tail beyond VaR, using exact empirical tail mass where applicable.</HelpTip></div><Alert title="Loss convention" variant="info">Loss = −P&amp;L. Named stress losses stay outside the VaR/ES distribution, and risk results inherit the selected factor model and valuation approximation.</Alert><OperationWorkbench operations={operations} title="Risk workflow" /></div>
}
