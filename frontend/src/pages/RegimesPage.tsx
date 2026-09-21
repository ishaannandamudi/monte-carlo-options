import type { ResearchOperation } from '../api/research'
import { hmmDataset, hmmParameters } from '../api/templates'
import { OperationWorkbench } from '../components/research/OperationWorkbench'
import { Alert, HelpTip, PageHeader } from '../components/ui'

const parameters = hmmParameters()
const dataset = hmmDataset()
const operations: readonly ResearchOperation[] = [
  { id: 'fit', label: 'Fit Gaussian HMM', description: 'Deterministic multi-start Baum-Welch estimation with state summaries and transition diagnostics.', path: '/api/regime/hmm/fit', template: { dataset, config: { num_states: 2, covariance_type: 'DIAGONAL', initialization: 'QUANTILE', num_starts: 3, max_iterations: 100, tolerance: 1e-6, seed: 7, return_probability_series: true, probability_output_mode: 'FULL' } }, timeoutMs: 300_000 },
  { id: 'decode', label: 'Decode regimes', description: 'Viterbi sequence plus filtered and smoothed probability diagnostics for known parameters.', path: '/api/regime/hmm/decode', template: { dataset, parameters, include_probabilities: true, probability_stride: 1 }, timeoutMs: 180_000, note: 'Smoothed probabilities use future observations and are not real-time estimates.' },
  { id: 'forecast', label: 'Regime forecast', description: 'Project current state probabilities through the fitted transition matrix.', path: '/api/regime/hmm/forecast', template: { parameters, current_probabilities: [0.4, 0.6], horizons: [1, 5, 10, 20] } },
  { id: 'model-order', label: 'Model-order study', description: 'Compare log likelihood, AIC, and BIC across candidate state counts without declaring a uniquely correct K.', path: '/api/regime/hmm/model-order', template: { dataset, state_counts: [2, 3, 4], config: { covariance_type: 'DIAGONAL', initialization: 'QUANTILE', num_starts: 2, max_iterations: 60, seed: 4 } }, timeoutMs: 300_000 },
  { id: 'markov', label: 'Markov diagnostics', description: 'Inspect stationary distribution, persistence, entropy, durations, and forecasts.', path: '/api/regime/markov/diagnostics', template: { chain: { transition_matrix: [[0.95, 0.05], [0.1, 0.9]] }, horizons: [1, 5, 20], current_probabilities: [0.5, 0.5] } },
  { id: 'risk', label: 'Regime-conditioned risk', description: 'Calculate per-state and mixture VaR/ES from a Gaussian mixture; mixture risk is computed from the mixture distribution.', path: '/api/regime/risk', template: { parameters, state_probabilities: [0.8, 0.2], factor_ids: ['return'], delta_exposures: [100], confidence_level: 0.99, num_simulations: 50_000, seed: 9 }, timeoutMs: 240_000 },
]

export default function RegimesPage() {
  return <div className="page-stack"><PageHeader eyebrow="Real-world regimes" title="Markov and hidden-state research" description="Fit, decode, forecast, compare, and use Gaussian regimes for descriptive real-world risk analysis." /><div className="help-strip"><HelpTip term="Filtered probability">State probability using observations available through that time only.</HelpTip><HelpTip term="Smoothed probability">Retrospective state probability using the full observation sample, including future data.</HelpTip></div><Alert title="Measure separation" variant="warning">Historical HMM means describe the real-world measure and are never substituted for risk-neutral derivative-pricing drift.</Alert><OperationWorkbench operations={operations} title="Regime workflow" /></div>
}
