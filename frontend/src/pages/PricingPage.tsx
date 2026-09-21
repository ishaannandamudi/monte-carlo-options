import { useRef, useState } from 'react'

import { executeResearchOperation, type ResearchOperation } from '../api/research'
import { hestonParameters, market, mertonParameters, simulation, vanillaContract } from '../api/templates'
import type { JsonObject } from '../api/types'
import { ResultPanel } from '../components/research/ResultPanel'
import { Alert, Button, Card, ErrorPanel, HelpTip, Input, LoadingState, PageHeader, Select } from '../components/ui'

type PricingEngine =
  | 'BLACK_SCHOLES'
  | 'GBM_MONTE_CARLO'
  | 'CRR'
  | 'HESTON_ANALYTICAL'
  | 'HESTON_MONTE_CARLO'
  | 'MERTON_ANALYTICAL'
  | 'MERTON_MONTE_CARLO'

interface PricingForm {
  optionType: 'CALL' | 'PUT'
  spot: number
  strike: number
  maturity: number
  rate: number
  dividend: number
  volatility: number
  simulations: number
  seed: number
  steps: number
  v0: number
  kappa: number
  theta: number
  sigmaV: number
  rho: number
  jumpIntensity: number
  jumpMean: number
  jumpVolatility: number
}

const initialForm: PricingForm = {
  optionType: 'CALL', spot: 100, strike: 100, maturity: 1, rate: 0.03, dividend: 0.01,
  volatility: 0.2, simulations: 20_000, seed: 42, steps: 500,
  v0: 0.04, kappa: 2, theta: 0.04, sigmaV: 0.3, rho: -0.7,
  jumpIntensity: 0.8, jumpMean: -0.12, jumpVolatility: 0.25,
}

const engineOptions = [
  { value: 'BLACK_SCHOLES', label: 'Black-Scholes-Merton' },
  { value: 'GBM_MONTE_CARLO', label: 'GBM Monte Carlo' },
  { value: 'CRR', label: 'CRR binomial tree' },
  { value: 'HESTON_ANALYTICAL', label: 'Heston analytical' },
  { value: 'HESTON_MONTE_CARLO', label: 'Heston Monte Carlo' },
  { value: 'MERTON_ANALYTICAL', label: 'Merton analytical' },
  { value: 'MERTON_MONTE_CARLO', label: 'Merton Monte Carlo' },
] as const

function operation(engine: PricingEngine): ResearchOperation {
  const mapping: Record<PricingEngine, { label: string; path: string }> = {
    BLACK_SCHOLES: { label: 'Black-Scholes-Merton', path: '/api/pricing/black-scholes' },
    GBM_MONTE_CARLO: { label: 'GBM Monte Carlo', path: '/api/pricing/monte-carlo' },
    CRR: { label: 'CRR binomial tree', path: '/api/pricing/binomial' },
    HESTON_ANALYTICAL: { label: 'Heston analytical', path: '/api/pricing/heston/analytical' },
    HESTON_MONTE_CARLO: { label: 'Heston Monte Carlo', path: '/api/pricing/heston/monte-carlo' },
    MERTON_ANALYTICAL: { label: 'Merton analytical', path: '/api/pricing/merton/analytical' },
    MERTON_MONTE_CARLO: { label: 'Merton Monte Carlo', path: '/api/pricing/merton/monte-carlo' },
  }
  return { id: engine, ...mapping[engine], description: mapping[engine].label, template: {}, timeoutMs: engine.includes('MONTE_CARLO') ? 180_000 : 60_000 }
}

function requestPayload(engine: PricingEngine, form: PricingForm): JsonObject {
  const contract = { ...vanillaContract(form.optionType), spot: form.spot, strike: form.strike, maturity_years: form.maturity }
  const marketInput = { ...market(), risk_free_rate: form.rate, dividend_yield: form.dividend, volatility: form.volatility }
  const base: JsonObject = { contract, market: marketInput }
  if (engine === 'GBM_MONTE_CARLO') return { ...base, simulation: { ...simulation(form.simulations), seed: form.seed } }
  if (engine === 'CRR') return { ...base, tree: { steps: form.steps, model: 'CRR' } }
  if (engine.startsWith('HESTON')) {
    const parameters = { ...hestonParameters(), v0: form.v0, kappa: form.kappa, theta: form.theta, sigma_v: form.sigmaV, rho: form.rho }
    return engine === 'HESTON_MONTE_CARLO'
      ? { ...base, parameters, simulation: { num_simulations: form.simulations, num_time_steps: form.steps, seed: form.seed, scheme: 'QUADRATIC_EXPONENTIAL', antithetic: true } }
      : { ...base, parameters }
  }
  if (engine.startsWith('MERTON')) {
    const parameters = { ...mertonParameters(), jump_intensity: form.jumpIntensity, jump_mean: form.jumpMean, jump_volatility: form.jumpVolatility }
    return engine === 'MERTON_MONTE_CARLO'
      ? { ...base, parameters, simulation: { num_simulations: form.simulations, seed: form.seed } }
      : { ...base, parameters }
  }
  return base
}

export default function PricingPage() {
  const [engine, setEngine] = useState<PricingEngine>('BLACK_SCHOLES')
  const [form, setForm] = useState<PricingForm>(initialForm)
  const [result, setResult] = useState<JsonObject | null>(null)
  const [error, setError] = useState<unknown>(null)
  const [pending, setPending] = useState(false)
  const controller = useRef<AbortController | null>(null)
  const update = <Key extends keyof PricingForm>(key: Key, value: PricingForm[Key]): void => setForm((current) => ({ ...current, [key]: value }))

  const price = async (): Promise<void> => {
    setPending(true); setError(null); setResult(null)
    const current = new AbortController(); controller.current = current
    try {
      const payload = requestPayload(engine, form)
      const priced = await executeResearchOperation(operation(engine), payload, current.signal)
      if (engine === 'BLACK_SCHOLES') {
        const greeks = await executeResearchOperation({ id: 'greeks', label: 'Analytical Greeks', description: '', path: '/api/greeks/black-scholes', template: {} }, payload, current.signal)
        setResult({ ...priced, greeks })
      } else if (engine === 'MERTON_ANALYTICAL') {
        const greeks = await executeResearchOperation({ id: 'merton-greeks', label: 'Merton Greeks', description: '', path: '/api/analytics/merton/greeks', template: {} }, payload, current.signal)
        setResult({ ...priced, greeks })
      } else setResult(priced)
    } catch (caught: unknown) { setError(caught) } finally { setPending(false); controller.current = null }
  }

  const numberInput = (label: string, key: keyof PricingForm, step = 'any', min?: number) => (
    <Input label={label} min={min} onChange={(event) => update(key, Number(event.target.value))} step={step} type="number" value={String(form[key])} />
  )
  const isHeston = engine.startsWith('HESTON')
  const isMerton = engine.startsWith('MERTON')
  const usesSimulation = engine.includes('MONTE_CARLO')

  return (
    <div className="page-stack">
      <PageHeader eyebrow="Valuation" title="Pricing workbench" description="Price a European call or put with seven established engines. Backend validation and numerical diagnostics remain authoritative." />
      <div className="workbench-grid">
        <Card className="workbench-form">
          <h2>Contract and model</h2>
          <Select label="Pricing engine" options={[...engineOptions]} value={engine} onChange={(event) => setEngine(event.target.value as PricingEngine)} />
          <div className="form-grid form-grid--2">
            <Select label="Option type" options={[{ label: 'Call', value: 'CALL' }, { label: 'Put', value: 'PUT' }]} value={form.optionType} onChange={(event) => update('optionType', event.target.value as 'CALL' | 'PUT')} />
            {numberInput('Spot · currency units', 'spot', '0.01', 0.000001)}
            {numberInput('Strike · currency units', 'strike', '0.01', 0.000001)}
            {numberInput('Maturity · years', 'maturity', '0.01', 0)}
            {numberInput('Risk-free rate · decimal', 'rate', '0.0001')}
            {numberInput('Dividend yield · decimal', 'dividend', '0.0001')}
            {numberInput('Volatility · decimal', 'volatility', '0.001', 0)}
          </div>
          {isHeston ? <fieldset><legend>Heston parameters</legend><div className="form-grid form-grid--2">{numberInput('v₀ · variance', 'v0', '0.001', 0)}{numberInput('κ · mean reversion', 'kappa', '0.1', 0)}{numberInput('θ · long-run variance', 'theta', '0.001', 0)}{numberInput('σᵥ · vol of variance', 'sigmaV', '0.01', 0)}{numberInput('ρ · correlation', 'rho', '0.05')}</div><p className="fine-print"><HelpTip term="Feller condition">The CIR variance boundary condition 2κθ ≥ σᵥ². A violation is a diagnostic, not an automatic rejection.</HelpTip></p></fieldset> : null}
          {isMerton ? <fieldset><legend>Merton jump parameters</legend><div className="form-grid form-grid--2">{numberInput('λ · jumps/year', 'jumpIntensity', '0.1', 0)}{numberInput('μⱼ · mean log jump', 'jumpMean', '0.01')}{numberInput('σⱼ · log-jump volatility', 'jumpVolatility', '0.01', 0)}</div></fieldset> : null}
          {usesSimulation || engine === 'CRR' ? <fieldset><legend>Numerical configuration</legend><div className="form-grid form-grid--2">{usesSimulation ? numberInput('Simulation paths · count', 'simulations', '1000', 1) : null}{usesSimulation ? numberInput('Seed · integer', 'seed', '1', 0) : null}{engine === 'CRR' || engine === 'HESTON_MONTE_CARLO' ? numberInput(engine === 'CRR' ? 'Tree steps · count' : 'Time steps · count', 'steps', '1', 1) : null}</div></fieldset> : null}
          <div className="button-row"><Button disabled={pending} onClick={() => void price()}>{pending ? 'Pricing…' : 'Price option'}</Button>{pending ? <Button variant="secondary" onClick={() => controller.current?.abort()}>Cancel</Button> : null}</div>
          <Alert title="Units" variant="info">Rates and volatility use decimal units. Vega and Rho are returned per absolute 1.00 unit; Theta is per calendar year.</Alert>
        </Card>
        <div className="workbench-result">{pending ? <LoadingState label={`Running ${operation(engine).label}`} /> : null}{error ? <ErrorPanel error={error} onRetry={() => void price()} /> : null}{result ? <ResultPanel result={result} title={`${operation(engine).label} result`} exportName="pricing-result.json" /> : null}{!pending && !error && !result ? <Card className="result-awaiting"><span className="eyebrow">Ready</span><h2>Price and inspect</h2><p>The response will preserve uncertainty, runtime, numerical metadata, warnings, and available Greeks.</p></Card> : null}</div>
      </div>
    </div>
  )
}
