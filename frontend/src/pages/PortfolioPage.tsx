import { useRef, useState } from 'react'

import { executeResearchOperation, type ResearchOperation } from '../api/research'
import { hestonParameters, market } from '../api/templates'
import type { JsonObject } from '../api/types'
import { ResultPanel } from '../components/research/ResultPanel'
import { Alert, Button, Card, ErrorPanel, Input, LoadingState, PageHeader, Select } from '../components/ui'

type InstrumentClass = 'VANILLA_EUROPEAN' | 'VANILLA_AMERICAN' | 'ASIAN' | 'DIGITAL' | 'BARRIER' | 'BASKET'
type PortfolioWorkflow = 'value' | 'greeks' | 'scenario' | 'grid' | 'stress' | 'pnl'

interface PositionDraft { key: number; positionId: string; quantity: number; instrumentClass: InstrumentClass; model: string }

const instrumentOptions = [
  { value: 'VANILLA_EUROPEAN', label: 'European vanilla' }, { value: 'VANILLA_AMERICAN', label: 'American vanilla' },
  { value: 'ASIAN', label: 'Asian' }, { value: 'DIGITAL', label: 'Digital' },
  { value: 'BARRIER', label: 'Barrier' }, { value: 'BASKET', label: 'Basket' },
] as const

const models: Record<InstrumentClass, readonly { value: string; label: string }[]> = {
  VANILLA_EUROPEAN: [{ value: 'BLACK_SCHOLES', label: 'Black-Scholes' }, { value: 'MONTE_CARLO_GBM', label: 'GBM Monte Carlo' }, { value: 'CRR_BINOMIAL', label: 'CRR' }, { value: 'HESTON_ANALYTICAL', label: 'Heston analytical' }, { value: 'MERTON_ANALYTICAL', label: 'Merton analytical' }],
  VANILLA_AMERICAN: [{ value: 'CRR_AMERICAN', label: 'CRR American' }, { value: 'LONGSTAFF_SCHWARTZ', label: 'Longstaff-Schwartz' }],
  ASIAN: [{ value: 'GEOMETRIC_ASIAN_ANALYTICAL', label: 'Geometric analytical' }, { value: 'ASIAN_MONTE_CARLO', label: 'Asian Monte Carlo' }],
  DIGITAL: [{ value: 'DIGITAL_ANALYTICAL', label: 'Digital analytical' }, { value: 'DIGITAL_MONTE_CARLO', label: 'Digital Monte Carlo' }],
  BARRIER: [{ value: 'BARRIER_ANALYTICAL', label: 'Barrier analytical' }, { value: 'BARRIER_MONTE_CARLO', label: 'Barrier Monte Carlo' }],
  BASKET: [{ value: 'GEOMETRIC_BASKET_ANALYTICAL', label: 'Geometric analytical' }, { value: 'BASKET_MONTE_CARLO', label: 'Basket Monte Carlo' }],
}

const workflowOptions = [
  { value: 'value', label: 'Portfolio value' }, { value: 'greeks', label: 'Portfolio Greeks' },
  { value: 'scenario', label: 'Scenario analysis' }, { value: 'grid', label: 'Spot × volatility grid' },
  { value: 'stress', label: 'Named stresses' }, { value: 'pnl', label: 'P&L explain' },
] as const

const operations: Record<PortfolioWorkflow, ResearchOperation> = {
  value: { id: 'value', label: 'Portfolio value', description: '', path: '/api/portfolio/value', template: {} },
  greeks: { id: 'greeks', label: 'Portfolio Greeks', description: '', path: '/api/portfolio/greeks', template: {}, timeoutMs: 180_000 },
  scenario: { id: 'scenario', label: 'Scenario analysis', description: '', path: '/api/portfolio/scenario', template: {}, timeoutMs: 180_000 },
  grid: { id: 'grid', label: 'Scenario grid', description: '', path: '/api/portfolio/scenario-grid', template: {}, timeoutMs: 240_000 },
  stress: { id: 'stress', label: 'Named stresses', description: '', path: '/api/portfolio/stress', template: {}, timeoutMs: 240_000 },
  pnl: { id: 'pnl', label: 'P&L explain', description: '', path: '/api/portfolio/pnl-explain', template: {}, timeoutMs: 180_000 },
}

function positionPayload(draft: PositionDraft, spot: number, strike: number, maturity: number, volatility: number): JsonObject {
  const commonMarket = { ...market(), volatility }
  const common = { position_id: draft.positionId, quantity: draft.quantity, instrument_class: draft.instrumentClass, underlying_id: 'SPY', model: draft.model }
  if (draft.instrumentClass === 'VANILLA_EUROPEAN' || draft.instrumentClass === 'VANILLA_AMERICAN') return { ...common, instrument: { option_type: draft.instrumentClass === 'VANILLA_AMERICAN' ? 'PUT' : 'CALL', exercise_style: draft.instrumentClass === 'VANILLA_AMERICAN' ? 'AMERICAN' : 'EUROPEAN', spot, strike, maturity_years: maturity }, market: commonMarket, tree: { steps: 200 }, heston_parameters: hestonParameters() }
  if (draft.instrumentClass === 'ASIAN') return { ...common, instrument: { option_type: 'CALL', spot, strike, maturity_years: maturity, averaging_type: 'GEOMETRIC', averaging_schedule: { monitoring_count: 12 } }, market: commonMarket }
  if (draft.instrumentClass === 'DIGITAL') return { ...common, instrument: { option_type: 'CALL', spot, strike, maturity_years: maturity, digital_type: 'CASH_OR_NOTHING', cash_payout: 1 }, market: commonMarket }
  if (draft.instrumentClass === 'BARRIER') return { ...common, instrument: { option_type: 'CALL', barrier_direction: 'UP', barrier_style: 'OUT', spot, strike, barrier: spot * 1.3, maturity_years: maturity, monitoring_mode: 'BROWNIAN_BRIDGE' }, market: commonMarket }
  return { position_id: draft.positionId, quantity: draft.quantity, instrument_class: draft.instrumentClass, model: draft.model, instrument: { option_type: 'CALL', strike, maturity_years: maturity, weights: [0.6, 0.4], basket_type: 'GEOMETRIC' }, market: { asset_ids: ['AAA', 'BBB'], spots: [spot, spot * 0.8], volatilities: [volatility, volatility + 0.05], risk_free_rate: 0.03, dividend_yields: [0.01, 0.02], correlation_matrix: [[1, 0.3], [0.3, 1]] } }
}

export default function PortfolioPage() {
  const [positions, setPositions] = useState<PositionDraft[]>([{ key: 1, positionId: 'call-1', quantity: 1, instrumentClass: 'VANILLA_EUROPEAN', model: 'BLACK_SCHOLES' }])
  const [nextKey, setNextKey] = useState(2)
  const [newClass, setNewClass] = useState<InstrumentClass>('VANILLA_EUROPEAN')
  const [workflow, setWorkflow] = useState<PortfolioWorkflow>('value')
  const [spot, setSpot] = useState(100); const [strike, setStrike] = useState(100); const [maturity, setMaturity] = useState(1); const [volatility, setVolatility] = useState(0.2)
  const [spotShock, setSpotShock] = useState(-0.1); const [volShock, setVolShock] = useState(0.05)
  const [result, setResult] = useState<JsonObject | null>(null); const [error, setError] = useState<unknown>(null); const [pending, setPending] = useState(false)
  const controller = useRef<AbortController | null>(null)

  const add = (): void => { const model = models[newClass][0]?.value ?? 'BLACK_SCHOLES'; setPositions((current) => [...current, { key: nextKey, positionId: `position-${String(nextKey)}`, quantity: 1, instrumentClass: newClass, model }]); setNextKey((value) => value + 1) }
  const update = (key: number, patch: Partial<PositionDraft>): void => setPositions((current) => current.map((item) => item.key === key ? { ...item, ...patch } : item))
  const portfolio: JsonObject = { portfolio_id: 'frontend-portfolio', base_currency: 'USD', positions: positions.map((item) => positionPayload(item, spot, strike, maturity, volatility)) }
  const payload = (): JsonObject => {
    if (workflow === 'value' || workflow === 'greeks') return { portfolio }
    if (workflow === 'scenario' || workflow === 'pnl') return { portfolio, scenario: { name: 'custom scenario', spot_relative_shift: spotShock, volatility_absolute_shift: volShock } }
    if (workflow === 'grid') return { portfolio, axes: [{ axis: 'SPOT_RELATIVE', values: [-0.15, -0.075, 0, 0.075, 0.15] }, { axis: 'VOLATILITY_ABSOLUTE', values: [-0.05, 0, 0.05, 0.1] }] }
    return { portfolio, scenarios: [{ name: 'Equity selloff', spot_relative_shift: -0.2, volatility_absolute_shift: 0.1 }, { name: 'Rates higher', rate_absolute_shift: 0.01 }, { name: 'One month decay', time_years_passed: 0.0833333333 }] }
  }
  const run = async (): Promise<void> => { setPending(true); setError(null); setResult(null); const current = new AbortController(); controller.current = current; try { setResult(await executeResearchOperation(operations[workflow], payload(), current.signal)) } catch (caught: unknown) { setError(caught) } finally { setPending(false); controller.current = null } }

  return <div className="page-stack">
    <PageHeader eyebrow="Portfolio" title="Portfolio, scenarios, and P&L" description="Build a signed mixed-instrument portfolio and run valuation, Greeks, full-repricing scenarios, grids, stresses, and P&L explain." />
    <div className="portfolio-layout">
      <Card className="portfolio-builder"><div className="section__header"><div><h2>Positions</h2><p>{positions.length} position{positions.length === 1 ? '' : 's'} · positive is long, negative is short</p></div></div>
        <div className="add-position"><Select label="Instrument to add" options={[...instrumentOptions]} value={newClass} onChange={(event) => setNewClass(event.target.value as InstrumentClass)} /><Button onClick={add}>Add position</Button></div>
        <div className="position-list">{positions.map((position) => <div className="position-row" key={position.key}><Input label="Position ID" value={position.positionId} onChange={(event) => update(position.key, { positionId: event.target.value })} /><Input label="Quantity · signed" type="number" step="0.1" value={String(position.quantity)} onChange={(event) => update(position.key, { quantity: Number(event.target.value) })} /><Select label="Instrument" options={[...instrumentOptions]} value={position.instrumentClass} onChange={(event) => { const instrumentClass = event.target.value as InstrumentClass; update(position.key, { instrumentClass, model: models[instrumentClass][0]?.value ?? '' }) }} /><Select label="Model" options={[...models[position.instrumentClass]]} value={position.model} onChange={(event) => update(position.key, { model: event.target.value })} /><Button aria-label={`Remove position ${position.positionId}`} disabled={positions.length === 1} onClick={() => setPositions((current) => current.filter((item) => item.key !== position.key))} variant="ghost">Remove</Button></div>)}</div>
      </Card>
      <Card><h2>Shared market and workflow</h2><div className="form-grid form-grid--4"><Input label="Spot · currency" type="number" min="0.01" value={spot} onChange={(event) => setSpot(Number(event.target.value))} /><Input label="Strike · currency" type="number" min="0.01" value={strike} onChange={(event) => setStrike(Number(event.target.value))} /><Input label="Maturity · years" type="number" min="0" step="0.1" value={maturity} onChange={(event) => setMaturity(Number(event.target.value))} /><Input label="Volatility · decimal" type="number" min="0" step="0.01" value={volatility} onChange={(event) => setVolatility(Number(event.target.value))} /></div><Select label="Analysis" options={[...workflowOptions]} value={workflow} onChange={(event) => setWorkflow(event.target.value as PortfolioWorkflow)} />{workflow === 'scenario' || workflow === 'pnl' ? <div className="form-grid form-grid--2"><Input label="Relative spot shock · decimal" type="number" step="0.01" value={spotShock} onChange={(event) => setSpotShock(Number(event.target.value))} /><Input label="Absolute volatility shock · decimal" type="number" step="0.01" value={volShock} onChange={(event) => setVolShock(Number(event.target.value))} /></div> : null}<div className="button-row"><Button disabled={pending} onClick={() => void run()}>{pending ? 'Running…' : `Run ${operations[workflow].label}`}</Button>{pending ? <Button variant="secondary" onClick={() => controller.current?.abort()}>Cancel</Button> : null}</div><Alert title="Authoritative valuation" variant="info">Full repricing is authoritative. Greek approximations and their residuals remain separate diagnostics.</Alert></Card>
    </div>
    {pending ? <LoadingState label={`Running ${operations[workflow].label}`} /> : null}{error ? <ErrorPanel error={error} onRetry={() => void run()} /> : null}{result ? <ResultPanel result={result} title={`${operations[workflow].label} result`} exportName="portfolio-result.json" /> : null}
  </div>
}
