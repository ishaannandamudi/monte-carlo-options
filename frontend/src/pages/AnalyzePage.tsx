import { useEffect, useMemo, useState, type SyntheticEvent } from 'react'
import { useQuery } from '@tanstack/react-query'

import {
  analyzeOptions,
  type AnalysisOptionType,
  type AnalyzedOptionContract,
  type OptionsAnalysisRequest,
} from '../api/optionsAnalyzer'
import { MultiSeriesLineChart } from '../components/visualization/MultiSeriesLineChart'
import { Alert, Badge, Button, Card, ErrorPanel, Input, LoadingState, Select } from '../components/ui'
import {
  formatCount,
  formatCurrency,
  formatDecimal,
  formatGreek,
  formatPercent,
  formatRuntime,
  formatVolatility,
} from '../utils/formatters'

type SortKey = 'strike' | 'bid' | 'ask' | 'market_midpoint' | 'fair_value' | 'fair_value_gap_pct' | 'implied_volatility' | 'fair_value_volatility' | 'delta' | 'theta_daily' | 'volume' | 'open_interest'

const loadingStages = [
  'Fetching underlying quote…',
  'Loading expirations and option chain…',
  'Calculating implied volatility…',
  'Pricing American contracts and Greeks…',
]

function finiteNumber(value: string): number | undefined {
  if (value.trim() === '') return undefined
  const parsed = Number(value)
  return Number.isFinite(parsed) ? parsed : undefined
}

function gapClass(value: number | null): string {
  if (value === null || value === 0) return ''
  return value > 0 ? 'gap-positive' : 'gap-negative'
}

function ContractDetails({ contract, onClose }: { contract: AnalyzedOptionContract; onClose: () => void }) {
  return (
    <Card className="contract-detail">
      <div className="contract-detail__header">
        <div><span className="eyebrow">Selected contract</span><h2>{contract.option_type} · {formatCurrency(contract.strike)} strike</h2><code>{contract.provider_symbol}</code></div>
        <Button onClick={onClose} size="small" variant="ghost">Close details</Button>
      </div>
      <div className="detail-grid">
        <section><h3>Market</h3><dl><dt>Bid / ask</dt><dd>{formatCurrency(contract.bid)} / {formatCurrency(contract.ask)}</dd><dt>Midpoint</dt><dd>{formatCurrency(contract.market_midpoint)}</dd><dt>Spread</dt><dd>{formatCurrency(contract.spread)} ({formatPercent(contract.spread_pct)})</dd><dt>Volume / open interest</dt><dd>{formatCount(contract.volume)} / {formatCount(contract.open_interest)}</dd><dt>Bid / ask size</dt><dd>{formatCount(contract.bid_size)} / {formatCount(contract.ask_size)}</dd></dl></section>
        <section><h3>Contract</h3><dl><dt>Expiration</dt><dd>{contract.expiration}</dd><dt>Days to expiration</dt><dd>{contract.days_to_expiration}</dd><dt>Exercise style</dt><dd>{contract.exercise_style}</dd><dt>Intrinsic / time value</dt><dd>{formatCurrency(contract.intrinsic_value)} / {formatCurrency(contract.time_value)}</dd></dl></section>
        <section><h3>Volatility</h3><dl><dt>Market IV</dt><dd>{formatVolatility(contract.implied_volatility)} <small>BSM-equivalent</small></dd><dt>Fair-value vol</dt><dd>{formatVolatility(contract.fair_value_volatility)}</dd><dt>Fair-vol source</dt><dd>{contract.fair_value_volatility_source.replaceAll('_', ' ')}</dd><dt>Provider IV</dt><dd>{formatVolatility(contract.provider_implied_volatility)}</dd></dl></section>
        <section><h3>Fair value</h3><dl><dt>CRR American value</dt><dd>{formatCurrency(contract.fair_value)}</dd><dt>Market midpoint</dt><dd>{formatCurrency(contract.market_midpoint)}</dd><dt>Fair-value gap</dt><dd className={gapClass(contract.fair_value_gap)}>{formatCurrency(contract.fair_value_gap)} ({formatPercent(contract.fair_value_gap_pct)})</dd><dt>Early-exercise premium</dt><dd>{formatCurrency(contract.early_exercise_premium)}</dd></dl></section>
        <section><h3>Greeks</h3><dl><dt>Delta</dt><dd>{formatGreek(contract.delta)}</dd><dt>Gamma</dt><dd>{formatGreek(contract.gamma)}</dd><dt>Vega / vol point</dt><dd>{formatGreek(contract.vega_per_vol_point)}</dd><dt>Theta / day</dt><dd>{formatGreek(contract.theta_daily)}</dd><dt>Rho</dt><dd>{formatGreek(contract.rho)}</dd></dl></section>
        <section><h3>Diagnostics</h3><dl><dt>Quote age</dt><dd>{formatDecimal(contract.quote_age_seconds, { maximumFractionDigits: 0 })} sec</dd><dt>Data quality</dt><dd>{contract.analysis_quality.replaceAll('_', ' ')}</dd><dt>IV status</dt><dd>{contract.implied_volatility_status ?? 'Unavailable'}</dd><dt>Tree</dt><dd>{contract.tree_steps} steps · {formatRuntime(contract.model_runtime_ms)}</dd></dl><div className="flag-list">{[...contract.quality_flags, ...contract.warning_flags].map((flag) => <Badge key={flag} variant="warning">{flag.replaceAll('_', ' ')}</Badge>)}</div></section>
      </div>
    </Card>
  )
}

export default function AnalyzePage() {
  const [symbol, setSymbol] = useState('')
  const [symbolError, setSymbolError] = useState<string>()
  const [request, setRequest] = useState<OptionsAnalysisRequest | null>(null)
  const [side, setSide] = useState<AnalysisOptionType>('BOTH')
  const [selectedSymbol, setSelectedSymbol] = useState<string>()
  const [sortKey, setSortKey] = useState<SortKey>('strike')
  const [sortAscending, setSortAscending] = useState(true)
  const [loadingStage, setLoadingStage] = useState(0)
  const [targetDays, setTargetDays] = useState('45')
  const [riskFreeRate, setRiskFreeRate] = useState('')
  const [dividendYield, setDividendYield] = useState('')
  const [treeSteps, setTreeSteps] = useState('400')
  const [strikeWindow, setStrikeWindow] = useState('0.15')
  const [maxContracts, setMaxContracts] = useState('20')
  const [maxQuoteAge, setMaxQuoteAge] = useState('900')
  const [allowLast, setAllowLast] = useState(false)

  const analysis = useQuery({
    queryKey: ['american-options-analysis', request],
    queryFn: ({ signal }) => analyzeOptions(request as OptionsAnalysisRequest, signal),
    enabled: request !== null,
    retry: false,
    staleTime: 120_000,
  })

  useEffect(() => {
    if (!analysis.isPending || request === null) return undefined
    const timer = window.setInterval(() => setLoadingStage((stage) => Math.min(stage + 1, loadingStages.length - 1)), 850)
    return () => window.clearInterval(timer)
  }, [analysis.isPending, request])

  const makeRequest = (expiration?: string | null): OptionsAnalysisRequest => ({
    symbol: symbol.trim().toUpperCase(),
    expiration: expiration ?? null,
    option_type: 'BOTH',
    max_contracts: finiteNumber(maxContracts),
    strike_window_pct: finiteNumber(strikeWindow),
    target_expiration_days: finiteNumber(targetDays),
    risk_free_rate: finiteNumber(riskFreeRate) ?? null,
    dividend_yield: finiteNumber(dividendYield) ?? null,
    tree_steps: finiteNumber(treeSteps),
    max_quote_age_seconds: finiteNumber(maxQuoteAge),
    allow_last_price_fallback: allowLast,
  })

  const submit = (event: SyntheticEvent<HTMLFormElement>) => {
    event.preventDefault()
    const normalized = symbol.trim().toUpperCase()
    if (!/^[A-Z][A-Z0-9.-]{0,14}$/.test(normalized)) {
      setSymbolError('Enter a valid ticker using letters, numbers, a dot, or a hyphen.')
      return
    }
    setSymbol(normalized)
    setSymbolError(undefined)
    setLoadingStage(0)
    setSelectedSymbol(undefined)
    setRequest(makeRequest())
  }

  const contracts = useMemo(() => {
    const filtered = (analysis.data?.contracts ?? []).filter((contract) => side === 'BOTH' || contract.option_type === side)
    return [...filtered].sort((left, right) => {
      const leftValue = left[sortKey]
      const rightValue = right[sortKey]
      if (leftValue === rightValue) return left.provider_symbol.localeCompare(right.provider_symbol)
      if (leftValue === null) return 1
      if (rightValue === null) return -1
      const order = leftValue < rightValue ? -1 : 1
      return sortAscending ? order : -order
    })
  }, [analysis.data, side, sortAscending, sortKey])
  const selected = analysis.data?.contracts.find((contract) => contract.provider_symbol === selectedSymbol)
  const sort = (key: SortKey) => {
    if (sortKey === key) setSortAscending((value) => !value)
    else { setSortKey(key); setSortAscending(true) }
  }
  const sortButton = (label: string, key: SortKey) => <button aria-label={`Sort by ${label}`} className="table-sort" onClick={() => sort(key)} type="button">{label}{sortKey === key ? (sortAscending ? ' ↑' : ' ↓') : ''}</button>

  return (
    <div className="page-stack analyzer-page">
      <header className="analyzer-hero">
        <span className="page-header__eyebrow">Market-data workflow</span>
        <h1>American Options Analyzer</h1>
        <p>Enter a ticker to compare market option prices with model-based fair values.</p>
        <form className="ticker-form" onSubmit={submit}>
          <Input autoComplete="off" error={symbolError} label="Ticker symbol" onChange={(event) => setSymbol(event.target.value)} placeholder="AAPL" value={symbol} />
          <Button disabled={analysis.isFetching} type="submit">Analyze</Button>
        </form>
        <div className="ticker-examples" aria-label="Example tickers">Examples: {['AAPL', 'MSFT', 'NVDA', 'SPY'].map((example) => <button key={example} onClick={() => setSymbol(example)} type="button">{example}</button>)}</div>
        <details className="advanced-panel">
          <summary>Advanced assumptions</summary>
          <div className="advanced-grid">
            <Input label="Target expiration days" min="1" onChange={(event) => setTargetDays(event.target.value)} type="number" value={targetDays} />
            <Input hint="Leave blank for configured default" label="Risk-free rate" onChange={(event) => setRiskFreeRate(event.target.value)} step="0.001" type="number" value={riskFreeRate} />
            <Input hint="Leave blank for configured default" label="Dividend yield" onChange={(event) => setDividendYield(event.target.value)} step="0.001" type="number" value={dividendYield} />
            <Input label="CRR tree steps" min="1" onChange={(event) => setTreeSteps(event.target.value)} type="number" value={treeSteps} />
            <Input label="Strike window" max="2" min="0.01" onChange={(event) => setStrikeWindow(event.target.value)} step="0.01" type="number" value={strikeWindow} />
            <Input label="Maximum contracts" max="100" min="1" onChange={(event) => setMaxContracts(event.target.value)} type="number" value={maxContracts} />
            <Input label="Maximum quote age (seconds)" min="1" onChange={(event) => setMaxQuoteAge(event.target.value)} type="number" value={maxQuoteAge} />
            <label className="checkbox-field"><input checked={allowLast} onChange={(event) => setAllowLast(event.target.checked)} type="checkbox" />Allow last-price fallback when midpoint is unavailable</label>
          </div>
        </details>
      </header>

      {analysis.isPending && request !== null ? <LoadingState label={loadingStages[loadingStage]} /> : null}
      {analysis.error ? <ErrorPanel error={analysis.error} onRetry={() => void analysis.refetch()} title="Options analysis unavailable" /> : null}

      {analysis.data ? (
        <>
          <section className="analysis-header" aria-labelledby="analysis-heading">
            <div><span className="page-header__eyebrow">Analysis result</span><h2 id="analysis-heading">{analysis.data.symbol}</h2><p>Underlying price <strong>{formatCurrency(analysis.data.underlying.price)}</strong> · Quote time {new Date(analysis.data.underlying.timestamp).toLocaleString()}</p></div>
            <Button disabled={analysis.isFetching} onClick={() => void analysis.refetch()} size="small" variant="secondary">Refresh data</Button>
            <div className="analysis-facts">
              <Select label="Selected expiration" onChange={(event) => { setLoadingStage(0); setSelectedSymbol(undefined); setRequest(makeRequest(event.target.value)) }} options={analysis.data.available_expirations.map((expiration) => ({ label: expiration, value: expiration }))} value={analysis.data.selected_expiration} />
              <div><span>Days to expiration</span><strong>{analysis.data.assumptions.selected_expiration_days}</strong></div>
              <div><span>ATM strike</span><strong>{formatCurrency(analysis.data.summary.atm_strike)}</strong></div>
              <div><span>Contracts analyzed</span><strong>{analysis.data.summary.contracts_analyzed}</strong></div>
              <div><span>Data source</span><strong>{analysis.data.data_source}</strong></div>
              <div><span>Data age</span><strong>{formatDecimal(analysis.data.underlying.data_age_seconds, { maximumFractionDigits: 0 })} sec</strong></div>
            </div>
            <p className="assumption-line">CRR American · {analysis.data.assumptions.tree_steps} steps · risk-free rate {formatPercent(analysis.data.assumptions.risk_free_rate)} ({analysis.data.assumptions.risk_free_rate_source.toLowerCase().replaceAll('_', ' ')}) · dividend yield {formatPercent(analysis.data.assumptions.dividend_yield)} ({analysis.data.assumptions.dividend_yield_source.toLowerCase().replaceAll('_', ' ')})</p>
          </section>

          {analysis.data.warnings.map((warning) => <Alert key={warning} title="Market-data warning" variant="warning">{warning}</Alert>)}

          <div className="chain-toolbar">
            <div aria-label="Option type" className="segmented-control" role="group">{(['CALL', 'PUT', 'BOTH'] as const).map((value) => <button aria-pressed={side === value} className={side === value ? 'is-active' : ''} key={value} onClick={() => setSide(value)} type="button">{value === 'CALL' ? 'Calls' : value === 'PUT' ? 'Puts' : 'Both'}</button>)}</div>
            <span>{contracts.length} shown · click column headings to sort</span>
          </div>

          <div className="chain-table-wrap">
            <table className="chain-table">
              <caption>American options model comparison for {analysis.data.symbol}</caption>
              <thead><tr><th>{sortButton('Strike', 'strike')}</th><th>Type</th><th>{sortButton('Bid', 'bid')}</th><th>{sortButton('Ask', 'ask')}</th><th>{sortButton('Market', 'market_midpoint')}</th><th>{sortButton('Fair value', 'fair_value')}</th><th>{sortButton('Gap %', 'fair_value_gap_pct')}</th><th>{sortButton('IV', 'implied_volatility')}</th><th>{sortButton('Fair vol', 'fair_value_volatility')}</th><th>{sortButton('Delta', 'delta')}</th><th>{sortButton('Theta/day', 'theta_daily')}</th><th>{sortButton('Volume', 'volume')}</th><th>{sortButton('Open interest', 'open_interest')}</th><th>Details</th></tr></thead>
              <tbody>{contracts.map((contract) => <tr className={selectedSymbol === contract.provider_symbol ? 'is-selected' : ''} key={contract.provider_symbol}><td>{formatCurrency(contract.strike)}</td><td>{contract.option_type}</td><td>{formatCurrency(contract.bid)}</td><td>{formatCurrency(contract.ask)}</td><td>{formatCurrency(contract.market_midpoint)}</td><td>{formatCurrency(contract.fair_value)}</td><td className={gapClass(contract.fair_value_gap_pct)}>{formatPercent(contract.fair_value_gap_pct)}</td><td>{formatVolatility(contract.implied_volatility)}</td><td>{formatVolatility(contract.fair_value_volatility)}</td><td>{formatGreek(contract.delta)}</td><td>{formatGreek(contract.theta_daily)}</td><td>{formatCount(contract.volume)}</td><td>{formatCount(contract.open_interest)}</td><td><Button aria-label={`View details for ${contract.provider_symbol}`} onClick={() => setSelectedSymbol(contract.provider_symbol)} size="small" variant="ghost">View</Button></td></tr>)}</tbody>
            </table>
          </div>

          {selected ? <ContractDetails contract={selected} onClose={() => setSelectedSymbol(undefined)} /> : null}

          <section aria-labelledby="smile-heading"><div className="section__header"><div><h2 id="smile-heading">Volatility smile</h2><p>BSM-equivalent market IV and the independent volatility used by the American fair-value model.</p></div></div><MultiSeriesLineChart series={[{ label: 'Market IV', points: analysis.data.smile.filter((point) => point.market_implied_volatility !== null && (side === 'BOTH' || point.option_type === side)).map((point) => ({ x: point.strike, y: point.market_implied_volatility as number, selected: point.provider_symbol === selectedSymbol })) }, { label: 'Fair-value vol', points: analysis.data.smile.filter((point) => point.fair_value_volatility !== null && (side === 'BOTH' || point.option_type === side)).map((point) => ({ x: point.strike, y: point.fair_value_volatility as number, selected: point.provider_symbol === selectedSymbol })) }]} title="Selected-expiration volatility smile" xLabel="Strike" yLabel="Implied volatility" /></section>
          <p className="fine-print">Local timing: market data {formatRuntime(analysis.data.timings.market_data_ms)}, IV {formatRuntime(analysis.data.timings.implied_volatility_ms)}, smile {formatRuntime(analysis.data.timings.smile_construction_ms)}, CRR and Greeks {formatRuntime(analysis.data.timings.pricing_and_greeks_ms)}.</p>
        </>
      ) : null}
    </div>
  )
}
