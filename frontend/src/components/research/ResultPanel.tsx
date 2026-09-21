import { useState } from 'react'

import { isJsonObject, prettyJson, type JsonObject, type JsonValue } from '../../api/types'
import { copyText, downloadJson } from '../../utils/export'
import {
  formatCurrency,
  formatDecimal,
  formatPercent,
  formatRuntime,
  formatScientific,
} from '../../utils/formatters'
import { AutoChart } from '../visualization/AutoChart'
import { Alert, Badge, Button, Card, DataTable, type DataTableColumn, MetricCard } from '../ui'

interface ResultPanelProps {
  result: JsonObject
  title?: string
  exportName?: string
}

const metricDefinitions = [
  ['price', 'Price'],
  ['portfolio_value', 'Portfolio value'],
  ['total_value', 'Net value'],
  ['gross_long_value', 'Gross long value'],
  ['gross_short_value', 'Gross short value'],
  ['base_value', 'Base value'],
  ['scenario_value', 'Scenario value'],
  ['pnl', 'P&L'],
  ['full_repricing_pnl', 'Full repricing P&L'],
  ['total_full_reprice_pnl', 'Full repricing P&L'],
  ['approximation_residual', 'Approximation residual'],
  ['value_at_risk', 'Value at Risk'],
  ['expected_shortfall', 'Expected Shortfall'],
  ['mixture_value_at_risk', 'Mixture VaR'],
  ['mixture_expected_shortfall', 'Mixture ES'],
  ['standard_error', 'Standard error'],
  ['rmse', 'RMSE'],
  ['objective_value', 'Objective'],
  ['log_likelihood', 'Log likelihood'],
  ['runtime_ms', 'Runtime'],
] as const

const greekKeys = ['delta', 'gamma', 'vega', 'theta', 'rho'] as const

function displayValue(key: string, value: JsonValue): string {
  if (value === null) return 'Unavailable'
  if (typeof value === 'string') return value
  if (typeof value === 'boolean') return value ? 'Yes' : 'No'
  if (typeof value !== 'number') return prettyJson(value)
  if (key.includes('runtime')) return formatRuntime(value)
  if (['price', 'value', 'pnl', 'shortfall', 'risk'].some((token) => key.includes(token))) {
    return formatCurrency(value)
  }
  if (key.includes('probability') || key.includes('confidence')) return formatPercent(value)
  if (Math.abs(value) > 0 && (Math.abs(value) < 1e-4 || Math.abs(value) >= 1e7)) {
    return formatScientific(value)
  }
  return formatDecimal(value, { maximumFractionDigits: 6 })
}

function labelValue(value: JsonValue | undefined, fallback: string): string {
  return typeof value === 'string' || typeof value === 'number' ? String(value) : fallback
}

function stringArray(value: JsonValue | undefined): string[] {
  if (!Array.isArray(value)) return []
  return value.flatMap((item): string[] => {
    if (typeof item === 'string') return [item]
    if (!isJsonObject(item)) return []
    const message = item.message ?? item.reason ?? item.flag ?? item.code
    const model = item.model
    const prefix = typeof model === 'string' ? `${model}: ` : ''
    return typeof message === 'string' ? [`${prefix}${message}`] : []
  })
}

function modelRows(result: JsonObject): JsonObject[] {
  const output = result.outputs ?? result.models ?? result.results
  return Array.isArray(output) ? output.filter(isJsonObject) : []
}

export function ResultPanel({ result, title = 'Research result', exportName = 'result.json' }: ResultPanelProps) {
  const [copyStatus, setCopyStatus] = useState<'idle' | 'copied' | 'failed'>('idle')
  const metrics = metricDefinitions.flatMap(([key, label]) =>
    key in result ? [{ key, label, value: result[key] ?? null }] : [],
  )
  const nestedGreeks = isJsonObject(result.greeks)
    ? result.greeks
    : isJsonObject(result.net_greeks)
      ? result.net_greeks
      : null
  const greeks = greekKeys.flatMap((key) => {
    const value = nestedGreeks?.[key] ?? result[key]
    return typeof value === 'number' || value === null ? [{ key, value }] : []
  })
  const warnings = [
    ...stringArray(result.warnings),
    ...stringArray(result.flags),
    ...stringArray(result.risk_flags),
    ...stringArray(result.unavailable_metrics).map((metric) => `${metric} unavailable`),
    ...stringArray(result.exclusions),
    ...stringArray(result.failed_models).map((model) => `${model} failed`),
    ...stringArray(result.excluded_models).map((model) => `${model} excluded`),
  ]
  const rows = modelRows(result)
  const explainTerms = isJsonObject(result.terms) ? result.terms : null
  const columns: readonly DataTableColumn<JsonObject>[] = [
    { key: 'model', header: 'Model / item', render: (row) => labelValue(row.model ?? row.name ?? row.position_id, 'Result') },
    { key: 'status', header: 'Status', render: (row) => <Badge variant={row.status === 'FAILED' ? 'danger' : row.status === 'EXCLUDED' ? 'warning' : 'success'}>{labelValue(row.status, 'SUCCESS')}</Badge> },
    { key: 'price', header: 'Price / value', align: 'right', render: (row) => displayValue('price', row.price ?? row.position_value ?? row.unit_price ?? row.value ?? null) },
    { key: 'runtime', header: 'Runtime', align: 'right', render: (row) => displayValue('runtime_ms', row.runtime_ms ?? null) },
  ]

  const handleCopy = async (): Promise<void> => {
    try {
      await copyText(prettyJson(result))
      setCopyStatus('copied')
    } catch {
      setCopyStatus('failed')
    }
  }

  return (
    <section className="result-panel" aria-live="polite">
      <div className="result-panel__header">
        <div><span className="eyebrow">Backend response</span><h2>{title}</h2></div>
        <div className="button-row">
          <Button onClick={() => void handleCopy()} size="small" variant="secondary">
            {copyStatus === 'copied' ? 'Copied' : copyStatus === 'failed' ? 'Copy unavailable' : 'Copy JSON'}
          </Button>
          <Button onClick={() => downloadJson(exportName, result)} size="small" variant="secondary">Export JSON</Button>
        </div>
      </div>

      {warnings.length > 0 ? <Alert title="Diagnostics and warnings" variant="warning"><ul className="compact-list">{warnings.map((warning) => <li key={warning}>{warning}</li>)}</ul></Alert> : null}

      {metrics.length > 0 ? <div className="metric-grid metric-grid--compact">{metrics.slice(0, 8).map((metric) => <MetricCard key={metric.key} label={metric.label} value={displayValue(metric.key, metric.value)} />)}</div> : null}

      {greeks.length > 0 ? <Card><h3>Greeks</h3><div className="greek-grid">{greeks.map(({ key, value }) => <div key={key}><span>{key.toUpperCase()}</span><strong>{displayValue(key, value)}</strong></div>)}</div><p className="fine-print">Raw backend units: Vega and Rho are per absolute unit; Theta is per calendar year.</p></Card> : null}

      {explainTerms ? <PnLExplain terms={explainTerms} fullPnl={result.total_full_reprice_pnl} /> : null}

      {rows.length > 0 && rows.some((row) => 'model' in row || 'status' in row) ? <DataTable caption="Backend result statuses" columns={columns} getRowKey={(row) => labelValue(row.model ?? row.name ?? row.position_id, JSON.stringify(row))} rows={rows} /> : null}

      <AutoChart result={result} />

      <details className="json-result"><summary>Complete structured result</summary><pre>{prettyJson(result)}</pre></details>
    </section>
  )
}

function PnLExplain({ terms, fullPnl }: { terms: JsonObject; fullPnl: JsonValue | undefined }) {
  const contributionKeys = ['delta', 'gamma', 'vega', 'theta', 'rho', 'unexplained_or_higher_order']
  const contributions = contributionKeys.flatMap((key) => typeof terms[key] === 'number' ? [{ key, value: terms[key] }] : [])
  const maxMagnitude = Math.max(1e-12, ...contributions.map((item) => Math.abs(item.value)))
  return <Card><h3>P&amp;L explain</h3><div className="contribution-bars">{contributions.map((item) => <div className="contribution-bar" key={item.key}><span>{item.key.replaceAll('_', ' ')}</span><div><i className={item.value < 0 ? 'negative' : 'positive'} style={{ width: `${String(Math.max(3, Math.abs(item.value) / maxMagnitude * 100))}%` }} /></div><strong>{formatCurrency(item.value)}</strong></div>)}</div><p className="fine-print">Explained total + higher-order residual = full repriced P&amp;L ({displayValue('pnl', fullPnl ?? null)}).</p></Card>
}
