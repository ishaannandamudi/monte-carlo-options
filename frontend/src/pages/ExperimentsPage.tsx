import { useQuery, useQueryClient } from '@tanstack/react-query'
import { useState } from 'react'

import { apiGet } from '../api/client'
import { compareExperiments, getExperiment, listExperiments, reproduceExperiment, type ExperimentRunSummary } from '../api/experiments'
import type { ResearchOperation } from '../api/research'
import { experimentDefinition } from '../api/templates'
import type { JsonObject } from '../api/types'
import { OperationWorkbench } from '../components/research/OperationWorkbench'
import { ResultPanel } from '../components/research/ResultPanel'
import { Alert, Badge, Button, Card, DataTable, EmptyState, ErrorPanel, Input, LoadingState, PageHeader, Select, Tabs, type DataTableColumn } from '../components/ui'
import { copyText, downloadJson } from '../utils/export'
import { formatRuntime } from '../utils/formatters'

const createOperations: readonly ResearchOperation[] = [
  { id: 'run', label: 'Run experiment', description: 'Execute a typed research operation and persist its inputs, outputs, metrics, environment, and signature.', path: '/api/experiments/run', template: experimentDefinition(), timeoutMs: 300_000 },
  { id: 'sweep', label: 'Parameter sweep', description: 'Run a bounded Cartesian sweep with deterministic child seed lineage.', path: '/api/experiments/sweep', template: { definition: experimentDefinition(), axes: [{ path: 'contract.strike', values: [90, 100, 110] }] }, timeoutMs: 600_000 },
  { id: 'benchmark', label: 'Benchmark', description: 'Warm an operation and persist repeated timing samples separately from tracking overhead.', path: '/api/experiments/benchmark', template: { target: { experiment_type: 'BENCHMARK', operation: 'NOOP', name: 'Frontend no-op benchmark', tags: ['frontend'], input_payload: { value: 1 } }, operation: 'frontend no-op benchmark', workload_size: 1, warmup_count: 1, repetition_count: 5 }, timeoutMs: 600_000 },
]

const experimentTypes = ['', 'PRICING', 'CONVERGENCE', 'CALIBRATION', 'MODEL_COMPARISON', 'PORTFOLIO_SCENARIO', 'VAR_ES', 'REGIME', 'QMC', 'BENCHMARK']

function statusVariant(status: string): 'success' | 'danger' | 'warning' | 'neutral' {
  if (status === 'COMPLETED') return 'success'
  if (status === 'FAILED') return 'danger'
  if (status === 'RUNNING') return 'warning'
  return 'neutral'
}

export default function ExperimentsPage() {
  const queryClient = useQueryClient()
  const [tab, setTab] = useState('runs')
  const [page, setPage] = useState(1)
  const [experimentType, setExperimentType] = useState('')
  const [status, setStatus] = useState('')
  const [tag, setTag] = useState('')
  const [sort, setSort] = useState('NEWEST')
  const [selectedRunId, setSelectedRunId] = useState<string | null>(null)
  const [comparisonIds, setComparisonIds] = useState<string[]>([])
  const [actionResult, setActionResult] = useState<JsonObject | null>(null)
  const [actionError, setActionError] = useState<unknown>(null)
  const [actionPending, setActionPending] = useState(false)
  const filters = { page, pageSize: 10, experimentType: experimentType || undefined, status: status || undefined, tag: tag.trim() || undefined, sort }
  const listing = useQuery({ queryKey: ['experiments', 'list', filters], queryFn: ({ signal }) => listExperiments(filters, signal) })
  const detail = useQuery({ queryKey: ['experiments', 'detail', selectedRunId], queryFn: ({ signal }) => getExperiment(selectedRunId ?? '', signal), enabled: selectedRunId !== null })

  const toggleCompare = (runId: string): void => setComparisonIds((current) => current.includes(runId) ? current.filter((item) => item !== runId) : current.length < 20 ? [...current, runId] : current)
  const action = async (work: () => Promise<JsonObject>): Promise<void> => { setActionPending(true); setActionError(null); setActionResult(null); try { setActionResult(await work()) } catch (caught: unknown) { setActionError(caught) } finally { setActionPending(false) } }
  const exportRun = async (runId: string): Promise<void> => { await action(async () => { const exported = await apiGet<JsonObject>(`/api/experiments/${encodeURIComponent(runId)}/export`); downloadJson(`experiment-${runId}.json`, exported); return exported }) }
  const maxPage = listing.data ? Math.max(1, Math.ceil(listing.data.total / listing.data.page_size)) : 1
  const columns: readonly DataTableColumn<ExperimentRunSummary>[] = [
    { key: 'select', header: 'Compare', render: (row) => <input aria-label={`Select run ${row.run_id} for comparison`} checked={comparisonIds.includes(row.run_id)} onChange={() => toggleCompare(row.run_id)} type="checkbox" /> },
    { key: 'id', header: 'Run ID', render: (row) => <button className="table-link" onClick={() => setSelectedRunId(row.run_id)} type="button">{row.run_id.slice(0, 12)}…</button> },
    { key: 'type', header: 'Type', render: (row) => row.experiment_type },
    { key: 'status', header: 'Status', render: (row) => <Badge variant={statusVariant(row.status)}>{row.status}</Badge> },
    { key: 'created', header: 'Created', render: (row) => new Date(row.created_at).toLocaleString() },
    { key: 'duration', header: 'Duration', align: 'right', render: (row) => formatRuntime(row.duration_ms) },
    { key: 'tags', header: 'Tags', render: (row) => row.tags?.join(', ') || '—' },
    { key: 'signature', header: 'Signature', render: (row) => <code>{row.input_signature.slice(0, 10)}</code> },
  ]

  return <div className="page-stack">
    <PageHeader eyebrow="Research lineage" title="Experiments" description="Run, search, inspect, compare, reproduce, sweep, and benchmark deterministic research records." />
    <Tabs label="Experiment sections" options={[{ id: 'runs', label: 'Runs' }, { id: 'create', label: 'Run / sweep / benchmark' }]} value={tab} onChange={setTab} />
    {tab === 'create' ? <OperationWorkbench operations={createOperations} title="Tracked research" onResult={(result) => { if (result) void queryClient.invalidateQueries({ queryKey: ['experiments'] }) }} /> : <>
      <Card><div className="experiment-filters"><Select label="Experiment type" options={experimentTypes.map((value) => ({ value, label: value || 'All types' }))} value={experimentType} onChange={(event) => { setExperimentType(event.target.value); setPage(1) }} /><Select label="Status" options={[{ value: '', label: 'All statuses' }, { value: 'COMPLETED', label: 'Completed' }, { value: 'FAILED', label: 'Failed' }, { value: 'RUNNING', label: 'Running' }, { value: 'PENDING', label: 'Pending' }]} value={status} onChange={(event) => { setStatus(event.target.value); setPage(1) }} /><Input label="Tag filter" placeholder="e.g. bsm" value={tag} onChange={(event) => { setTag(event.target.value); setPage(1) }} /><Select label="Sort" options={[{ value: 'NEWEST', label: 'Newest' }, { value: 'OLDEST', label: 'Oldest' }, { value: 'RUNTIME_ASC', label: 'Runtime ascending' }, { value: 'RUNTIME_DESC', label: 'Runtime descending' }]} value={sort} onChange={(event) => setSort(event.target.value)} /></div></Card>
      {listing.isPending ? <LoadingState label="Loading experiment runs" /> : null}{listing.error ? <ErrorPanel error={listing.error} onRetry={() => void listing.refetch()} /> : null}
      {listing.data?.items.length === 0 ? <EmptyState title="No matching experiments" description="Change the filters or create a tracked run." /> : null}
      {listing.data && listing.data.items.length > 0 ? <><DataTable caption="Experiment runs" columns={columns} rows={listing.data.items} getRowKey={(row) => row.run_id} /><div className="pagination"><Button disabled={page <= 1} onClick={() => setPage((value) => value - 1)} variant="secondary">Previous</Button><span>Page {page} of {maxPage} · {listing.data.total} runs</span><Button disabled={page >= maxPage} onClick={() => setPage((value) => value + 1)} variant="secondary">Next</Button></div><div className="button-row"><Button disabled={comparisonIds.length < 2 || actionPending} onClick={() => void action(() => compareExperiments(comparisonIds))}>Compare selected ({comparisonIds.length})</Button><Button onClick={() => setComparisonIds([])} variant="ghost">Clear selection</Button></div></> : null}
      {selectedRunId ? <Card className="experiment-detail"><div className="result-panel__header"><div><span className="eyebrow">Experiment detail</span><h2>{selectedRunId}</h2></div><Button onClick={() => setSelectedRunId(null)} variant="ghost">Close</Button></div>{detail.isPending ? <LoadingState label="Loading experiment detail" /> : null}{detail.error ? <ErrorPanel error={detail.error} onRetry={() => void detail.refetch()} /> : null}{detail.data ? <ExperimentDetail run={detail.data} onAction={action} onExport={exportRun} /> : null}</Card> : null}
      {actionPending ? <LoadingState label="Running experiment action" /> : null}{actionError ? <ErrorPanel error={actionError} /> : null}{actionResult ? <ResultPanel result={actionResult} title="Experiment action result" exportName="experiment-action.json" /> : null}
    </>}
    <Alert title="Artifact safety" variant="info">Artifact metadata is listed as plain text. Arbitrary stored HTML is never injected into this application.</Alert>
  </div>
}

function ExperimentDetail({ run, onAction, onExport }: { run: ExperimentRunSummary; onAction: (work: () => Promise<JsonObject>) => Promise<void>; onExport: (runId: string) => Promise<void> }) {
  const metrics = run.metrics ?? []
  return <div className="detail-stack"><div className="detail-metadata"><div><span>Status</span><Badge variant={statusVariant(run.status)}>{run.status}</Badge></div><div><span>Type</span><strong>{run.experiment_type}</strong></div><div><span>Duration</span><strong>{formatRuntime(run.duration_ms)}</strong></div><div><span>Signature</span><code>{run.input_signature}</code></div></div><div className="button-row"><Button size="small" onClick={() => void copyText(run.run_id)} variant="secondary">Copy run ID</Button><Button size="small" onClick={() => void copyText(run.input_signature)} variant="secondary">Copy signature</Button><Button size="small" disabled={run.status !== 'COMPLETED'} onClick={() => void onAction(() => reproduceExperiment(run.run_id))}>Reproduce</Button><Button size="small" onClick={() => void onExport(run.run_id)} variant="secondary">Export stored JSON</Button></div>{metrics.length > 0 ? <div className="metric-grid metric-grid--compact">{metrics.slice(0, 8).map((metric) => <Card key={metric.name}><span className="metric-card__label">{metric.name}</span><strong className="metric-card__value">{metric.value}</strong><span className="metric-card__detail">{metric.unit ?? metric.category ?? ''}</span></Card>)}</div> : null}{run.artifacts && run.artifacts.length > 0 ? <DataTable caption="Experiment artifacts" columns={[{ key: 'type', header: 'Type', render: (item) => item.artifact_type }, { key: 'path', header: 'Stored path', render: (item) => <code>{item.path}</code> }, { key: 'size', header: 'Bytes', align: 'right', render: (item) => item.size_bytes }, { key: 'description', header: 'Description', render: (item) => item.description ?? '—' }]} rows={run.artifacts} getRowKey={(item) => item.artifact_id} /> : null}<details className="json-result"><summary>Definition, environment, and result summary</summary><pre>{JSON.stringify({ definition: run.definition, config_snapshot: run.config_snapshot, environment_snapshot: run.environment_snapshot, result_summary: run.result_summary, failure: run.failure, flags: run.flags, notes: run.notes }, null, 2)}</pre></details></div>
}
