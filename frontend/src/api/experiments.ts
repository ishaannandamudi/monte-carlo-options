import { apiGet, apiPost } from './client'
import type { JsonObject } from './types'

export interface ExperimentRunSummary {
  run_id: string
  experiment_id: string
  parent_run_id?: string | null
  status: 'PENDING' | 'RUNNING' | 'COMPLETED' | 'FAILED'
  created_at: string
  completed_at?: string | null
  duration_ms?: number | null
  experiment_type: string
  input_signature: string
  definition: JsonObject
  config_snapshot: JsonObject
  environment_snapshot: JsonObject
  result_summary?: JsonObject | null
  metrics?: Array<{ name: string; value: number; unit?: string | null; category?: string | null }>
  artifacts?: Array<{ artifact_id: string; artifact_type: string; path: string; size_bytes: number; description?: string | null }>
  tags?: string[]
  flags?: string[]
  failure?: JsonObject | null
  notes?: string | null
}

export interface ExperimentListResult {
  items: ExperimentRunSummary[]
  page: number
  page_size: number
  total: number
}

export interface ExperimentFilters {
  page: number
  pageSize: number
  experimentType?: string
  status?: string
  tag?: string
  sort?: string
}

export function listExperiments(filters: ExperimentFilters, signal?: AbortSignal): Promise<ExperimentListResult> {
  const query = new URLSearchParams({ page: String(filters.page), page_size: String(filters.pageSize), sort: filters.sort ?? 'NEWEST' })
  if (filters.experimentType) query.set('experiment_type', filters.experimentType)
  if (filters.status) query.set('status', filters.status)
  if (filters.tag) query.set('tag', filters.tag)
  return apiGet<ExperimentListResult>(`/api/experiments?${query.toString()}`, signal)
}

export function getExperiment(runId: string, signal?: AbortSignal): Promise<ExperimentRunSummary> {
  return apiGet<ExperimentRunSummary>(`/api/experiments/${encodeURIComponent(runId)}`, signal)
}

export function reproduceExperiment(runId: string): Promise<JsonObject> {
  return apiPost<JsonObject>(`/api/experiments/${encodeURIComponent(runId)}/reproduce`, { tolerance: { absolute: 1e-10, relative: 1e-8 } }, { timeoutMs: 300_000 })
}

export function compareExperiments(runIds: readonly string[]): Promise<JsonObject> {
  return apiPost<JsonObject>('/api/experiments/compare', { run_ids: runIds, max_differences: 200 }, { timeoutMs: 180_000 })
}
