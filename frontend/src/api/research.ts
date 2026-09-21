import { ApiError, apiPost } from './client'
import { isJsonObject, type JsonObject } from './types'

export interface ResearchOperation {
  id: string
  label: string
  description: string
  path: string
  template: JsonObject
  timeoutMs?: number
  note?: string
}

export async function executeResearchOperation(
  operation: ResearchOperation,
  payload: JsonObject,
  signal?: AbortSignal,
): Promise<JsonObject> {
  const response = await apiPost<unknown>(operation.path, payload, {
    signal,
    timeoutMs: operation.timeoutMs ?? 120_000,
  })
  if (!isJsonObject(response)) {
    throw new ApiError('The backend returned an unexpected response shape.')
  }
  return response
}
