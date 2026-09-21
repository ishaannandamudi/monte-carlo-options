import { apiConfig } from './config'

export interface FastApiValidationIssue {
  location: string
  message: string
  type?: string
}

export class ApiError extends Error {
  readonly status: number | null
  readonly issues: readonly FastApiValidationIssue[]
  readonly isTimeout: boolean

  constructor(
    message: string,
    options: {
      status?: number | null
      issues?: readonly FastApiValidationIssue[]
      isTimeout?: boolean
      cause?: unknown
    } = {},
  ) {
    super(message, { cause: options.cause })
    this.name = 'ApiError'
    this.status = options.status ?? null
    this.issues = options.issues ?? []
    this.isTimeout = options.isTimeout ?? false
  }
}

export interface RequestOptions extends Omit<RequestInit, 'body'> {
  body?: unknown
  timeoutMs?: number
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null
}

function validationIssues(detail: unknown): FastApiValidationIssue[] {
  if (!Array.isArray(detail)) return []
  return detail.flatMap((item): FastApiValidationIssue[] => {
    if (!isRecord(item) || typeof item.msg !== 'string') return []
    const rawLocation = Array.isArray(item.loc) ? item.loc : []
    return [
      {
        location: rawLocation.map(String).join('.'),
        message: item.msg,
        ...(typeof item.type === 'string' ? { type: item.type } : {}),
      },
    ]
  })
}

function responseMessage(payload: unknown, fallback: string): string {
  if (!isRecord(payload)) return fallback
  if (typeof payload.detail === 'string') return payload.detail
  const issues = validationIssues(payload.detail)
  if (issues.length > 0) return issues.map((issue) => issue.message).join('; ')
  if (typeof payload.message === 'string') return payload.message
  return fallback
}

function mergeAbortSignals(
  callerSignal: AbortSignal | null | undefined,
  timeoutMs: number,
): { signal: AbortSignal; cleanup: () => void; timeoutController: AbortController } {
  const timeoutController = new AbortController()
  const requestController = new AbortController()
  const timeout = window.setTimeout(() => timeoutController.abort(), timeoutMs)
  const abortRequest = (): void => requestController.abort()

  timeoutController.signal.addEventListener('abort', abortRequest, { once: true })
  callerSignal?.addEventListener('abort', abortRequest, { once: true })
  if (callerSignal?.aborted) requestController.abort()

  return {
    signal: requestController.signal,
    timeoutController,
    cleanup: () => {
      window.clearTimeout(timeout)
      timeoutController.signal.removeEventListener('abort', abortRequest)
      callerSignal?.removeEventListener('abort', abortRequest)
    },
  }
}

export async function apiRequest<T>(path: string, options: RequestOptions = {}): Promise<T> {
  const { body, timeoutMs = apiConfig.timeoutMs, signal: callerSignal, ...requestInit } = options
  const { signal, cleanup, timeoutController } = mergeAbortSignals(callerSignal, timeoutMs)
  const headers = new Headers(requestInit.headers)
  if (body !== undefined && !headers.has('Content-Type')) {
    headers.set('Content-Type', 'application/json')
  }

  try {
    const response = await fetch(`${apiConfig.baseUrl}${path}`, {
      ...requestInit,
      body: body === undefined ? undefined : JSON.stringify(body),
      headers,
      signal,
    })
    const contentType = response.headers.get('content-type') ?? ''
    const payload: unknown =
      response.status === 204
        ? null
        : contentType.includes('application/json')
          ? await response.json()
          : await response.text()

    if (!response.ok) {
      const issues = isRecord(payload) ? validationIssues(payload.detail) : []
      throw new ApiError(
        responseMessage(payload, `Request failed with status ${String(response.status)}`),
        { status: response.status, issues },
      )
    }
    return payload as T
  } catch (error: unknown) {
    if (error instanceof ApiError) throw error
    if (timeoutController.signal.aborted) {
      throw new ApiError('The backend did not respond before the request timed out.', {
        isTimeout: true,
        cause: error,
      })
    }
    if (callerSignal?.aborted) {
      throw new ApiError('The request was cancelled.', { cause: error })
    }
    throw new ApiError('Unable to connect to the quantitative research backend.', {
      cause: error,
    })
  } finally {
    cleanup()
  }
}

export function apiGet<T>(path: string, signal?: AbortSignal): Promise<T> {
  return apiRequest<T>(path, { method: 'GET', signal })
}

export function apiPost<T>(
  path: string,
  body: unknown,
  options: Omit<RequestOptions, 'body' | 'method'> = {},
): Promise<T> {
  return apiRequest<T>(path, { ...options, body, method: 'POST' })
}
