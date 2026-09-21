const fallbackApiBaseUrl = 'http://localhost:8000'

function normalizeBaseUrl(value: string | undefined): string {
  const candidate = value?.trim() || fallbackApiBaseUrl
  return candidate.replace(/\/+$/, '')
}

export const apiConfig = Object.freeze({
  baseUrl: normalizeBaseUrl(import.meta.env.VITE_API_BASE_URL),
  timeoutMs: 8_000,
})
