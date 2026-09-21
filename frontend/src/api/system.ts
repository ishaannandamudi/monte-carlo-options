import { queryOptions } from '@tanstack/react-query'

import { apiGet } from './client'

export interface HealthResponse {
  status: 'ok'
  service: string
}

export interface VersionResponse {
  service: string
  version: string
}

export const systemQueryKeys = {
  all: ['system'] as const,
  health: () => [...systemQueryKeys.all, 'health'] as const,
  version: () => [...systemQueryKeys.all, 'version'] as const,
}

export function healthQueryOptions() {
  return queryOptions({
    queryKey: systemQueryKeys.health(),
    queryFn: ({ signal }) => apiGet<HealthResponse>('/health', signal),
    refetchInterval: 30_000,
  })
}

export function versionQueryOptions() {
  return queryOptions({
    queryKey: systemQueryKeys.version(),
    queryFn: ({ signal }) => apiGet<VersionResponse>('/version', signal),
    staleTime: Number.POSITIVE_INFINITY,
  })
}
