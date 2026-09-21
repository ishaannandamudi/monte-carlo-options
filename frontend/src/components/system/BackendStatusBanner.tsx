import { useQuery } from '@tanstack/react-query'

import { healthQueryOptions } from '../../api/system'
import { Badge } from '../ui'

export function BackendStatusBanner() {
  const health = useQuery(healthQueryOptions())

  if (health.isPending) {
    return (
      <div className="backend-status backend-status--checking" role="status">
        <span className="status-dot" aria-hidden="true" />
        Checking backend
      </div>
    )
  }

  if (health.isError) {
    return (
      <div className="backend-status backend-status--offline" role="alert">
        <span className="status-dot" aria-hidden="true" />
        Backend disconnected
      </div>
    )
  }

  return (
    <div className="backend-status backend-status--online" role="status">
      <span className="status-dot" aria-hidden="true" />
      Backend connected
      <Badge variant="success">{health.data.status}</Badge>
    </div>
  )
}
