import { ApiError } from '../../api/client'
import { Button } from './Button'

interface ErrorPanelProps {
  error: unknown
  title?: string
  onRetry?: () => void
}

function errorMessage(error: unknown): string {
  if (error instanceof ApiError || error instanceof Error) return error.message
  return 'An unexpected error interrupted this request.'
}

export function ErrorPanel({ error, title = 'Request unavailable', onRetry }: ErrorPanelProps) {
  const issues = error instanceof ApiError ? error.issues : []
  return (
    <div className="error-panel" role="alert">
      <div className="error-panel__icon" aria-hidden="true">
        !
      </div>
      <div className="error-panel__body">
        <strong>{title}</strong>
        <p>{errorMessage(error)}</p>
        {issues.length > 0 ? (
          <ul>
            {issues.map((issue) => (
              <li key={`${issue.location}-${issue.message}`}>
                {issue.location ? `${issue.location}: ` : ''}
                {issue.message}
              </li>
            ))}
          </ul>
        ) : null}
        {onRetry ? (
          <Button onClick={onRetry} size="small" variant="secondary">
            Retry connection
          </Button>
        ) : null}
      </div>
    </div>
  )
}
