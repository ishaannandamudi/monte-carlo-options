interface LoadingStateProps {
  label?: string
  compact?: boolean
}

export function LoadingState({ label = 'Loading', compact = false }: LoadingStateProps) {
  return (
    <div className={`loading-state${compact ? ' loading-state--compact' : ''}`} role="status">
      <span className="spinner" aria-hidden="true" />
      <span>{label}</span>
    </div>
  )
}
