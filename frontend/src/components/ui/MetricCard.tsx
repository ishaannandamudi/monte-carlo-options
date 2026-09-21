import type { ReactNode } from 'react'

import { Card } from './Card'

interface MetricCardProps {
  label: string
  value: ReactNode
  detail?: ReactNode
  status?: 'neutral' | 'positive' | 'warning' | 'negative'
}

export function MetricCard({ label, value, detail, status = 'neutral' }: MetricCardProps) {
  return (
    <Card className={`metric-card metric-card--${status}`}>
      <span className="metric-card__label">{label}</span>
      <strong className="metric-card__value">{value}</strong>
      {detail ? <span className="metric-card__detail">{detail}</span> : null}
    </Card>
  )
}
