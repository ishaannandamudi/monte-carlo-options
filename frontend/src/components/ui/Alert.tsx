import type { ReactNode } from 'react'

interface AlertProps {
  title?: string
  children: ReactNode
  variant?: 'info' | 'success' | 'warning' | 'danger'
}

export function Alert({ title, children, variant = 'info' }: AlertProps) {
  return (
    <div className={`alert alert--${variant}`} role={variant === 'danger' ? 'alert' : 'status'}>
      <span className="alert__marker" aria-hidden="true" />
      <div>
        {title ? <strong className="alert__title">{title}</strong> : null}
        <div className="alert__content">{children}</div>
      </div>
    </div>
  )
}
