import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'

import { ApiError } from '../../api/client'
import { ErrorPanel } from './ErrorPanel'
import { LoadingState } from './LoadingState'
import { MetricCard } from './MetricCard'

describe('shared state components', () => {
  it('renders an accessible loading state', () => {
    render(<LoadingState label="Loading calibration inputs" />)

    expect(screen.getByRole('status')).toHaveTextContent('Loading calibration inputs')
  })

  it('renders normalized validation errors without stack traces', () => {
    const error = new ApiError('Request validation failed', {
      status: 422,
      issues: [{ location: 'body.spot', message: 'Input should be greater than 0' }],
    })
    render(<ErrorPanel error={error} />)

    const panel = screen.getByRole('alert')
    expect(panel).toHaveTextContent('Request validation failed')
    expect(panel).toHaveTextContent('body.spot: Input should be greater than 0')
    expect(panel).not.toHaveTextContent('ApiError:')
  })

  it('renders a labeled metric with status styling', () => {
    render(<MetricCard detail="FastAPI" label="Backend" status="positive" value="Connected" />)

    expect(screen.getByText('Backend')).toBeInTheDocument()
    expect(screen.getByText('Connected')).toBeInTheDocument()
    expect(screen.getByText('FastAPI').closest('.metric-card')).toHaveClass('metric-card--positive')
  })
})
