import { screen, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { describe, expect, it } from 'vitest'

import { installApiMock } from '../test/mock-api'
import { jsonResponse, renderApp } from '../test/test-utils'

const runs = [
  { run_id: 'alpha-run-0001', experiment_id: 'exp-1', status: 'COMPLETED', created_at: '2026-01-02T00:00:00Z', duration_ms: 12.5, experiment_type: 'PRICING', input_signature: 'a'.repeat(64), definition: { name: 'price' }, config_snapshot: {}, environment_snapshot: {}, tags: ['bsm'] },
  { run_id: 'beta-run-00002', experiment_id: 'exp-2', status: 'FAILED', created_at: '2026-01-01T00:00:00Z', duration_ms: 5, experiment_type: 'QMC', input_signature: 'b'.repeat(64), definition: { name: 'qmc' }, config_snapshot: {}, environment_snapshot: {}, tags: ['sobol'] },
]

function experimentMock() {
  return installApiMock((url) => {
    if (url.includes('/api/experiments?')) return jsonResponse({ items: runs, page: url.includes('page=2') ? 2 : 1, page_size: 10, total: 12 })
    if (url.endsWith('/api/experiments/alpha-run-0001')) return jsonResponse({ ...runs[0], metrics: [{ name: 'price', value: 10.45, unit: 'USD' }], artifacts: [{ artifact_id: 'artifact-1', artifact_type: 'JSON', path: 'result.json', size_bytes: 120, created_at: '2026-01-02T00:00:01Z' }], result_summary: { price: 10.45 } })
    if (url.endsWith('/api/experiments/compare')) return jsonResponse({ baseline_run_id: 'alpha-run-0001', comparisons: [{ candidate_run_id: 'beta-run-00002', flags: ['RESULT_CHANGED'] }] })
    if (url.endsWith('/api/experiments/alpha-run-0001/reproduce')) return jsonResponse({ original_run_id: 'alpha-run-0001', replay_run_id: 'run-replay', signature_match: true, result_match: true, runtime_difference_ms: 0.4 })
    return undefined
  })
}

describe('experiment workflows', () => {
  it('filters and paginates a deterministic run list', async () => {
    const mock = experimentMock()
    const user = userEvent.setup()
    renderApp('/experiments')

    expect(await screen.findByText('alpha-run-00…')).toBeInTheDocument()
    await user.selectOptions(screen.getByLabelText('Experiment type'), 'PRICING')
    await user.type(screen.getByLabelText('Tag filter'), 'bsm')
    await user.click(screen.getByRole('button', { name: 'Next' }))

    await waitFor(() => expect(mock).toHaveBeenCalledWith(expect.stringContaining('page=2'), expect.anything()))
    expect(mock).toHaveBeenCalledWith(expect.stringContaining('experiment_type=PRICING'), expect.anything())
    expect(mock).toHaveBeenCalledWith(expect.stringContaining('tag=bsm'), expect.anything())
  })

  it('opens run details and reproduces a completed run', async () => {
    experimentMock()
    const user = userEvent.setup()
    renderApp('/experiments')

    await user.click(await screen.findByRole('button', { name: 'alpha-run-00…' }))
    expect(await screen.findByText('price')).toBeInTheDocument()
    expect(screen.getByText('result.json')).toBeInTheDocument()
    await user.click(screen.getByRole('button', { name: 'Reproduce' }))

    expect(await screen.findByText(/"signature_match": true/)).toBeInTheDocument()
    expect(screen.getByText(/"result_match": true/)).toBeInTheDocument()
  })

  it('compares selected runs and preserves diagnostic flags', async () => {
    experimentMock()
    const user = userEvent.setup()
    renderApp('/experiments')

    await user.click(await screen.findByLabelText('Select run alpha-run-0001 for comparison'))
    await user.click(screen.getByLabelText('Select run beta-run-00002 for comparison'))
    await user.click(screen.getByRole('button', { name: 'Compare selected (2)' }))

    expect(await screen.findByText(/RESULT_CHANGED/)).toBeInTheDocument()
  })
})
