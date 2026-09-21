import { useEffect, useRef, useState } from 'react'

import { executeResearchOperation, type ResearchOperation } from '../../api/research'
import { parseJsonObject, prettyJson, type JsonObject } from '../../api/types'
import { Alert, Button, Card, ErrorPanel, JsonEditor, LoadingState, Select } from '../ui'
import { ResultPanel } from './ResultPanel'

interface OperationWorkbenchProps {
  operations: readonly ResearchOperation[]
  initialOperationId?: string
  onResult?: (result: JsonObject | null) => void
  title?: string
}

export function OperationWorkbench({ operations, initialOperationId, onResult, title = 'Research request' }: OperationWorkbenchProps) {
  const initial = operations.find((item) => item.id === initialOperationId) ?? operations[0]
  if (!initial) throw new Error('OperationWorkbench requires at least one operation.')
  const [operationId, setOperationId] = useState(initial.id)
  const [source, setSource] = useState(prettyJson(initial.template))
  const [result, setResult] = useState<JsonObject | null>(null)
  const [error, setError] = useState<unknown>(null)
  const [pending, setPending] = useState(false)
  const controllerRef = useRef<AbortController | null>(null)
  const operation = operations.find((item) => item.id === operationId) ?? initial

  useEffect(() => () => controllerRef.current?.abort(), [])

  const selectOperation = (nextId: string): void => {
    const next = operations.find((item) => item.id === nextId)
    if (!next) return
    controllerRef.current?.abort()
    setOperationId(next.id)
    setSource(prettyJson(next.template))
    setResult(null)
    setError(null)
    onResult?.(null)
  }

  const run = async (): Promise<void> => {
    setError(null)
    setResult(null)
    onResult?.(null)
    let payload: JsonObject
    try {
      payload = parseJsonObject(source)
    } catch (parseError: unknown) {
      setError(parseError)
      return
    }
    const controller = new AbortController()
    controllerRef.current = controller
    setPending(true)
    try {
      const response = await executeResearchOperation(operation, payload, controller.signal)
      setResult(response)
      onResult?.(response)
    } catch (requestError: unknown) {
      setError(requestError)
    } finally {
      if (controllerRef.current === controller) controllerRef.current = null
      setPending(false)
    }
  }

  return (
    <div className="workbench-grid">
      <Card className="workbench-form">
        <h2>{title}</h2>
        <Select label="Workflow" onChange={(event) => selectOperation(event.target.value)} options={operations.map((item) => ({ label: item.label, value: item.id }))} value={operation.id} />
        <Alert title={operation.label} variant="info">{operation.description}</Alert>
        {operation.note ? <p className="fine-print">{operation.note}</p> : null}
        <JsonEditor description="Validated by the backend. Edit the bounded template or paste an exported research request." label="Request JSON" onChange={setSource} value={source} />
        <div className="button-row">
          <Button disabled={pending} onClick={() => void run()}>{pending ? 'Running…' : 'Run workflow'}</Button>
          {pending ? <Button onClick={() => controllerRef.current?.abort()} variant="secondary">Cancel</Button> : null}
          <Button onClick={() => setSource(prettyJson(operation.template))} variant="ghost">Reset template</Button>
        </div>
      </Card>
      <div className="workbench-result">
        {pending ? <LoadingState label={`Running ${operation.label}`} /> : null}
        {error ? <ErrorPanel error={error} onRetry={() => void run()} /> : null}
        {result ? <ResultPanel exportName={`${operation.id}-result.json`} result={result} title={`${operation.label} result`} /> : null}
        {!pending && !error && !result ? <Card className="result-awaiting"><span className="eyebrow">Ready</span><h2>Configure and run</h2><p>Results, warnings, partial statuses, charts, and full JSON will appear here.</p></Card> : null}
      </div>
    </div>
  )
}
