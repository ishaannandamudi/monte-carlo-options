import { useId } from 'react'

interface JsonEditorProps {
  label: string
  value: string
  onChange: (value: string) => void
  description?: string
  rows?: number
}

export function JsonEditor({ label, value, onChange, description, rows = 18 }: JsonEditorProps) {
  const id = useId()
  const descriptionId = description ? `${id}-description` : undefined
  return (
    <label className="json-editor" htmlFor={id}>
      <span className="field__label">{label}</span>
      {description ? (
        <span className="field__hint" id={descriptionId}>
          {description}
        </span>
      ) : null}
      <textarea
        aria-describedby={descriptionId}
        className="json-editor__input"
        id={id}
        onChange={(event) => onChange(event.target.value)}
        rows={rows}
        spellCheck={false}
        value={value}
      />
    </label>
  )
}
