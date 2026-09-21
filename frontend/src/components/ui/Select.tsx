import { useId, type SelectHTMLAttributes } from 'react'

export interface SelectOption {
  label: string
  value: string
  disabled?: boolean
}

interface SelectProps extends SelectHTMLAttributes<HTMLSelectElement> {
  label: string
  options: readonly SelectOption[]
  hint?: string
  error?: string
}

export function Select({ label, options, hint, error, id, className = '', ...props }: SelectProps) {
  const generatedId = useId()
  const selectId = id ?? generatedId
  const descriptionId = error || hint ? `${selectId}-description` : undefined
  return (
    <label className="field" htmlFor={selectId}>
      <span className="field__label">{label}</span>
      <select
        {...props}
        aria-describedby={descriptionId}
        aria-invalid={Boolean(error)}
        className={`select ${error ? 'select--error' : ''} ${className}`.trim()}
        id={selectId}
      >
        {options.map((option) => (
          <option disabled={option.disabled} key={option.value} value={option.value}>
            {option.label}
          </option>
        ))}
      </select>
      {error ? (
        <span className="field__error" id={descriptionId}>
          {error}
        </span>
      ) : hint ? (
        <span className="field__hint" id={descriptionId}>
          {hint}
        </span>
      ) : null}
    </label>
  )
}
