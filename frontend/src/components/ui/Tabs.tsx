export interface TabOption {
  id: string
  label: string
}

interface TabsProps {
  label: string
  options: readonly TabOption[]
  value: string
  onChange: (value: string) => void
}

export function Tabs({ label, options, value, onChange }: TabsProps) {
  return (
    <div aria-label={label} className="tabs" role="tablist">
      {options.map((option) => (
        <button
          aria-selected={option.id === value}
          className={`tabs__tab${option.id === value ? ' tabs__tab--active' : ''}`}
          key={option.id}
          onClick={() => onChange(option.id)}
          role="tab"
          type="button"
        >
          {option.label}
        </button>
      ))}
    </div>
  )
}
