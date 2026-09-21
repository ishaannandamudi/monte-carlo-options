import { formatDecimal } from '../../utils/formatters'

export interface HeatmapCell {
  x: string
  y: string
  value: number
}

interface HeatmapProps {
  cells: readonly HeatmapCell[]
  title: string
  valueLabel: string
}

export function Heatmap({ cells, title, valueLabel }: HeatmapProps) {
  if (cells.length === 0) return null
  const values = cells.map((cell) => cell.value)
  const min = Math.min(...values)
  const max = Math.max(...values)
  const span = max - min || 1
  return (
    <figure className="chart-card">
      <figcaption>{title}</figcaption>
      <div className="heatmap" role="img" aria-label={`${title}; color encodes ${valueLabel}`}>
        {cells.map((cell) => {
          const intensity = (cell.value - min) / span
          return (
            <div
              className="heatmap__cell"
              key={`${cell.x}-${cell.y}`}
              style={{
                backgroundColor: `rgb(84 210 194 / ${String(0.06 + intensity * 0.3)})`,
                borderColor: `rgb(84 210 194 / ${String(0.2 + intensity * 0.45)})`,
              }}
              title={`${cell.x}, ${cell.y}: ${formatDecimal(cell.value)}`}
            >
              <span>{formatDecimal(cell.value, { maximumFractionDigits: 3 })}</span>
              <small>{cell.x} · {cell.y}</small>
            </div>
          )
        })}
      </div>
      <table className="sr-only"><caption>{title} data</caption><thead><tr><th>X</th><th>Y</th><th>{valueLabel}</th></tr></thead><tbody>{cells.map((cell) => <tr key={`${cell.x}-${cell.y}`}><td>{cell.x}</td><td>{cell.y}</td><td>{cell.value}</td></tr>)}</tbody></table>
    </figure>
  )
}
