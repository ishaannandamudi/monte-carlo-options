import { formatDecimal } from '../../utils/formatters'

export interface ChartPoint {
  x: number
  y: number
  label?: string
}

interface LineChartProps {
  points: readonly ChartPoint[]
  title: string
  xLabel: string
  yLabel: string
}

export function LineChart({ points, title, xLabel, yLabel }: LineChartProps) {
  if (points.length < 2) return null
  const width = 720
  const height = 260
  const padding = 44
  const xs = points.map((point) => point.x)
  const ys = points.map((point) => point.y)
  const minX = Math.min(...xs)
  const maxX = Math.max(...xs)
  const minY = Math.min(...ys)
  const maxY = Math.max(...ys)
  const xSpan = maxX - minX || 1
  const ySpan = maxY - minY || 1
  const coordinates = points.map((point) => ({
    ...point,
    px: padding + ((point.x - minX) / xSpan) * (width - padding * 2),
    py: height - padding - ((point.y - minY) / ySpan) * (height - padding * 2),
  }))
  const path = coordinates.map((point, index) => `${index === 0 ? 'M' : 'L'}${String(point.px)},${String(point.py)}`).join(' ')

  return (
    <figure className="chart-card">
      <figcaption>{title}</figcaption>
      <svg aria-label={`${title}. ${xLabel} by ${yLabel}.`} role="img" viewBox={`0 0 ${String(width)} ${String(height)}`}>
        <line className="chart-axis" x1={padding} x2={padding} y1={padding} y2={height - padding} />
        <line className="chart-axis" x1={padding} x2={width - padding} y1={height - padding} y2={height - padding} />
        <path className="chart-line" d={path} />
        {coordinates.map((point) => (
          <circle className="chart-point" cx={point.px} cy={point.py} key={`${String(point.x)}-${String(point.y)}`} r="4">
            <title>{`${point.label ?? formatDecimal(point.x)}: ${formatDecimal(point.y)}`}</title>
          </circle>
        ))}
        <text className="chart-label" x={width / 2} y={height - 7}>{xLabel}</text>
        <text className="chart-label" x={7} y={18}>{yLabel}</text>
        <text className="chart-tick" x={padding} y={height - padding + 18}>{formatDecimal(minX)}</text>
        <text className="chart-tick" textAnchor="end" x={width - padding} y={height - padding + 18}>{formatDecimal(maxX)}</text>
        <text className="chart-tick" x={padding + 4} y={padding + 10}>{formatDecimal(maxY)}</text>
        <text className="chart-tick" x={padding + 4} y={height - padding - 7}>{formatDecimal(minY)}</text>
      </svg>
      <table className="sr-only">
        <caption>{title} data</caption>
        <thead><tr><th>{xLabel}</th><th>{yLabel}</th></tr></thead>
        <tbody>{points.map((point) => <tr key={`${String(point.x)}-${String(point.y)}`}><td>{point.x}</td><td>{point.y}</td></tr>)}</tbody>
      </table>
    </figure>
  )
}
