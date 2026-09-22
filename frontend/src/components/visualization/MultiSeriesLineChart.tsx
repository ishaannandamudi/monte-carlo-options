import { formatDecimal, formatVolatility } from '../../utils/formatters'

export interface ChartSeries {
  label: string
  points: readonly { x: number; y: number; selected?: boolean }[]
}

interface MultiSeriesLineChartProps {
  series: readonly ChartSeries[]
  title: string
  xLabel: string
  yLabel: string
}

export function MultiSeriesLineChart({ series, title, xLabel, yLabel }: MultiSeriesLineChartProps) {
  const allPoints = series.flatMap((item) => item.points)
  if (allPoints.length < 2) return <p className="fine-print">Not enough valid IV points to draw the smile.</p>
  const width = 720
  const height = 280
  const padding = 48
  const minX = Math.min(...allPoints.map((point) => point.x))
  const maxX = Math.max(...allPoints.map((point) => point.x))
  const minY = Math.min(...allPoints.map((point) => point.y))
  const maxY = Math.max(...allPoints.map((point) => point.y))
  const xSpan = maxX - minX || 1
  const ySpan = maxY - minY || 1
  const coordinates = (x: number, y: number) => ({
    x: padding + ((x - minX) / xSpan) * (width - padding * 2),
    y: height - padding - ((y - minY) / ySpan) * (height - padding * 2),
  })

  return (
    <figure className="chart-card analyzer-chart">
      <figcaption>{title}</figcaption>
      <div className="chart-legend" aria-label="Chart legend">
        {series.map((item, index) => <span className={`chart-legend__item chart-series-${String(index)}`} key={item.label}>{item.label}</span>)}
      </div>
      <svg aria-label={`${title}. ${xLabel} by ${yLabel}.`} role="img" viewBox={`0 0 ${String(width)} ${String(height)}`}>
        <line className="chart-axis" x1={padding} x2={padding} y1={padding} y2={height - padding} />
        <line className="chart-axis" x1={padding} x2={width - padding} y1={height - padding} y2={height - padding} />
        {series.map((item, seriesIndex) => {
          const points = [...item.points].sort((left, right) => left.x - right.x)
          const path = points.map((point, index) => {
            const location = coordinates(point.x, point.y)
            return `${index === 0 ? 'M' : 'L'}${String(location.x)},${String(location.y)}`
          }).join(' ')
          return <path className={`chart-line chart-series-${String(seriesIndex)}`} d={path} key={item.label} />
        })}
        {series.flatMap((item, seriesIndex) => item.points.map((point, pointIndex) => {
          const location = coordinates(point.x, point.y)
          return (
            <circle className={`chart-point chart-series-${String(seriesIndex)}${point.selected ? ' chart-point--selected' : ''}`} cx={location.x} cy={location.y} key={`${item.label}-${String(point.x)}-${String(pointIndex)}`} r={point.selected ? 6 : 4}>
              <title>{`${item.label}, strike ${formatDecimal(point.x)}: ${formatVolatility(point.y)}`}</title>
            </circle>
          )
        }))}
        <text className="chart-label" x={width / 2} y={height - 7}>{xLabel}</text>
        <text className="chart-label" x={7} y={18}>{yLabel}</text>
      </svg>
      <table className="sr-only">
        <caption>{title} data</caption>
        <thead><tr><th>Series</th><th>{xLabel}</th><th>{yLabel}</th></tr></thead>
        <tbody>{series.flatMap((item) => item.points.map((point, pointIndex) => <tr key={`${item.label}-${String(point.x)}-${String(pointIndex)}`}><td>{item.label}</td><td>{point.x}</td><td>{point.y}</td></tr>))}</tbody>
      </table>
    </figure>
  )
}
