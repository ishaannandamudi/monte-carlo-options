import { isJsonObject, type JsonObject, type JsonValue } from '../../api/types'
import { Heatmap, type HeatmapCell } from './Heatmap'
import { LineChart, type ChartPoint } from './LineChart'

const xCandidates = ['strike', 'log_moneyness', 'maturity_years', 'num_simulations', 'num_samples', 'num_states', 'observation_index', 'steps', 'value', 'horizon']
const yCandidates = ['implied_volatility', 'heston_implied_volatility', 'model_implied_volatility', 'raw_residual', 'price', 'estimate', 'absolute_error', 'standard_error', 'full_repricing_pnl', 'pnl', 'portfolio_value', 'rmse', 'objective_value', 'bic', 'aic']

function objects(value: JsonValue | undefined): JsonObject[] {
  return Array.isArray(value) ? value.filter(isJsonObject) : []
}

function findNumericKey(rows: readonly JsonObject[], candidates: readonly string[]): string | null {
  return candidates.find((key) => rows.some((row) => typeof row[key] === 'number')) ?? null
}

function lineData(result: JsonObject): { points: ChartPoint[]; xKey: string; yKey: string } | null {
  const probabilityRows = objects(result.probability_series)
  if (probabilityRows.length >= 2) {
    const points = probabilityRows.flatMap((row): ChartPoint[] => {
      const x = row.observation_index
      const probabilities = row.filtered_probabilities
      const y = Array.isArray(probabilities) ? probabilities[0] : undefined
      return typeof x === 'number' && typeof y === 'number' ? [{ x, y }] : []
    })
    if (points.length >= 2) return { points, xKey: 'observation_index', yKey: 'filtered state 0 probability' }
  }
  const sources = [result.points, result.results, result.convergence_points, result.smile_points, result.residuals]
  for (const source of sources) {
    const rows = objects(source)
    if (rows.length < 2) continue
    const xKey = findNumericKey(rows, xCandidates)
    const yKey = findNumericKey(rows, yCandidates)
    if (!xKey || !yKey) continue
    const points = rows.flatMap((row): ChartPoint[] => {
      const x = row[xKey]
      const y = row[yKey]
      return typeof x === 'number' && typeof y === 'number' ? [{ x, y }] : []
    })
    if (points.length >= 2) return { points, xKey, yKey }
  }
  return null
}

function heatmapData(result: JsonObject): HeatmapCell[] {
  const surface = isJsonObject(result.surface) ? result.surface : null
  const smileRows = surface ? objects(surface.smiles) : []
  const nestedSurfacePoints = smileRows.flatMap((smile) => objects(smile.points))
  const pointRows = objects(result.points)
  const surfacePointRows = pointRows.some((row) => typeof row.maturity_years === 'number') ? pointRows : []
  const rows = objects(result.cells ?? result.grid_points ?? result.surface_points).concat(nestedSurfacePoints, surfacePointRows)
  return rows.flatMap((row): HeatmapCell[] => {
    const coordinates = isJsonObject(row.coordinates) ? Object.entries(row.coordinates) : []
    const x = row.strike ?? row.log_moneyness ?? row.spot_shift ?? row.x ?? coordinates[0]?.[1]
    const y = row.maturity_years ?? row.volatility_shift ?? row.y ?? coordinates[1]?.[1]
    const value = row.full_repricing_pnl ?? row.pnl ?? row.heston_implied_volatility ?? row.implied_volatility ?? row.implied_volatility_residual ?? row.residual ?? row.value
    return (typeof x === 'number' || typeof x === 'string') &&
      (typeof y === 'number' || typeof y === 'string') && typeof value === 'number'
      ? [{ x: String(x), y: String(y), value }]
      : []
  })
}

export function AutoChart({ result }: { result: JsonObject }) {
  const cells = heatmapData(result)
  if (cells.length > 1) return <Heatmap cells={cells} title="Result surface" valueLabel="Result" />
  const data = lineData(result)
  if (!data) return null
  return <LineChart points={data.points} title="Result series" xLabel={data.xKey.replaceAll('_', ' ')} yLabel={data.yKey.replaceAll('_', ' ')} />
}
