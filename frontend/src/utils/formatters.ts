const locale = 'en-US'

function finiteOrFallback(value: number | null | undefined, fallback: string): number | string {
  return value !== null && value !== undefined && Number.isFinite(value) ? value : fallback
}

export function formatCurrency(
  value: number | null | undefined,
  currency = 'USD',
  fallback = '—',
): string {
  const normalized = finiteOrFallback(value, fallback)
  if (typeof normalized === 'string') return normalized
  return new Intl.NumberFormat(locale, {
    style: 'currency',
    currency,
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  }).format(normalized)
}

export function formatPercent(
  value: number | null | undefined,
  fractionDigits = 2,
  fallback = '—',
): string {
  const normalized = finiteOrFallback(value, fallback)
  if (typeof normalized === 'string') return normalized
  return new Intl.NumberFormat(locale, {
    style: 'percent',
    minimumFractionDigits: fractionDigits,
    maximumFractionDigits: fractionDigits,
  }).format(normalized)
}

export function formatVolatility(
  value: number | null | undefined,
  fractionDigits = 2,
  fallback = '—',
): string {
  return formatPercent(value, fractionDigits, fallback)
}

export function formatProbability(
  value: number | null | undefined,
  fractionDigits = 2,
  fallback = '—',
): string {
  return formatPercent(value, fractionDigits, fallback)
}

export function formatRuntime(valueMs: number | null | undefined, fallback = '—'): string {
  const normalized = finiteOrFallback(valueMs, fallback)
  if (typeof normalized === 'string') return normalized
  if (Math.abs(normalized) < 1) return `${(normalized * 1_000).toFixed(0)} µs`
  if (Math.abs(normalized) < 1_000) return `${normalized.toFixed(normalized < 10 ? 2 : 1)} ms`
  return `${(normalized / 1_000).toFixed(2)} s`
}

export function formatScientific(
  value: number | null | undefined,
  fractionDigits = 3,
  fallback = '—',
): string {
  const normalized = finiteOrFallback(value, fallback)
  return typeof normalized === 'string' ? normalized : normalized.toExponential(fractionDigits)
}

export function formatDecimal(
  value: number | null | undefined,
  options: { minimumFractionDigits?: number; maximumFractionDigits?: number } = {},
  fallback = '—',
): string {
  const normalized = finiteOrFallback(value, fallback)
  if (typeof normalized === 'string') return normalized
  return new Intl.NumberFormat(locale, {
    minimumFractionDigits: options.minimumFractionDigits ?? 0,
    maximumFractionDigits: options.maximumFractionDigits ?? 4,
  }).format(normalized)
}

export function formatCount(value: number | null | undefined, fallback = '—'): string {
  const normalized = finiteOrFallback(value, fallback)
  return typeof normalized === 'string'
    ? normalized
    : new Intl.NumberFormat(locale, { maximumFractionDigits: 0 }).format(normalized)
}

export function formatBasisPoints(value: number | null | undefined, fallback = '—'): string {
  const normalized = finiteOrFallback(value, fallback)
  return typeof normalized === 'string' ? normalized : `${formatDecimal(normalized * 10_000, { maximumFractionDigits: 2 })} bp`
}

export function formatGreek(value: number | null | undefined, fallback = '—'): string {
  const normalized = finiteOrFallback(value, fallback)
  if (typeof normalized === 'string') return normalized
  return Math.abs(normalized) > 0 && Math.abs(normalized) < 1e-5
    ? formatScientific(normalized, 3, fallback)
    : formatDecimal(normalized, { maximumFractionDigits: 6 }, fallback)
}
