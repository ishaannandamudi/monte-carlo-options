import { describe, expect, it } from 'vitest'

import {
  formatCurrency,
  formatBasisPoints,
  formatCount,
  formatDecimal,
  formatPercent,
  formatProbability,
  formatRuntime,
  formatGreek,
  formatScientific,
  formatVolatility,
} from './formatters'

describe('number formatters', () => {
  it('formats model outputs without floating-point noise', () => {
    expect(formatCurrency(1234.567)).toBe('$1,234.57')
    expect(formatPercent(0.123456)).toBe('12.35%')
    expect(formatVolatility(0.2)).toBe('20.00%')
    expect(formatProbability(0.875)).toBe('87.50%')
    expect(formatDecimal(1.234567)).toBe('1.2346')
  })

  it('selects readable runtime units', () => {
    expect(formatRuntime(0.25)).toBe('250 µs')
    expect(formatRuntime(8.125)).toBe('8.13 ms')
    expect(formatRuntime(1500)).toBe('1.50 s')
  })

  it('formats scientific values and rejects non-finite display values', () => {
    expect(formatScientific(0.000012345)).toBe('1.234e-5')
    expect(formatDecimal(Number.NaN)).toBe('—')
    expect(formatCurrency(Number.POSITIVE_INFINITY)).toBe('—')
  })

  it('formats counts, basis points, and Greek values', () => {
    expect(formatCount(1234567)).toBe('1,234,567')
    expect(formatBasisPoints(0.0025)).toBe('25 bp')
    expect(formatGreek(0.51234567)).toBe('0.512346')
  })
})
