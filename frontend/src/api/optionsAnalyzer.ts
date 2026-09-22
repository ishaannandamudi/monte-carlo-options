import { apiPost } from './client'

export type AnalysisOptionType = 'CALL' | 'PUT' | 'BOTH'
export type OptionType = 'CALL' | 'PUT'
export type FairValueVolatilitySource =
  | 'SMILE_INTERPOLATED'
  | 'ATM_REFERENCE'
  | 'CONTRACT_IV_FALLBACK'
  | 'UNAVAILABLE'

export interface OptionsAnalysisRequest {
  symbol: string
  expiration?: string | null
  option_type?: AnalysisOptionType
  max_contracts?: number
  strike_window_pct?: number
  target_expiration_days?: number
  risk_free_rate?: number | null
  dividend_yield?: number | null
  tree_steps?: number
  max_quote_age_seconds?: number
  allow_last_price_fallback?: boolean
}

export interface AnalyzedOptionContract {
  provider_symbol: string
  option_type: OptionType
  strike: number
  expiration: string
  exercise_style: 'AMERICAN'
  provider_exercise_style: string
  bid: number | null
  ask: number | null
  market_midpoint: number | null
  market_price_source: 'MIDPOINT' | 'LAST' | null
  last: number | null
  bid_size: number | null
  ask_size: number | null
  volume: number | null
  open_interest: number | null
  quote_timestamp: string
  quote_age_seconds: number
  days_to_expiration: number
  time_to_expiration_years: number
  implied_volatility: number | null
  implied_volatility_status: string | null
  implied_volatility_method: 'BSM_EQUIVALENT'
  provider_implied_volatility: number | null
  fair_value_volatility: number | null
  fair_value_volatility_source: FairValueVolatilitySource
  fair_value_volatility_independent: boolean
  fair_value: number | null
  fair_value_gap: number | null
  fair_value_gap_pct: number | null
  delta: number | null
  gamma: number | null
  theta: number | null
  theta_daily: number | null
  vega: number | null
  vega_per_vol_point: number | null
  rho: number | null
  intrinsic_value: number
  time_value: number | null
  moneyness: number
  log_moneyness: number
  spread: number | null
  spread_pct: number | null
  european_reference_value: number | null
  early_exercise_premium: number | null
  tree_steps: number
  crr_pricing_runtime_ms: number | null
  greeks_runtime_ms: number | null
  model_runtime_ms: number | null
  analysis_quality: 'HIGH_DATA_QUALITY' | 'MODERATE_DATA_QUALITY' | 'LOW_DATA_QUALITY'
  quality_flags: string[]
  warning_flags: string[]
}

export interface AnalyzerSmilePoint {
  provider_symbol: string
  option_type: OptionType
  strike: number
  log_moneyness: number
  market_implied_volatility: number | null
  fair_value_volatility: number | null
  fair_value_volatility_source: FairValueVolatilitySource
}

export interface OptionsAnalysisResult {
  symbol: string
  data_source: 'MASSIVE' | 'FAKE'
  underlying: {
    price: number
    bid: number | null
    ask: number | null
    last: number | null
    timestamp: string
    data_age_seconds: number
    source: 'MASSIVE' | 'FAKE'
  }
  selected_expiration: string
  available_expirations: string[]
  assumptions: {
    risk_free_rate: number
    risk_free_rate_source: 'CONFIGURED_DEFAULT' | 'REQUEST_OVERRIDE'
    dividend_yield: number
    dividend_yield_source: 'CONFIGURED_DEFAULT' | 'REQUEST_OVERRIDE'
    valuation_time: string
    pricing_model: 'CRR_AMERICAN'
    implied_volatility_method: 'BSM_EQUIVALENT'
    tree_steps: number
    target_expiration_days: number
    selected_expiration_days: number
    strike_window_pct: number
    max_contracts: number
    max_quote_age_seconds: number
    allow_last_price_fallback: boolean
    maximum_relative_spread: number
  }
  summary: {
    contracts_analyzed: number
    calls: number
    puts: number
    atm_strike: number
    source_contracts_considered: number
    contracts_with_market_iv: number
    contracts_with_independent_fair_vol: number
  }
  contracts: AnalyzedOptionContract[]
  smile: AnalyzerSmilePoint[]
  warnings: string[]
  timings: {
    market_data_ms: number
    implied_volatility_ms: number
    smile_construction_ms: number
    crr_pricing_ms: number
    greeks_ms: number
    pricing_and_greeks_ms: number
    total_ms: number
  }
}

export function analyzeOptions(
  request: OptionsAnalysisRequest,
  signal?: AbortSignal,
): Promise<OptionsAnalysisResult> {
  return apiPost<OptionsAnalysisResult>('/api/options/analyze', request, {
    signal,
    timeoutMs: 60_000,
  })
}
