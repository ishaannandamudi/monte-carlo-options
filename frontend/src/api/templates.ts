import type { JsonObject } from './types'

export function vanillaContract(optionType = 'CALL'): JsonObject {
  return {
    option_type: optionType,
    exercise_style: 'EUROPEAN',
    spot: 100,
    strike: 100,
    maturity_years: 1,
  }
}

export function market(): JsonObject {
  return { risk_free_rate: 0.03, volatility: 0.2, dividend_yield: 0.01 }
}

export function simulation(numSimulations = 10_000): JsonObject {
  return {
    num_simulations: numSimulations,
    seed: 42,
    variance_reduction: 'ANTITHETIC',
    confidence_level: 0.95,
  }
}

export function hestonParameters(): JsonObject {
  return { v0: 0.04, kappa: 2, theta: 0.04, sigma_v: 0.3, rho: -0.7 }
}

export function mertonParameters(): JsonObject {
  return { jump_intensity: 0.8, jump_mean: -0.12, jump_volatility: 0.25 }
}

export function vanillaPosition(positionId = 'call-1', quantity = 1): JsonObject {
  return {
    position_id: positionId,
    quantity,
    instrument_class: 'VANILLA_EUROPEAN',
    underlying_id: 'SPY',
    instrument: vanillaContract(),
    market: market(),
    model: 'BLACK_SCHOLES',
  }
}

export function portfolio(): JsonObject {
  return { portfolio_id: 'research-portfolio', base_currency: 'USD', positions: [vanillaPosition()] }
}

export function riskMappings(): JsonObject[] {
  return [{ factor_id: 'SPY_SPOT_RETURN', factor_type: 'SPOT_RETURN', underlying_id: 'SPY' }]
}

export function covariance(): JsonObject {
  return {
    factor_ids: ['SPY_SPOT_RETURN'],
    covariance_matrix: [[0.0004]],
    mean_changes: [0],
  }
}

export function historicalDataset(): JsonObject {
  const values = [-0.03, -0.015, -0.005, 0.004, 0.012, 0.02]
  return {
    scenarios: values.map((value, index) => ({
      timestamp: `2025-01-${String(index + 2).padStart(2, '0')}T00:00:00Z`,
      observations: [{ factor_id: 'SPY_SPOT_RETURN', value }],
    })),
  }
}

export function experimentDefinition(): JsonObject {
  return {
    experiment_type: 'PRICING',
    operation: 'BLACK_SCHOLES',
    name: 'BSM research price',
    tags: ['frontend', 'phase22'],
    input_payload: { contract: vanillaContract(), market: market() },
  }
}

export function hmmDataset(): JsonObject {
  return {
    observations: [
      [-0.012], [-0.008], [-0.015], [-0.004], [0.006], [0.012], [0.009], [0.014],
      [-0.01], [-0.006], [0.008], [0.011], [-0.009], [0.007], [0.013], [-0.011],
    ],
    feature_names: ['return'],
  }
}

export function hmmParameters(): JsonObject {
  return {
    num_states: 2,
    feature_dimension: 1,
    covariance_type: 'DIAGONAL',
    initial_probabilities: [0.5, 0.5],
    transition_matrix: [[0.95, 0.05], [0.1, 0.9]],
    means: [[-0.01], [0.01]],
    covariances: [[[0.0001]], [[0.00015]]],
  }
}

export function qmcConfig(method = 'SCRAMBLED_SOBOL', dimension = 1): JsonObject {
  return {
    method,
    dimension,
    num_samples: 1024,
    scramble: method === 'SCRAMBLED_SOBOL',
    seed: 7,
    num_randomizations: method === 'SCRAMBLED_SOBOL' ? 8 : 1,
    power_of_two_policy: 'REQUIRE_POWER_OF_TWO',
    path_construction: 'STANDARD',
  }
}

export function hestonDataset(): JsonObject {
  return {
    spot: 100,
    risk_free_rate: 0.02,
    dividend_yield: 0.01,
    underlying_symbol: 'SYNTHETIC',
    observations: [
      { strike: 90.9372934468, maturity_years: 0.5, option_type: 'PUT', observed_price: 2.1464870015, observed_implied_volatility: 0.2123362263, bid: 2.1264870015, ask: 2.1664870015 },
      { strike: 100.5012520859, maturity_years: 0.5, option_type: 'CALL', observed_price: 5.4197036213, observed_implied_volatility: 0.1932367327, bid: 5.3997036213, ask: 5.4397036213 },
      { strike: 111.0710610356, maturity_years: 0.5, option_type: 'CALL', observed_price: 1.5072755323, observed_implied_volatility: 0.1738305422, bid: 1.4872755323, ask: 1.5272755323 },
    ],
  }
}
