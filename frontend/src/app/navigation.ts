export interface NavigationItem {
  label: string
  path: string
  code: string
  description: string
  phase: 'available'
}

export const navigationItems: readonly NavigationItem[] = [
  {
    label: 'Analyze',
    path: '/',
    code: 'AN',
    description: 'American options from a ticker',
    phase: 'available',
  },
  {
    label: 'Research',
    path: '/research',
    code: 'RS',
    description: 'Advanced quantitative workbench',
    phase: 'available',
  },
] as const

export const researchNavigationItems: readonly NavigationItem[] = [
  {
    label: 'Pricing',
    path: '/research/pricing',
    code: 'PX',
    description: 'Analytical and simulation engines',
    phase: 'available',
  },
  {
    label: 'Volatility',
    path: '/research/volatility',
    code: 'VOL',
    description: 'Implied volatility, smiles, and surfaces',
    phase: 'available',
  },
  {
    label: 'Models',
    path: '/research/models',
    code: 'MDL',
    description: 'Calibration and model comparison',
    phase: 'available',
  },
  {
    label: 'Portfolio',
    path: '/research/portfolio',
    code: 'PF',
    description: 'Valuation, Greeks, and scenarios',
    phase: 'available',
  },
  {
    label: 'Risk',
    path: '/research/risk',
    code: 'RISK',
    description: 'VaR, expected shortfall, and backtesting',
    phase: 'available',
  },
  {
    label: 'Regimes',
    path: '/research/regimes',
    code: 'HMM',
    description: 'Markov and latent-regime research',
    phase: 'available',
  },
  {
    label: 'QMC',
    path: '/research/qmc',
    code: 'QMC',
    description: 'Sobol and randomized QMC analysis',
    phase: 'available',
  },
  {
    label: 'Experiments',
    path: '/research/experiments',
    code: 'EXP',
    description: 'Tracked runs, comparisons, and benchmarks',
    phase: 'available',
  },
] as const
