export interface NavigationItem {
  label: string
  path: string
  code: string
  description: string
  phase: 'available'
}

export const navigationItems: readonly NavigationItem[] = [
  {
    label: 'Overview',
    path: '/',
    code: 'OV',
    description: 'Platform connection and roadmap',
    phase: 'available',
  },
  {
    label: 'Pricing',
    path: '/pricing',
    code: 'PX',
    description: 'Analytical and simulation engines',
    phase: 'available',
  },
  {
    label: 'Volatility',
    path: '/volatility',
    code: 'VOL',
    description: 'Implied volatility, smiles, and surfaces',
    phase: 'available',
  },
  {
    label: 'Models',
    path: '/models',
    code: 'MDL',
    description: 'Calibration and model comparison',
    phase: 'available',
  },
  {
    label: 'Portfolio',
    path: '/portfolio',
    code: 'PF',
    description: 'Valuation, Greeks, and scenarios',
    phase: 'available',
  },
  {
    label: 'Risk',
    path: '/risk',
    code: 'RISK',
    description: 'VaR, expected shortfall, and backtesting',
    phase: 'available',
  },
  {
    label: 'Regimes',
    path: '/regimes',
    code: 'HMM',
    description: 'Markov and latent-regime research',
    phase: 'available',
  },
  {
    label: 'QMC',
    path: '/qmc',
    code: 'QMC',
    description: 'Sobol and randomized QMC analysis',
    phase: 'available',
  },
  {
    label: 'Experiments',
    path: '/experiments',
    code: 'EXP',
    description: 'Tracked runs, comparisons, and benchmarks',
    phase: 'available',
  },
] as const
