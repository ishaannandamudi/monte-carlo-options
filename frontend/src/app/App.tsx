import { lazy, useMemo } from 'react'
import {
  createBrowserRouter,
  createMemoryRouter,
  RouterProvider,
  type RouteObject,
} from 'react-router'

import { AppShell } from '../components/layout/AppShell'

const OverviewPage = lazy(() => import('../pages/OverviewPage'))
const AnalyzePage = lazy(() => import('../pages/AnalyzePage'))
const PricingPage = lazy(() => import('../pages/PricingPage'))
const VolatilityPage = lazy(() => import('../pages/VolatilityPage'))
const ModelsPage = lazy(() => import('../pages/ModelsPage'))
const PortfolioPage = lazy(() => import('../pages/PortfolioPage'))
const RiskPage = lazy(() => import('../pages/RiskPage'))
const RegimesPage = lazy(() => import('../pages/RegimesPage'))
const QmcPage = lazy(() => import('../pages/QmcPage'))
const ExperimentsPage = lazy(() => import('../pages/ExperimentsPage'))
const NotFoundPage = lazy(() => import('../pages/NotFoundPage'))

const appRoutes: RouteObject[] = [
  {
    path: '/',
    element: <AppShell />,
    children: [
      { index: true, element: <AnalyzePage /> },
      { path: 'research', element: <OverviewPage /> },
      { path: 'research/pricing', element: <PricingPage /> },
      { path: 'research/volatility', element: <VolatilityPage /> },
      { path: 'research/models', element: <ModelsPage /> },
      { path: 'research/portfolio', element: <PortfolioPage /> },
      { path: 'research/risk', element: <RiskPage /> },
      { path: 'research/regimes', element: <RegimesPage /> },
      { path: 'research/qmc', element: <QmcPage /> },
      { path: 'research/experiments', element: <ExperimentsPage /> },
      // Preserve every Phase 22 deep link while presenting the grouped Research paths.
      { path: 'pricing', element: <PricingPage /> },
      { path: 'volatility', element: <VolatilityPage /> },
      { path: 'models', element: <ModelsPage /> },
      { path: 'portfolio', element: <PortfolioPage /> },
      { path: 'risk', element: <RiskPage /> },
      { path: 'regimes', element: <RegimesPage /> },
      { path: 'qmc', element: <QmcPage /> },
      { path: 'experiments', element: <ExperimentsPage /> },
      { path: '*', element: <NotFoundPage /> },
    ],
  },
]

interface AppProps {
  initialEntries?: string[]
}

export function App({ initialEntries }: AppProps) {
  const router = useMemo(
    () =>
      initialEntries
        ? createMemoryRouter(appRoutes, { initialEntries })
        : createBrowserRouter(appRoutes),
    [initialEntries],
  )
  return <RouterProvider router={router} />
}
