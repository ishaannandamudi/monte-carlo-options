import { Suspense } from 'react'
import { NavLink, Outlet } from 'react-router'

import { navigationItems, researchNavigationItems } from '../../app/navigation'
import { BackendStatusBanner } from '../system/BackendStatusBanner'
import { LoadingState } from '../ui'

export function AppShell() {
  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand__mark" aria-hidden="true">
            Q
          </div>
          <div>
            <strong>Quant Research</strong>
            <span>Derivatives workbench</span>
          </div>
        </div>

        <nav aria-label="Primary navigation" className="primary-nav">
          {navigationItems.map((item) => (
            <NavLink
              className={({ isActive }) => `nav-link${isActive ? ' nav-link--active' : ''}`}
              end={item.path === '/'}
              key={item.path}
              to={item.path}
            >
              <span className="nav-link__code" aria-hidden="true">
                {item.code}
              </span>
              <span>{item.label}</span>
              <span className="nav-link__available" aria-hidden="true">›</span>
            </NavLink>
          ))}
          <details className="research-nav" open>
            <summary>Research tools</summary>
            {researchNavigationItems.map((item) => (
              <NavLink
                className={({ isActive }) => `nav-link nav-link--nested${isActive ? ' nav-link--active' : ''}`}
                key={item.path}
                to={item.path}
              >
                <span className="nav-link__code" aria-hidden="true">{item.code}</span>
                <span>{item.label}</span>
              </NavLink>
            ))}
          </details>
        </nav>

        <div className="sidebar__footer">
          <span>Local research environment</span>
          <strong>CORE PHASES 01—22</strong>
        </div>
      </aside>

      <div className="workspace">
        <header className="topbar">
          <div>
            <span className="topbar__context">Workspace</span>
            <strong>American Options Research</strong>
          </div>
          <BackendStatusBanner />
        </header>
        <main className="content" id="main-content">
          <Suspense fallback={<LoadingState label="Loading workspace" />}>
            <Outlet />
          </Suspense>
        </main>
      </div>
    </div>
  )
}
