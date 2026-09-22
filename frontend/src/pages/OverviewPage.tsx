import { useQuery } from '@tanstack/react-query'
import { Link } from 'react-router'

import { healthQueryOptions, versionQueryOptions } from '../api/system'
import { listExperiments } from '../api/experiments'
import { researchNavigationItems } from '../app/navigation'
import {
  Alert,
  Badge,
  Card,
  ErrorPanel,
  LoadingState,
  MetricCard,
  PageHeader,
  Section,
} from '../components/ui'

export default function OverviewPage() {
  const health = useQuery(healthQueryOptions())
  const version = useQuery(versionQueryOptions())
  const recentRuns = useQuery({
    queryKey: ['experiments', 'recent'],
    queryFn: ({ signal }) => listExperiments({ page: 1, pageSize: 3, sort: 'NEWEST' }, signal),
    retry: false,
  })
  const isLoading = health.isPending || version.isPending
  const connectionError = health.error ?? version.error

  return (
    <div className="page-stack">
      <PageHeader
        description="A local workspace for pricing, calibration, model comparison, portfolio analytics, risk, regimes, and reproducible numerical experiments."
        eyebrow="Research control plane"
        title="Quantitative Derivatives Workbench"
        actions={<Badge variant="success">Core roadmap complete</Badge>}
      />

      {connectionError ? (
        <ErrorPanel
          error={connectionError}
          onRetry={() => {
            void health.refetch()
            void version.refetch()
          }}
          title="Backend disconnected"
        />
      ) : null}

      {isLoading ? (
        <LoadingState label="Connecting to the research backend" />
      ) : (
        <div className="metric-grid" aria-label="Backend status summary">
          <MetricCard
            detail={health.data?.service ?? 'monte-carlo-options'}
            label="Backend connection"
            status={health.isSuccess ? 'positive' : 'negative'}
            value={health.isSuccess ? 'Connected' : 'Unavailable'}
          />
          <MetricCard
            detail="FastAPI liveness"
            label="Health status"
            status={health.data?.status === 'ok' ? 'positive' : 'warning'}
            value={health.data?.status.toUpperCase() ?? 'UNKNOWN'}
          />
          <MetricCard
            detail="Backend application"
            label="Service version"
            status="neutral"
            value={version.data?.version ?? 'Unavailable'}
          />
          <MetricCard
            detail="React · TypeScript · Vite"
            label="Frontend status"
            status="neutral"
            value="Ready"
          />
        </div>
      )}

      <Section
        description="Open an integrated workflow backed by the validated Phase 1–21 quantitative APIs."
        title="Research areas"
      >
        <div className="quick-grid">
          {researchNavigationItems.map((item) => (
              <Link className="quick-link" key={item.path} to={item.path}>
                <Card interactive>
                  <div className="quick-link__topline">
                    <span className="quick-link__code">{item.code}</span>
                    <Badge variant="accent">Open</Badge>
                  </div>
                  <strong>{item.label}</strong>
                  <p>{item.description}</p>
                  <span className="quick-link__action">Open area →</span>
                </Card>
              </Link>
          ))}
        </div>
      </Section>

      <Section title="Delivery roadmap">
        <div className="roadmap-grid">
          <Card>
            <span className="roadmap-card__phase">PHASES 01—21</span>
            <strong>Quantitative backend</strong>
            <p>
              Pricing, calibration, model risk, portfolios, VaR/ES, regimes, QMC, and experiment
              lineage are implemented and validated.
            </p>
            <Badge variant="success">Complete</Badge>
          </Card>
          <Card>
            <span className="roadmap-card__phase">PHASE 22</span>
            <strong>Research terminal</strong>
            <p>
              Pricing, volatility, models, portfolios, risk, regimes, QMC, and tracked experiments
              are integrated through the typed application boundary.
            </p>
            <Badge variant="success">Complete</Badge>
          </Card>
          <Card>
            <span className="roadmap-card__phase">RESEARCH MODE</span>
            <strong>Local and reproducible</strong>
            <p>
              Backend resource limits, deterministic seeds, explicit units, warnings, and complete
              structured results remain visible in every workflow.
            </p>
            <Badge variant="accent">Active</Badge>
          </Card>
        </div>
      </Section>

      <Section title="Recent experiments" description="Latest locally tracked Phase 21 runs.">
        {recentRuns.isPending ? <LoadingState compact label="Loading recent runs" /> : null}
        {recentRuns.data?.items.length === 0 ? <Card><p>No tracked runs yet. Create one in Experiments.</p></Card> : null}
        {recentRuns.data && recentRuns.data.items.length > 0 ? <div className="recent-runs">{recentRuns.data.items.map((run) => <Link className="recent-run" key={run.run_id} to="/experiments"><span><Badge variant={run.status === 'COMPLETED' ? 'success' : run.status === 'FAILED' ? 'danger' : 'warning'}>{run.status}</Badge>{run.experiment_type}</span><code>{run.run_id.slice(0, 12)}…</code></Link>)}</div> : null}
        {recentRuns.error ? <p className="fine-print">Experiment history is unavailable; core workflows remain usable.</p> : null}
      </Section>

      <Alert title="Data policy" variant="info">
        This terminal displays only backend-returned research data and project metadata. It does
        not fabricate market prices, portfolio values, samples, or model outputs.
      </Alert>
    </div>
  )
}
