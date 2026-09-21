# Phase 22A frontend foundation

> Historical note: Phase 22A records the foundation that preceded the completed production
> frontend. The placeholder routes described below were replaced by the workflows documented in
> [phase22-frontend-production.md](phase22-frontend-production.md).

## Scope and architectural role

Phase 22A establishes the browser application boundary over the completed Phase 1–21 quantitative
backend. It contains one lightweight Overview and reserves the named areas required by Phase 22.
Pricing, calibration, portfolio, risk, regime, QMC, and experiment workflows remain placeholders.
This keeps the foundation reviewable and avoids embedding domain assumptions before those workflows
have specifications.

The backend remains authoritative for mathematical inputs, validation, pricing, diagnostics, and
resource limits. The frontend formats results and manages interaction; it does not duplicate
quantitative calculations.

## Stack choice

React and strict TypeScript provide typed composable UI. Vite supplies a small development and
build layer. React Router handles browser URLs and nested shell routes. TanStack Query manages
remote state without adding a separate client-state store. Plain CSS keeps the first design system
dependency-free. Vitest and React Testing Library exercise user-observable behavior in jsdom.

Dependencies are exact-version pinned with a pnpm lockfile. The package requires Node 22.12 or
newer, matching the supported Vite toolchain. TypeScript compilation, ESLint, tests, and the Vite
production bundle are independent scripts suitable for CI.

## Routing and application shell

`App` builds a browser router in production and a memory router when tests supply initial entries.
The root `AppShell` owns the persistent sidebar, compact top bar, backend connection indicator, and
nested content outlet. Overview is the index. Pricing, Volatility, Models, Portfolio, Risk,
Regimes, QMC, and Experiments all resolve to a shared polished placeholder. A wildcard below the
shell renders a not-found page.

Page modules load through `React.lazy`, so future domain pages already have route-level split
points. Navigation metadata lives in one file and drives the sidebar, Overview quick links, and
placeholder titles. This prevents labels and paths from drifting.

The shell uses a fixed sidebar on wide screens and a normal-flow horizontal navigation region on
narrow screens. Content widths, focus treatment, contrast, and reduced-motion handling are defined
centrally. No animation framework or second styling abstraction is present.

## Component system

`components/ui` contains Button, Card, MetricCard, Alert, Badge, Input, Select, DataTable,
LoadingState, EmptyState, ErrorPanel, PageHeader, and Section. Components own semantic structure
and stable class names. Design tokens and the component/layout style sheets own presentation.
Callers select small documented variants instead of copying declarations into pages.

Input and Select connect labels, hints, errors, `aria-describedby`, and `aria-invalid`. Loading and
connection indicators expose status roles. ErrorPanel exposes a safe alert, normalized validation
issues, and an optional retry action. DataTable requires a caption and row-key function and has an
explicit empty row. These are foundations; workflow-specific tables and form composition remain
for Phase 22.

## API client architecture

`api/config.ts` reads `VITE_API_BASE_URL` with `http://localhost:8000` as the local default.
`api/client.ts` owns `fetch`, JSON encoding/decoding, an eight-second default timeout, caller
`AbortSignal` propagation, and response-status checks. It converts transport failures, timeouts,
cancellation, normal HTTP failures, and FastAPI validation arrays into `ApiError`. Components see
a stable message plus optional `{location, message, type}` issues and never render raw stack traces.

Phase 22A hand-maintains only `HealthResponse` and `VersionResponse`. Generating every backend type
would create a large unused artifact. The generic request boundary and per-domain modules provide a
direct migration point for selective OpenAPI generation during full Phase 22.

The backend exposes dependency-free `GET /health` and `GET /version`. Version returns the package
version from `backend.app.__version__`, preventing duplicate version constants. Overview presents
both values, while the shell uses the shared health query for a persistent connected/checking/
disconnected state. A failure renders degraded UI and leaves routing operational.

## Query and cache policy

One `QueryClient` is created at application startup. Queries use a 30-second default stale time and
five-minute garbage-collection time, and they do not refetch merely because the browser regains
focus. Network and server failures retry at most twice. HTTP client errors below 500 do not retry.
Mutations default to no retry because future writes may not be idempotent.

Health has a 30-second polling interval. Version remains fresh for the process lifetime. Shared
query keys mean Overview and the connectivity banner observe one cache entry and one in-flight
request. Test helpers disable retries to make failure assertions immediate and deterministic.

## Environment and CORS

The frontend example environment contains only `VITE_API_BASE_URL`; there are no browser secrets.
Backend `Settings` adds `cors_allowed_origins`, loaded from comma-separated
`CORS_ALLOWED_ORIGINS`. Defaults allow `http://localhost:5173` and
`http://127.0.0.1:5173`. `CORSMiddleware` echoes only configured origins, permits the existing API
methods and headers, and disables credentialed cross-origin requests. Tests prove a configured
local origin succeeds and an unconfigured origin receives no allow-origin header.

## Formatting and standard states

`utils/formatters.ts` uses a fixed `en-US` display locale and bounded precision for currency,
percent, volatility, probability, runtime, scientific, and generic decimal values. Null,
undefined, NaN, and infinity return an em dash by default. Model values remain JavaScript numbers;
formatting does not alter API data.

LoadingState, ErrorPanel, EmptyState, and BackendStatusBanner cover pending, failed, absent, and
degraded connectivity states. The Overview never invents market or portfolio metrics: it displays
only service metadata, project status, and navigation.

## Tests and validation

Frontend tests cover the shell, navigation, successful health/version rendering, degraded backend
behavior, loading, normalized validation errors, metric cards, and every numeric formatter family.
They mock only the browser network boundary. Production compilation checks strict TypeScript;
ESLint enables type-aware rules and React Hooks checks with zero warnings.

Backend tests cover `/version`, local CORS preflight, rejected origins, and environment parsing in
addition to the complete ordinary Phase 1–21 regression suite. Ruff, Ruff formatting, and strict
mypy remain the Python quality gates. A real local startup/connectivity smoke test verifies the
Vite default URL can read FastAPI health with the correct CORS header.

## Full Phase 22 continuation

The continuation will replace the eight placeholders with bounded workflows, choose the consumed
backend endpoints, add their request/response types, and build domain forms, tables, charts,
exports, and tests. Likely groupings are analytical/simulation pricing, volatility and calibration,
model comparison, portfolio/scenarios, VaR/ES, regimes, QMC diagnostics, and experiment lineage.
It should reuse the shell, navigation registry, components, request helper, query client, states,
formatters, and CORS configuration established here.

Phase 22A intentionally adds no new financial models, browser-side pricing logic, live trading,
HFT connection, Redux store, Docker layer, or fabricated market data.
