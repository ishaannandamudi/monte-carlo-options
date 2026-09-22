# American Options Analyzer and Quantitative Research Frontend

The default `/` route is a ticker-only American equity-options analyzer. Enter a symbol to fetch
its backend-owned Massive quote, expirations, and bounded chain, then review market midpoint,
American CRR fair value, BSM-equivalent IV, independent fair-value volatility, Greeks, liquidity,
and quality diagnostics. The full Phase 22 terminal remains available under `/research`.

The backend remains authoritative for contracts, validation, pricing,
calibration, portfolio valuation, risk, numerical methods, resource limits, and experiment
lineage. The browser manages forms, requests, comparisons, charts, formatting, and exports.

## Stack

- React 19 and strict TypeScript
- Vite production and development tooling
- React Router with lazy page modules
- TanStack Query for health, version, experiment lists, and experiment details
- Tokenized plain CSS with no runtime styling framework
- Accessible SVG line charts and CSS heatmaps with adjacent structured data
- Vitest, React Testing Library, and jsdom
- ESLint with type-aware TypeScript and React Hooks rules

Node.js 22.12 or newer is required. Exact dependencies are recorded in `pnpm-lock.yaml`.

## Install and run

Start FastAPI from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
.venv/bin/uvicorn backend.app.main:app --host 127.0.0.1 --port 8000
```

In another terminal:

```bash
cd frontend
pnpm install
pnpm dev
```

Vite serves `http://127.0.0.1:5173`. The equivalent npm setup is `npm install` followed by
`npm run dev`; use one package manager consistently.

## Commands

```bash
pnpm dev          # Vite development server
pnpm test         # one Vitest/RTL run
pnpm test:watch   # interactive test watcher
pnpm typecheck    # strict TypeScript project check
pnpm lint         # zero-warning ESLint check
pnpm build        # typecheck and optimized route-split bundle
pnpm preview      # serve the production bundle locally
```

## Environment and CORS

`.env.example` contains the only public browser setting:

```text
VITE_API_BASE_URL=http://localhost:8000
```

The value is normalized before paths are joined. There are no provider keys, database credentials,
or other secrets in the browser configuration. FastAPI accepts a comma-separated
`CORS_ALLOWED_ORIGINS`; defaults allow `http://localhost:5173` and
`http://127.0.0.1:5173` without enabling a wildcard origin.

## Routes and workflows

- **Analyze (`/`)**: ticker-only chain retrieval and American option analysis through
  `POST /api/options/analyze`; expiration, calls/puts, sorting, contract details, warnings, and the
  two-series volatility smile are available after one request. TanStack Query caches each complete
  symbol/expiration/assumption request for two minutes and exposes manual refresh.
- **Research (`/research`)**: backend health/version, frontend readiness, capabilities, recent experiments, and
  core-roadmap status.
- **Pricing**: explicit European inputs for Black-Scholes-Merton, GBM Monte Carlo, CRR, Heston
  analytical/Monte Carlo, and Merton analytical/Monte Carlo; model-specific parameters appear only
  when needed.
- **Volatility**: implied-volatility inversion, price-implied smiles, total-variance surface build,
  Heston/Merton smiles, and fitted Heston surfaces.
- **Models**: Phase 15 comparison and market fit, Heston pricing/sensitivity/calibration/fit
  diagnostics, and Merton pricing/comparison/Greeks/jump diagnostics.
- **Portfolio**: signed mixed-instrument builder, valuation, net/gross/scaled Greeks, full-repricing
  scenarios, spot-volatility grids, named stresses, and P&L explain.
- **Risk**: historical, parametric-normal, and factor Monte Carlo VaR/ES, contributions, combined
  risk/stress report, and VaR backtesting.
- **Regimes**: Gaussian-HMM fit/decode/forecast/model-order research, Markov diagnostics, and
  regime-mixture VaR/ES.
- **QMC**: European, Asian, and basket QMC, sampling diagnostics, method comparisons, and
  convergence. Deterministic Sobol uncertainty is shown as unavailable.
- **Experiments**: filterable/paginated runs, details, metadata, metrics, artifacts, JSON export,
  run comparison, reproduction, bounded sweeps, and benchmarks.

Analyzer assumptions are collapsed under **Advanced assumptions**. They include target expiration
days, rate/dividend overrides, tree steps, strike window, maximum contracts, quote-age threshold,
and last-price fallback. Blank rate and dividend values use the visible backend configuration.

Advanced workflows use validated JSON templates because the backend domain models are broad and
research-oriented. Templates expose the meaningful method, target, weighting, optimizer, seed,
sample, horizon, and diagnostic controls and remain fully editable. The focused Pricing and
Portfolio pages provide conventional form controls for their common paths.

## Folder structure

```text
src/
├── api/             # typed transport, domain templates, health and experiment clients
├── app/             # router, navigation and QueryClient
├── components/
│   ├── layout/      # persistent application shell
│   ├── research/    # operation runner and structured result presentation
│   ├── system/      # backend connectivity
│   ├── ui/          # reusable controls and states
│   └── visualization/ # accessible SVG line chart and heatmap
├── pages/           # ticker analyzer plus nine lazy-loaded research workspaces and tests
├── styles/          # tokens, global, component and responsive layout rules
├── test/            # render and HTTP-mocking utilities
└── utils/           # number formatting and safe browser export/copy
```

## API and query architecture

`api/client.ts` owns JSON serialization, response parsing, HTTP status checks, normalized FastAPI
validation issues, cancellation, and timeouts. An `ApiError` contains a safe message, status,
timeout marker, and field issues; stack traces are never rendered. Ordinary calls use an eight-
second default. Research operations select 60–180 second timeouts, while calibration, risk, HMM,
sweep, and benchmark requests use up to five or ten minutes. Every running workbench exposes a
cancel action backed by `AbortController`.

Small stable contracts such as health, version, experiment list/detail, and common JSON values are
hand typed. Domain request templates use a recursive `JsonValue` boundary and are checked against
the FastAPI OpenAPI route table in the backend suite. This avoids hundreds of unused generated
types while keeping `any` out of core client code.

TanStack Query shares health/version calls between the shell and Overview, polls health every 30
seconds, and caches experiment list/detail requests by their complete filter key. Client HTTP
errors do not retry; transient/network failures retry at most twice. Experiment creation invalidates
experiment queries. Mutating research runs are not retried automatically.

## Results, charts, units, and failures

Every asynchronous workflow has pending, retryable error, empty, or structured result UI. Common
metrics, Greeks, warnings, model success/excluded/failed statuses, P&L contribution identity,
series charts, surface/grid heatmaps, and full folded JSON are preserved. Results can be copied or
downloaded as JSON. Artifact metadata is plain text; arbitrary HTML is never injected.

Rates, volatility, and probabilities use decimal input units. Maturity is years. Heston `v0` and
`theta` are variances; `sigma_v` is volatility of variance. Merton jump mean is a log jump and
intensity is jumps per year. Backend Greek units are preserved: Vega and Rho are per absolute unit,
and Theta is per calendar year. Help tips define Greek, VaR/ES, Feller, regime-probability, and QMC
terms without changing those conventions.

## Tests and browser validation

Vitest/React Testing Library covers the shell, navigation, backend success/failure, normalized API
errors, components, formatting, Pricing, partial model comparison, portfolio editing/valuation,
scenarios, VaR/ES, HMM, deterministic and randomized QMC, and experiment filtering, pagination,
detail, comparison, and reproduction. These component-level integration tests mock only HTTP and
exercise the production router/query/component stack.

Playwright is intentionally absent: adding a browser binary and a second runner solely for this
local application would materially increase dependencies. Release validation instead builds the
production bundle, starts real FastAPI and Vite development servers, verifies health/version/CORS
over HTTP, and performs a visual browser smoke test. A future deployment pipeline can add
Playwright when browser provisioning is shared infrastructure.

See `docs/phase22-frontend-production.md` for architecture and
`benchmarks/phase22-frontend.md` for measured build/test output.

## Limits

The frontend does not reproduce backend mathematics, bypass resource limits, fetch fabricated
market data, provide trade execution, or implement HFT. Charts visualize only returned bounded
points; they never reconstruct missing samples. The terminal is a local research interface rather
than an authentication, multi-user, deployment, or broker platform.
