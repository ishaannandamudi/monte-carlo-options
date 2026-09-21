# Phase 22 production frontend

## 1. Goals and scope

Phase 22 completes the core research roadmap by placing the validated Phase 1–21 FastAPI
capabilities behind one browser research terminal. The frontend organizes inputs, submits bounded
requests, explains returned diagnostics, and visualizes results. FastAPI and the Python domain
models remain authoritative for validation, numerical methods, resource limits, and experiment
records. No pricing, risk, calibration, or regime mathematics is reimplemented in TypeScript.

The terminal is designed for local quantitative research. It does not add live market feeds,
broker connectivity, authentication, trade execution, or HFT behavior.

## 2. Technology stack

The client uses React 19, strict TypeScript, Vite, React Router, TanStack Query, plain tokenized CSS,
and accessible SVG/CSS visualizations. Vitest, React Testing Library, jsdom, and type-aware ESLint
provide frontend validation. Dependencies are exact-version pinned in the pnpm lockfile, and Node
22.12 or newer is required by the Vite toolchain.

## 3. Application shell

`AppShell` owns the persistent navigation, compact header, backend status indicator, and nested
route outlet. The shell continues to render when FastAPI is unavailable, so a connectivity problem
does not destroy local form state or navigation. Overview reports backend health and package
version, frontend readiness, the completed core roadmap, major capabilities, quick links, and recent
experiment records when available.

## 4. Routing and lazy loading

The route registry is the single source for Overview, Pricing, Volatility, Models, Portfolio, Risk,
Regimes, QMC, and Experiments labels and paths. Each page is imported through `React.lazy`, giving
every workspace an independent production chunk. A wildcard route provides an explicit not-found
page. Tests can use the same application with a memory router.

## 5. Component system

Reusable UI primitives provide buttons, cards, fields, selects, alerts, badges, tables, tabs,
metric cards, help tips, loading states, empty states, and normalized error panels. Domain pages
compose these primitives. `OperationWorkbench` provides a common template/edit/run/cancel/reset
flow for research endpoints with large typed Pydantic payloads. `ResultPanel` renders common
metrics, Greeks, warnings, per-model statuses, P&L contributions, charts, structured JSON, copy,
and download actions.

The focused Pricing and Portfolio pages use conventional form controls for their common paths.
Broader research methods use editable, backend-valid JSON templates so advanced options remain
visible without building a second, incomplete schema system in the browser.

## 6. API client and type strategy

`api/client.ts` owns URL joining, JSON serialization, response decoding, cancellation, timeout
handling, and error normalization. `ApiError` exposes only a safe message, HTTP status, timeout
flag, and normalized FastAPI validation issues. Server tracebacks are never displayed.

Health, version, and experiment records have explicit TypeScript interfaces. Broad domain payloads
use a recursive `JsonValue`/`JsonObject` boundary without `any`; backend-valid templates make their
shape reviewable. A backend test compares every consumed method/path against the live FastAPI
OpenAPI document, preventing endpoint-name drift while Python Pydantic models remain authoritative.

## 7. Query cache and request policy

TanStack Query shares health and version state between the shell and Overview. Health refreshes
every 30 seconds; version is stable for the process lifetime. Experiment list/detail keys include
their complete filter or run identifier, and successful run, sweep, benchmark, or reproduction
operations invalidate experiment queries.

Client HTTP errors do not retry. Transient connectivity failures retry at most twice. Mutating
research operations do not retry automatically. Ordinary requests use a short timeout, while
calibration, full portfolio risk, HMM, sweep, and benchmark operations receive endpoint-specific
60-second to 10-minute budgets. Every active workbench exposes cancellation through an
`AbortController`.

## 8. Health, version, CORS, and configuration

FastAPI exposes dependency-free `/health` and `/version`; the latter reads the package version.
`VITE_API_BASE_URL` is the only public browser configuration and defaults to
`http://localhost:8000`. `CORS_ALLOWED_ORIGINS` is a comma-separated backend setting with explicit
localhost and loopback Vite origins. Wildcard origins and cross-origin credentials are disabled.

## 9. Pricing and Greeks

Pricing accepts explicit European option and market inputs and selects Black-Scholes-Merton, GBM
Monte Carlo, CRR, Heston analytical, Heston Monte Carlo, Merton analytical, or Merton Monte Carlo.
Only relevant model fields are shown. Results preserve price, uncertainty, intervals, runtime,
warnings, metadata, and available Greeks. The reusable Greek panel shows Delta, Gamma, Vega,
Theta, and Rho with backend units; no unavailable Greek is synthesized.

## 10. Model comparison

The Models workspace exposes Phase 15 comparison, market-fit, and risk requests. Returned model
rows retain success, excluded, failed, and unavailable statuses, plus price, dispersion, deviations,
equivalent IV, runtime, and Greeks when supplied. Partial results are never filtered out merely
because another model failed.

## 11. Volatility workflow

The Volatility workspace covers implied-volatility inversion, price-derived smiles, Phase 6
total-variance surfaces, Heston smiles and fitted surfaces, and Merton smiles. Line charts use
strike or log-moneyness on the horizontal axis and implied volatility on the vertical axis.
Surfaces and residual grids use readable heatmaps with adjacent values rather than a decorative 3D
projection.

## 12. Heston workflow

The Models workspace exposes analytical and Monte Carlo pricing, sensitivity, calibration, and fit
diagnostics. Calibration templates include target, weighting, optimizer, bounds, multistart, robust
loss, warm-start, and resource controls supported by the API. The result view preserves calibrated
parameters, objective/RMSE, convergence, runtime, Feller and active-bound warnings, residuals,
per-expiry data, fitted surfaces, and identifiability output. The UI presents diagnostics without
claiming that fitted parameters are uniquely identified.

## 13. Merton workflow

Merton operations include analytical and Monte Carlo prices, analytical-versus-Monte-Carlo
comparison, smile generation, sensitivities, Greeks, and jump diagnostics. Templates clearly name
jump intensity, mean log jump, jump volatility, compensator-related results, and expected jump
counts.

## 14. Portfolio construction and valuation

The Portfolio page maintains signed positions with stable IDs. Users can add, remove, and edit
European, American, Asian, digital, barrier, and basket positions, their instrument inputs, model
selection, quantity, and market inputs. Positive and negative quantities preserve long/short
semantics. Valuation results expose net value, gross long/short values, completeness, position unit
prices and values, model status, and warnings.

Portfolio-Greek operations preserve net, gross, scaled, and backend-provided grouped values for
underlying, expiry, option type, instrument class, and model. Tables and automatic charts remain
bounded by backend response sizes.

## 15. Scenarios, stress, and P&L explain

Scenario templates support relative or absolute spot shocks where accepted, volatility/rate/dividend
changes, time decay, Heston shocks, basket correlation, and per-asset spot changes. Results show base
and scenario value, full repricing P&L, Greek approximation, residual, flags, and contributions.
Spot-by-volatility grids render as heatmaps. Named/custom stress output retains portfolio,
position, and underlying contributions.

P&L explain renders Delta, Gamma, Vega, Theta, Rho, explained total, higher-order residual, and
full-repricing P&L. A signed contribution-bar view and identity note make the relationship
`explained + residual = full repriced P&L` visible.

## 16. Risk and backtesting

The Risk workspace exposes historical, parametric-normal, and correlated-factor Monte Carlo
VaR/ES; component, marginal, incremental, and empirical-tail contributions; combined risk/stress
reports; and VaR backtests. Forms identify confidence, horizon, valuation method, distribution,
simulation count, and contribution controls. Results retain base value, VaR, ES, runtime,
assumptions, warnings, concentration diagnostics, worst scenarios, and returned histogram or
quantile information. The browser never manufactures samples absent from the response.

Backtests display observation/exception counts, rates, Kupiec and Christoffersen diagnostics,
conditional coverage, transitions, exception dates, and traffic-light status when returned.

## 17. Regimes and Markov research

Regimes provides Gaussian-HMM fit, decode, forecast, model-order, Markov, and regime-risk
operations. Inputs expose observations, state count, covariance form, initialization, starts,
iteration cap, and tolerance. Results retain likelihood, AIC/BIC, convergence, state summaries,
transition matrices, stationary distributions, durations, decoded states, and forecast
probabilities.

Probability series are charted only when returned. Labels state that smoothed probabilities use
future observations and are not real-time estimates. Model-order output plots criteria across K
without selecting a supposedly correct K. Regime risk reports per-state and mixture VaR/ES and
explains that mixture risk is evaluated from the mixture distribution, not averaged state VaRs.

## 18. QMC workflow

QMC covers European, Asian, and basket estimates, sampling diagnostics, pseudorandom/Sobol/RQMC
comparison, and convergence. Templates expose samples, seed, replicates, power-of-two policy, path
construction, Brownian bridge, and PCA when supported. Results retain dimension, samples,
replicates, reference, errors, runtime, and empirical slopes. Deterministic Sobol uncertainty is
rendered as “Unavailable”; it is never shown as a zero standard error. Only independently scrambled
replicates supply RQMC standard error and confidence intervals.

## 19. Experiments, sweeps, and benchmarks

Experiments lists deterministically sorted, paginated runs with type/status/tag filters, creation
time, duration, tags, and signature prefix. Detail shows metadata, normalized inputs, environment,
metrics, result summary, artifacts, lineage, annotations, and failures in folded JSON. Artifacts are
metadata or safe downloads; arbitrary HTML is never injected.

Selected runs can be compared for configuration, metrics, runtime, environment, and diagnostic
differences. Reproduction reports original/replay IDs, signature and numerical matches, timing,
and mismatches. Bounded sweep and benchmark templates expose parent/child lineage, combinations,
warmups, repetitions, timing summaries, and baseline comparisons while respecting backend limits.

## 20. Visualization strategy

The frontend deliberately uses small dependency-free SVG line charts and CSS heatmaps. An
`AutoChart` recognizes bounded point series, surface/grid cells, convergence data, and probability
series from backend responses. Every chart has an accessible name and is accompanied by structured
metrics, a table, or complete folded JSON. This keeps numerical provenance inspectable and avoids
inventing values for incomplete responses.

## 21. Loading, errors, warnings, and partial results

Every asynchronous surface has a visible pending state and an explicit error or empty state.
Workbenches keep the prior result while a new request runs, provide cancellation, and surface
normalized field issues. Warnings include Feller violations, weak identifiability, unavailable
Greeks, discontinuous-payoff notes, approximation residuals, compatibility exclusions, resource
limits, and deterministic-Sobol uncertainty notes when returned.

## 22. Units and number formatting

Formatting is centralized for currency, decimal, percent, probability, volatility, Greeks,
scientific notation, runtimes, counts, and basis points. Inputs state maturity in years; rates,
dividend yields, volatility, and confidence as decimals; Heston `v0`/`theta` as variance; Merton
mean jump as a log jump; and jump intensity as jumps per year. Formatting never changes stored
numeric values.

## 23. Accessibility and responsive behavior

Inputs have programmatic labels, hints, error associations, and invalid state. Buttons, tables,
headers, alerts, details, and status regions use native semantics. Keyboard focus is visible; color
is not the sole status cue; reduced-motion preferences are honored. Wide screens use a persistent
sidebar and multi-column layouts. Smaller screens use a wrapping navigation strip, single-column
forms/cards, horizontally scrollable tables, and bounded charts.

## 24. Security

Only `VITE_*` public settings enter the bundle. The repository and built output are checked for
keys, tokens, passwords, provider secrets, and experiment-database credentials. JSON is rendered as
text, and React escaping is retained. The client has no `dangerouslySetInnerHTML`, credential
storage, trade execution, or broker integration.

## 25. Test strategy and API contract

Vitest/React Testing Library covers shell/navigation, health success/failure, formatters, API error
parsing, Pricing and Greeks, partial model comparison, portfolio editing and valuation, scenarios,
risk, HMM, deterministic and randomized QMC, and experiment list/detail/compare/reproduce flows.
HTTP is mocked at the transport boundary so tests exercise the real router, query layer, and page
components. FastAPI tests cover health/version/CORS and assert every frontend operation against
OpenAPI. The full Phase 1–21 ordinary and opt-in performance suites remain required.

Playwright is not included because provisioning a browser binary and second runner solely for this
local repository would add disproportionate weight. Release validation instead builds the
production bundle, starts real FastAPI and Vite development servers, tests health/version/CORS over
HTTP, and performs a visual browser smoke test.

## 26. Production build

`pnpm build` first performs strict TypeScript checking and then creates an optimized route-split
Vite bundle. Each major workspace remains a lazy chunk, shared React/query/router code is cached,
and no backend response data is embedded. Exact build duration, total output, major chunks, test
duration, and representative connectivity timing are recorded in
`benchmarks/phase22-frontend.md`.

## 27. Limitations and post-roadmap direction

The terminal intentionally favors editable research templates for rarely used high-dimensional
requests. It is a local single-user interface without authentication, collaborative sessions,
server-side rendering, offline mode, hosted artifact serving, or deployment automation. Charts
cover bounded returned structures rather than every possible nested diagnostic.

Phase 22 closes the 22-phase core roadmap. Future work should treat live data operations,
deployment, portfolio optimization, additional financial models, and HFT/broker execution as
separately scoped projects. None is implemented by this phase.
