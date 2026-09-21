# Phase 22 frontend build and validation report

Measured on 2026-09-21 on macOS 26.6.2 arm64 with Node 24.19.0, pnpm 11.19.0, and
Python 3.12.14. Timings are wall-clock observations from one local validation run; they are
engineering baselines rather than cross-machine performance guarantees.

## Production build

Command: `pnpm build` (`tsc -b && vite build`)

- Result: passed; 185 modules transformed
- Final Vite bundling phase: 409 ms (329 ms in the separately timed build)
- Complete command including TypeScript: 3.25 s
- Total emitted file bytes: 446,379 bytes (435.92 KiB; filesystem allocation 472 KiB)
- Initial JavaScript: 360.27 kB raw / 112.99 kB gzip
- Shared stylesheet: 21.57 kB raw / 4.89 kB gzip

Major route and shared chunks:

| Chunk | Raw | Gzip |
|---|---:|---:|
| Shared application/runtime | 360.27 kB | 112.99 kB |
| Result panel + charts | 14.39 kB | 5.40 kB |
| Experiments | 8.74 kB | 3.18 kB |
| Portfolio | 8.41 kB | 2.89 kB |
| Pricing | 6.63 kB | 2.54 kB |
| Models | 4.80 kB | 1.89 kB |
| Overview | 4.53 kB | 1.78 kB |
| Volatility | 3.48 kB | 1.64 kB |
| QMC | 3.27 kB | 1.47 kB |
| Risk | 3.17 kB | 1.48 kB |
| Regimes | 3.03 kB | 1.37 kB |

All nine workspaces remain route-level lazy chunks. No backend result, market observation, secret,
or experiment record is embedded in the bundle.

## Automated validation timings

| Validation | Result | Wall time |
|---|---:|---:|
| Frontend Vitest/RTL | 23 passed in 6 files | 12.91 s final run (14.34 s measured run) |
| Frontend strict TypeScript | passed | included in build gate |
| Frontend ESLint, zero warnings | passed | included in validation gate |
| Backend ordinary pytest | 1,237 passed, 59 deselected | 28.85 s final run (59.63 s prior run) |
| Backend performance pytest | 56 passed, 3 skipped, 1,237 deselected | 85.78 s |
| Ruff lint | passed | — |
| Ruff format check | 313 files formatted | — |
| Strict mypy application target | 144 source files, no issues | — |

The three skipped performance cases retain their existing environment-dependent opt-in conditions;
no assertions were weakened.

## Live integration observations

Real Uvicorn and Vite development servers were started on loopback for browser and CORS checks.

| Interaction | HTTP result | Local total time |
|---|---:|---:|
| `GET /health` | 200 | 5.057 ms |
| `GET /version` | 200, version `0.22.0` | 1.343 ms |
| Black-Scholes pricing | 200, price `8.827321225352122` | 2.593 ms |
| Vite development startup | ready | 216 ms |

The configured `http://127.0.0.1:5173` origin received
`Access-Control-Allow-Origin`; `https://unrelated.example` did not. A live browser smoke test
verified Overview connectivity/version, lazy Pricing navigation, a real Black-Scholes request,
returned price and all five analytical Greeks, and the mixed-instrument Portfolio workspace.

## Browser test scope

Component-level integration tests exercise the production router, TanStack Query, forms, result
rendering, and mocked HTTP boundary. A standalone Playwright suite is intentionally not added in
Phase 22 because its browser provisioning and second runner would be disproportionate for this local
repository. The live-server browser smoke test and FastAPI OpenAPI contract test cover the remaining
integration boundary. A deployment pipeline with shared browser infrastructure can add persistent
cross-browser E2E coverage later.
