# Phase 19 regime/HMM benchmark report

## Method

Benchmarks use deterministic synthetic Gaussian HMM observations, PCG64 seeds, diagonal covariance
unless stated otherwise, and the optimized log-space NumPy/SciPy implementation. The opt-in tests
are marked `performance`, so ordinary CI does not repeatedly run the 10,000-observation cases.
Runtime depends on BLAS, CPU, state separation, and the number of EM iterations; the scenarios are
resource-shape references rather than service-level guarantees.

Run locally with:

```bash
pytest -m performance backend/tests/test_regime_performance.py
```

## Scenarios

| Scenario | Fit configuration | Expected memory order |
|---|---:|---:|
| 1,000 observations, 2 states | 1 start, at most 10 EM iterations | `O(TK + K^2)` |
| 10,000 observations, 2 states | 1 start, at most 10 EM iterations | `O(TK + K^2)` |
| 10,000 observations, 4 states | 1 start, at most 10 EM iterations | `O(TK + K^2)` |
| 10,000 observations, 3 features, 3 states | full covariance research case | `O(TK + KD^2)` |
| 5 starts | five independent sequential fits | approximately 5x one-start work |
| model order K=2..5 | four independently seeded fits | sum of candidate costs |

## Local measurements

Measured on 2026-09-20 in the project virtual environment on the local development machine:

| Operation | Bound | Wall time |
|---|---:|---:|
| univariate fit, 1,000 x 2 | 1 start, 10 EM iterations max | 0.59 s |
| univariate fit, 10,000 x 2 | 1 start, 10 EM iterations max | 6.43 s |
| univariate fit, 10,000 x 4 | 1 start, 10 EM iterations max | 5.79 s |
| full-covariance fit, 10,000 x 3 features x 3 states | 1 start, 5 iterations max | 4.22 s |
| univariate fit, 1,000 x 2 | 5 starts, 10 iterations max | 3.03 s |
| model order K=2..5, 1,000 observations | 1 start/candidate, 5 iterations max | 1.96 s |
| Viterbi plus forward/backward decode, 1,000 x 2 | one call | 72.1 ms |
| one-step filtering, 1,000 calls | independent prior per call | 36.9 ms |
| Delta mixture risk, 100,000 samples, 1 factor, 2 states | one call | 11.3 ms |

The three marked performance tests completed in 13.71 seconds together. EM can converge before the
maximum iteration count, so fit timings also reflect the generated sample and convergence path.

Forward, backward, Viterbi, and one-step filtering share the emission engine. Decode requires
`O(TK)` scores plus a `T x K` predecessor array. One-step filtering is `O(K^2 + KD^2)` for full
covariance. Mixture regime risk is linear in simulations times feature dimension after one
covariance factorization per state.

## Memory observations

The implementation never stores an `T x K x K` xi tensor. Each EM E-step stores alpha, beta,
filtered, smoothed, and emission arrays of shape `T x K`, then accumulates xi into a `K x K`
expected-transition matrix. For 100,000 observations and 10 states, each float64 `T x K` array is
about 8 MB. API caps reject oversized dimensions before inference and bound serialized probability
series separately from internal inference.

## Interpretation

Fit time scales approximately linearly in observations and iterations and roughly quadratically in
states through transition updates. Multi-start and model-order experiments intentionally multiply
fit cost and remain sequential in Phase 19. Regime-risk Monte Carlo uses vectorized draws and a
single path loop over states, not over simulations. No multiprocessing, GPU, or quasi-Monte Carlo
backend is included.
