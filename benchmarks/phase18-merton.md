# Phase 18 Merton benchmark

This report records local, single-process NumPy/SciPy measurements on 2026-09-20. Timings are
descriptive rather than service guarantees.

## Workloads

| Workload | Runtime | Notes |
|---|---:|---|
| Analytical single call | 1.14 ms | 13 terms at `1e-12` tail tolerance |
| Analytical 10 strikes | 3.78 ms | Independent strike loop; 13 terms each |
| Analytical 100 strikes | 35.85 ms | Independent strike loop; 13 terms each |
| Monte Carlo 100k paths | 4.42 ms | error -0.02139; SE 0.05969 |
| Monte Carlo 500k paths | 17.43 ms | error +0.02555; SE 0.02701 |
| Monte Carlo 1m paths | 35.79 ms | error +0.01514; SE 0.01910 |
| 100-strike IV smile | 89.35 ms | 100 successful Phase 4 inversions |
| Full paths 10k × 100 | 17.46 ms | 16.08 MB returned spot/count matrices |

Terminal pricing is O(paths) memory. Its principal arrays are terminal spots, counts, diffusion
normals, aggregate-jump normals, and discounted payoffs; one million entries per float array are
about 8 MB and one million `int64` counts are about 8 MB. It never allocates `paths × jumps`.
The optional 10k × 101 spot matrix is about 8.08 MB and the 10k × 100 count matrix is about 8 MB,
excluding temporary vectors.

The benchmark also reports Monte Carlo error against the analytical mixture and analytical term
counts. Run the opt-in check with:

```bash
.venv/bin/pytest -q -m performance backend/tests/test_merton_performance.py
```
