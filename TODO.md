# Research TODO

Living task list for the parameter manifold investigation. Prioritized by impact.

## In Progress

## High Priority

## High Priority

- [ ] **Validate mode_counting on synthetic data** — Generate point clouds with known cluster counts, measure accuracy vs scale and N
- [ ] **What drives k dynamics?** — Correlate k(t) transitions with known behavioral events (turns, mergers, polarization changes)
- [ ] **Bootstrap cluster stability** — Resample fits, re-run HDBSCAN, measure cluster overlap scores

## Medium Priority

- [ ] **Starling data acquisition** — Only 2 frames; need more to characterize the species
- [ ] **Temporal autocorrelation of k(t)** — Compute ACF to see if steepness has characteristic timescales
- [ ] **sigma_half vs body size** — Correlate with known physical measurements (body length, wingspan) across species
- [ ] **Cross-species sigma_half/avg_nn ratio stability** — Test if the 0.55 ratio holds across more datasets

## Low Priority

- [ ] **Mechanistic derivation of 3PL** — Can the mode-count curve be derived from spatial point process theory?
- [ ] **Alternative clustering methods** — Compare HDBSCAN with spectral clustering, GMM
- [ ] **Interactive visualization** — Plotly dashboard for exploring the manifold

## Done

- [x] Cache nn_dists — saves to `scenarios/{name}/nn_dists.npy`, loaded on re-runs
- [x] Fix DBSCAN eps=0 crash
- [x] Fix float32 catastrophic cancellation in mean-shift
- [x] Fix mean-shift non-convergence (relaxation factor)
- [x] Optimize max_iter and add incremental saving
- [x] Fit shape curve with Hill model (R²=0.95)
- [x] sigma_half vs physical NN-distance analysis (r=0.988)
- [x] Temporal trajectories k(t) and sigma_half(t)
- [x] N-dependence analysis with partial correlations
- [x] Create DISCOVERIES.md
