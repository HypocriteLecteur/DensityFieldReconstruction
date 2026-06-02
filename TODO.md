# Research TODO

Living task list for the parameter manifold investigation.

## Remaining

- [x] **Mechanistic derivation of 3PL** — k≈3 follows from dimensional analysis (d=3), not a discovery. 3PL form, shape curve, and log10_gamma remain empirical facts without theory.
- [ ] **Interactive visualization** — Plotly dashboard for exploring the manifold interactively

## Done

- [x] Validate mode_counting on synthetic data — 12/16 perfect, mean err 6.4%, fails only when clusters overlap
- [x] Bootstrap cluster stability — 7 robust clusters (0.886±0.082), UMAP inflates to 15
- [x] GMM vs HDBSCAN comparison — method-dependent (10 vs 7 components, ARI=0.119)
- [x] sigma_half/nn ratio bootstrap CIs — species-specific, CIs don't overlap
- [x] Temporal autocorrelation of k(t) — jackdaw tau0~39fr, not white noise
- [x] Cache nn_dists — saves to `scenarios/{name}/nn_dists.npy`
- [x] Fix DBSCAN eps=0 crash
- [x] Fix float32 catastrophic cancellation in mean-shift
- [x] Fix mean-shift non-convergence (relaxation factor)
- [x] Optimize max_iter and add incremental saving
- [x] Fit shape curve with Hill model (R²=0.95)
- [x] sigma_half vs physical NN-distance analysis (r=0.988)
- [x] Temporal trajectories k(t) and sigma_half(t)
- [x] N-dependence analysis with partial correlations
- [x] Create DISCOVERIES.md
