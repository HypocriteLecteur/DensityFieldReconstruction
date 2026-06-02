# Scientific Discoveries

Parameter manifold investigation of biological swarm configurations via the centered 3PL model.

**Last updated**: 2026-06-02

---

## 1. The Intrinsic Parameter Manifold

The 3PL model `m(sigma) = 1 + (N-1) / (1 + (2^(1/gamma)-1) × (sigma/sigma_half)^k)^gamma` yields 3 parameters per frame:

| Parameter | Description | Range (all data) |
|-----------|-------------|-------------------|
| `k` | Steepness at half-mode scale | 0.95 – 20.0 |
| `sigma_half` | Scale where half the modes have merged | 0.37 – 3.73 |
| `log10_gamma` | Asymmetry (log10 space) | −1.13 – 5.00 |

**Datasets**: 4 species, 1,119 frames total (swift: 765, jackdaw: 200, jackdaw2: 152, starling: 2).

---

## 2. Two-Dimensional Manifold Structure

The parameter space collapses to an **intrinsic 2D manifold**:

- **Axis 1 (shape)**: `k_proj` — projected k on the k–log10_gamma shape curve (steepness/asymmetry)
- **Axis 2 (scale)**: `sigma_half` — characteristic scale of substructure

The k–log10_gamma relationship is well-fit by a **Hill model**:

```
k = 1.34 + 18.66 / (1 + ((log10_gamma − (−1.01)) / 0.36)^2.13)
```

R² = 0.95. This captures the steep decay from high asymmetry (low k) to near-symmetric (high k) with a flat plateau at k ≈ 1–2.

**Per-species manifold coordinates** (mean k_proj, sigma_half):

| Species | k_proj | sigma_half | Physical interpretation |
|---------|--------|------------|------------------------|
| swift | 3.52 | 2.21 | Gradual transition, large inter-agent spacing |
| jackdaw | 4.72 | 0.84 | Moderate steepness, moderate spacing |
| jackdaw2 | 4.16 | 1.14 | Moderate steepness, moderate spacing |
| starling | 18.84 | 0.38 | Very steep, tight packing (only 2 frames!) |

---

## 3. sigma_half = 0.55 × Physical Nearest-Neighbor Distance

**This is the key discovery.** Across all 4 species, the 3PL parameter `sigma_half` is directly proportional to the median physical nearest-neighbor distance in the point cloud:

| Species | mean sigma_half | mean avg_nn_dist | ratio |
|---------|----------------|------------------|-------|
| starling | 0.375 | 0.702 | 0.534 |
| jackdaw | 0.838 | 1.361 | 0.617 |
| jackdaw2 | 1.141 | 2.187 | 0.522 |
| swift | 2.211 | 4.012 | 0.550 |

**Cross-species correlation: r = 0.988**

Within-species: r(sigma_half, avg_nn_dist) = 0.55–0.97.

**Implication:** The "characteristic scale of substructure" from the 3PL model is not a separate behavioral parameter — it directly reads out the physical inter-agent spacing. Species separate on this axis because they fly at different densities, not because of different collective dynamics.

**Partial correlation**: After controlling for avg_nn_dist, sigma_half has negligible residual correlation with flock size N (swift r = −0.06, jackdaw2 r = −0.22). The apparent N-dependence is spurious and entirely mediated by physical spacing.

---

## 4. Temporal Dynamics

**sigma_half is stable within species over time** — it's a species-level property (physically: preferred flock density):

| Species | sigma_half CV | k CV | Interpretation |
|---------|---------------|------|----------------|
| jackdaw | 0.04 | 0.99 | Near-constant density, highly dynamic steepness |
| jackdaw2 | 0.10 | 1.14 | Stable density, very dynamic steepness |
| swift | 0.22 | 0.83 | Both vary, wider range of flock states |

**k (steepness) is highly dynamic** — it tracks behavioral state transitions (e.g., milling → polarized → milling).

---

## 5. Cluster Stability (Bootstrap)

HDBSCAN on the raw 3D standardized parameter space (not UMAP) finds **7 robust clusters** + 9 noise points:

| Cluster | Size | Stability | Composition |
|---------|------|-----------|-------------|
| 0 | 39 | 0.885 | swift (39) |
| 1 | 21 | 0.880 | jackdaw2 (21) |
| 2 | 10 | 0.887 | jackdaw (9), jackdaw2 (1) |
| 3 | 13 | 0.889 | swift (13) |
| 4 | 25 | 0.886 | jackdaw (13), jackdaw2 (11) |
| 5 | 6 | 0.892 | swift (6) |
| 6 | 996 | 0.884 | swift (699), jackdaw (178), others |

**Overall stability: 0.886 ± 0.082** (100 bootstrap resamples).

**UMAP inflates cluster count.** The UMAP embedding amplifies local structure, splitting the data into 15 clusters. Bootstrap on the original parameter space reveals only 7 clusters are robust to resampling. The UMAP clusters are useful for exploration but overstate the true number of behavioral archetypes.

One mega-cluster (996/1119 fits) contains the majority of all data across species, plus six small species-specific clusters. Clusters are dominated by species, suggesting the manifold captures inter-species differences more than intra-species behavioral states.

---

## 5. Clustering: 15 Behavioral Archetypes

HDBSCAN on UMAP embedding of (k, sigma_half, log10_gamma) finds 15 clusters + 4 noise points. Clusters are strongly species-specific:

- **swift** dominates large clusters (455, 176, 41, 40, 28, 21 points) — highest diversity
- **jackdaw** and **jackdaw2** form tight, well-separated clusters by k and sigma_half
- **starling** (2 frames) clusters with high-k jackdaw frames — insufficient data for generalization

---

## 6. Methodological Improvements

All implemented and committed (see git log for details):

| Issue | Root cause | Fix |
|-------|-----------|-----|
| DBSCAN eps=0 crash | `avg_nn_dist` intermittent zero from CUDA sync | CPU-based scipy cdist |
| Mean-shift never converges | float32 catastrophic cancellation in `∥x∥²+∥y∥²−2x·y` | Direct coordinate differences |
| Mode count corrupt at small sigma | `torch.cdist` precision bug on identical vectors | Direct-diff distance computation |
| Slow iterations (>60s/frame) | tol=0 causes infinite iteration; relax=1.9 causes oscillation | Standard mean-shift (1.0×), reduced max_iter |
| Cache loss on crash | No incremental saving | Save every 50 steps during Phase 2, resume on restart |
| Shape curve oscillation | Spline overfitting (R²=0.997 spurious) | Hill model (R²=0.95, honest) |

---

## 7. Open Questions

1. **Starling is severely undersampled** (2 frames). Cannot characterize the species.
2. **What drives k dynamics?** Steepness varies wildly within species — what behavioral event does a k-transition correspond to?
3. **Validation on synthetic data**: Run mode_counting on point clouds with known cluster counts to measure accuracy vs scale.
4. **Mechanistic derivation**: Can the 3PL form be derived from spatial point process theory or active matter physics?
5. ~~Cluster stability~~: Confirmed — 7 robust clusters (0.886±0.082), UMAP inflates to 15.
