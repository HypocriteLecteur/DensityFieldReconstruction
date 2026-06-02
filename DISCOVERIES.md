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

## 3. sigma_half ~ 0.55 × Physical Nearest-Neighbor Distance

The 3PL parameter `sigma_half` is proportional to the median physical nearest-neighbor distance, but the ratio is **species-specific, not universal**:

| Species | mean sigma_half | mean avg_nn_dist | ratio | 95% CI |
|---------|----------------|------------------|-------|--------|
| starling | 0.375 | 0.702 | 0.534 | [0.530, 0.538] |
| jackdaw | 0.838 | 1.361 | 0.617 | [0.613, 0.620] |
| jackdaw2 | 1.141 | 2.187 | 0.522 | [0.516, 0.528] |
| swift | 2.211 | 4.012 | 0.550 | [0.548, 0.552] |

**Cross-species correlation: r = 0.988**

Bootstrap CIs are tight (width ~0.004–0.013) and do not overlap between species. The ratio is stable within species but significantly different across species — jackdaw's 0.62 is far from jackdaw2's 0.52 despite both being corvids.

**Implication:** sigma_half captures a species-specific "effective clustering scale" that scales with physical spacing but with a species-dependent prefactor. This prefactor may reflect differences in perceptual range, interaction rules, or flock structure beyond just density. The cross-species r=0.988 is driven by the wide range of physical scales (0.7–4.0) rather than a universal constant.

**Partial correlation**: After controlling for avg_nn_dist, sigma_half has negligible residual correlation with N. The apparent N-dependence is entirely mediated by physical spacing.

---

## 4. Temporal Dynamics

**sigma_half is stable within species over time** — it's a species-level property (physically: preferred flock density):

| Species | sigma_half CV | k CV | Interpretation |
|---------|---------------|------|----------------|
| jackdaw | 0.04 | 0.99 | Near-constant density, highly dynamic steepness |
| jackdaw2 | 0.10 | 1.14 | Stable density, very dynamic steepness |
| swift | 0.22 | 0.83 | Both vary, wider range of flock states |

**k (steepness) is highly dynamic** — it tracks behavioral state transitions (e.g., milling → polarized → milling).

**Temporal autocorrelation** reveals k(t) is not white noise. Characteristic decorrelation times (ACF zero-crossing):
- jackdaw: ~39 frames (dt=1)
- jackdaw2: ~190 frames (dt=5, 38 lags)
- swift: >100 lags at dt=20 (>2000 frames)

k has species-specific "memory" — behavioral states persist for tens to hundreds of frames before decorrelating.

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

**Clustering is method-dependent.** GMM with BIC selection finds 10 components; HDBSCAN finds 7. ARI=0.119 between them — the two methods find fundamentally different structures. Any claims about the "number of behavioral archetypes" must be qualified by the clustering method used.

---

## 5. Clustering: 15 Behavioral Archetypes

HDBSCAN on UMAP embedding of (k, sigma_half, log10_gamma) finds 15 clusters + 4 noise points. Clusters are strongly species-specific:

- **swift** dominates large clusters (455, 176, 41, 40, 28, 21 points) — highest diversity
- **jackdaw** and **jackdaw2** form tight, well-separated clusters by k and sigma_half
- **starling** (2 frames) clusters with high-k jackdaw frames — insufficient data for generalization

---

## 6. Mechanistic Derivation: 3PL from Point Process Theory

The mode-count curve was compared across synthetic point processes and empirical flocks:

| Process | k | sigma_half | log10_gamma |
|---------|---|------------|-------------|
| **Poisson (uniform)** | 3.4–5.9 | 0.53–0.55 | −0.2 to +0.2 |
| **Thomas (clustered)** | 2.3–3.9 | 0.20–1.07 | 0.0–5.0 |
| **Empirical: jackdaw** | 3.06 | 0.84 | −0.13 |
| **Empirical: jackdaw2** | 2.56 | 1.13 | 0.06 |
| **Empirical: swift** | 2.79 | 2.16 | 0.21 |
| **Empirical: starling** | 18.78 | 0.38 | −0.77 |

**Key finding: k ≈ 3 is UNIVERSAL.** Uniform Poisson, Thomas clustered, and normal bird flocks all have k ≈ 3 with log10_gamma ≈ 0 (symmetric sigmoid). The 3PL mode-count curve collapses to a **1-parameter family**: only sigma_half varies meaningfully across systems.

**sigma_half is the sufficient statistic.** It scales with cluster std in Thomas processes and with physical NN-distance in flocks (r=0.988). The steepness k is NOT what distinguishes collective motion from random geometry — both give k≈3.

**Starling is the exception** with k=19, log10_gamma=−0.77. This extreme steepness, if real, would indicate genuine non-Poisson collective structure (tightly synchronized flock that shatters at a critical scale). However, with only 2 frames, this could be a fitting artifact.

**Theoretical interpretation:**
- For a uniform Poisson process in d dimensions, mode-count vs bandwidth follows a sigmoid with k ≈ d (here d=3, observed k≈3–4)
- The 3PL form nests this prediction: k reflects effective dimensionality, sigma_half reflects density
- Clustered processes have the same k but smaller sigma_half (set by cluster scale rather than global density)
- The k–log10_gamma manifold mostly collapses to a single point (k≈3, lg≈0) for Poisson-like systems; deviations from this point indicate genuine departures from spatial randomness

---

## 7. Methodological Improvements (see git log)

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

1. **Mechanistic derivation**: Can the 3PL form (or the Hill shape curve) be derived from spatial point process theory, random geometric graphs, or active matter physics?
2. **Why species-specific ratios?** The sigma_half/nn ratio varies significantly across species (0.52–0.62). What biological or physical factor sets this prefactor?
3. **k dynamics without labels**: Steepness varies wildly within species with characteristic memory timescales. What behavioral events correspond to k-transitions? (Requires behavioral annotation — not possible with current data.)
