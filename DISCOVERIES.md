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

## 6. What Is and Isn't Explained

### Empirically established

1. **sigma_half ∝ avg_nn_dist** (r = 0.988 cross-species, tight bootstrap CIs). The 3PL characteristic scale is anchored to physical inter-agent spacing. This is the single most robust finding.

2. **k ≈ 3 for most systems** (Poisson, Thomas clustered, jackdaw, jackdaw2, swift). This follows from dimensional analysis: for a homogeneous point process in d=3 dimensions, the number of KDE modes at bandwidth σ scales as (σ/σ₀)^(−d) in the intermediate regime, giving a sigmoid with effective steepness ~d.

3. **The shape curve k = f(log10_gamma) exists** — k and log10_gamma are not independent but trace a 1D manifold fit by a Hill model.

### Not explained (open theoretical problems)

1. **Why the 3PL form?** The centered 3PL `m(σ) = 1 + (N−1)/(1 + (2^(1/γ)−1)(σ/σ_half)^k)^γ` is a 4-parameter sigmoid that fits the data well, but no derivation from first principles exists. Why this specific parameterization rather than, say, a gamma CDF or a Hill function directly on m(σ)?

2. **What is log10_gamma physically?** It controls asymmetry of the sigmoid. For Poisson processes, γ≈1 (symmetric). For some Thomas processes, γ deviates. There is no theoretical prediction for what sets γ in a given point process.

3. **Why the Hill shape curve?** The empirical relationship `k = c + a/(1+((lg−d)/s)^p)` is purely descriptive. There is no theory predicting why k and log10_gamma should be coupled in this specific functional form, or what sets the parameters (a, d, s, p, c).

4. **Starling** (k=19, only 2 frames) is either a genuine departure from Poisson-like behavior or a fitting artifact. Cannot distinguish without more data.

### What this means

The 3PL model is a **phenomenological success but a theoretical puzzle**. It compresses the mode-count curve into 3 interpretable parameters, reveals a 2D intrinsic manifold, and anchors one axis (sigma_half) to physical spacing. But the specific functional form, the shape curve, and the physical meaning of log10_gamma remain empirical facts without theoretical foundation.

The k≈3 universality is **not a discovery** — it's the null expectation from dimensional analysis of a 3D homogeneous Poisson process. The discovery would be if real flocks showed k significantly different from 3, which would indicate non-Poisson collective structure. Currently, only starling hints at this (and the evidence is thin).

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

## 8. Open Questions

1. **Why the 3PL form?** The centered 3PL fits well but has no theoretical derivation. Is there a point-process argument for this specific sigmoid family?
2. **What is log10_gamma physically?** It controls sigmoid asymmetry. For Poisson γ≈1 (symmetric). What process generates γ≠1?
3. **Why the Hill shape curve?** k and log10_gamma trace a 1D manifold but the functional form is purely empirical.
4. **Is starling real?** k=19 is the only evidence for non-Poisson collective structure. With 2 frames, it could be artifact. More data needed.
5. **Why species-specific sigma_half/nn ratios?** (0.52–0.62, CIs don't overlap). What sets the prefactor?
