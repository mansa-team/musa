# T=10-Stable Correlation — Oxossi Study Guide

Pipeline: stocks API (SCORE>60, ticker endswith 3; LL/RL/ML 10y) → per-channel log1p+detrend → pooled OLS `eps = detrended − (βM·M + βS·S)`, M = XS mean, S = Selic-4189 YE standardized → Spearman(eps) + const-target shrink → blend w/ fundamental-distance proxy → time-bootstrap Fisher-z gate → sparse P for BL (Q = XANGO diffs, Ω = CI widths). Live: 42→9 stocks, raw=shrunk mean −0.095, 21/36 stable, meanCI 0.178, verdict STABLE.

---

## 1. Covariance in N>T Regime

Core idea: sample covariance needs `N(N−1)/2` off-diagonals from `T` obs. When `N>T` it is singular (rank ≤ T−1), not positive-definite, so `Σ⁻¹` (mean-variance weights) explodes. Condition number `κ = λmax/λmin` diagnoses it: κ→∞ = uninvertible. Fix = shrink or cut N before inverting.
Why at T=10: 9 stocks → 36 params from 10 points (~3.6 params/obs); 42 stocks → 861 params = pure noise. T=10 cannot support N>~10 without structure.
Formulas: `params = N(N−1)/2`; `κ(Σ) = λmax/λmin`; need `Σ ≻ 0` for `w ∝ Σ⁻¹μ`.
Refs: Ledoit & Wolf (2004) Honey-I-Shrunk-the-Sample-Covariance; Vershynin (2018) HDP.
Pitfall: inverting raw sample Σ at N≈T and "fixing" with tiny diagonal ridge — hides singularity, weights still garbage.

## 2. Ledoit-Wolf Shrinkage

Core idea: convex combo `Σ* = δF + (1−δ)S` of noisy sample `S` and structured target `F`. Shrinks extreme correlations toward target, restores PD, minimizes MSE vs. either alone. Oxossi uses constant-correlation target (mean-r off-diagonal), `δ=0.5` fixed.
Why at T=10: optimal-δ estimation itself is noisy at T=10; fixed 0.5 is a bias-variance truce (live: raw mean −0.095 = shrunk mean → target already ≈ sample mean, shrink only tames extremes).
Formulas: `Σ* = δF + (1−δ)S`; const-corr `F_ij = r̄·√(S_ii·S_jj)`; our `δ = 0.5`.
Refs: Ledoit & Wolf (2003) Improved Estimation (const-corr target); Ledoit & Wolf (2004) Honey (identity target + optimal δ).
Pitfall: tuning δ on the same 10 points (overfit) or using identity target on correlations (shrinks toward 0 = assumes uncorrelated — wrong prior here).

## 3. Market-Neutral Residuals

Core idea: raw profit co-movement is mostly shared market/rate beta, not idiosyncratic linkage. Per-channel `log1p` + linear detrend, then pooled OLS `detrended ~ M + S` strips the two factors; correlate residuals `eps` (Spearman rank). Spearman survives outliers/monotonic-nonlinearity Pearson misses.
Why at T=10: Pearson on 10 points = one outlier decides the sign; Spearman + detrending removes spurious trend-correlation (two growers look "correlated" only via time).
Formulas: `y = log1p(x) − trend(t)`; `eps = y − (βM·M + βS·S)`; `ρ_S = Pearson(rank(eps_i), rank(eps_j))`.
Refs: Fama & French (1993) Common Risk Factors; Sharpe (1964) CAPM (market factor logic).
Pitfall: pooled OLS forces one global (βM,βS) — heterogeneous betas leak into eps. Upgrade path is MixedLM hierarchical (per-ticker random slopes) when T allows; at T=10 pooled is the honest compromise.

## 4. Bootstrap CIs Done Right

Core idea: resample the TIME index (rows, with replacement) B≈1000×, recompute ρ each draw, Fisher-z transform `z = arctanh(r)`, take percentiles in z-space, back-transform. Gate: stable iff CI excludes 0 (or width < threshold). z-space is near-normal; r-space is bounded/skewed.
Why at T=10: analytic SE `1/√(T−3)` ≈ 0.38 — enormous. Only wide-CI-aware gating (live: meanCI 0.178, 21/36 stable) stops BL from betting on noise.
Formulas: `z = arctanh(r)`; `CI = tanh(z̄ ± z_α·SE_z)`; rule `stable ⟺ 0 ∉ CI`.
Refs: Efron (1979) Bootstrap Methods; Fisher (1921) Frequency Distribution of r (z-transform).
Pitfall (KNOWN BUG in `prof-corr.py`): resampling STOCKS (cross-section axis) instead of TIME. That measures universe-composition sensitivity, not correlation uncertainty. Must resample year rows of the eps panel.

## 5. Universe Selection

Core idea: shrink N before estimating — XANGO SCORE>60 + suffix-3 filter (quality/liquidity), DropCorrelated 0.7 (kill near-duplicates), hard cap 10–12 names. Fewer, better series beat many noisy ones.
Why at T=10: param count is quadratic — 12 stocks = 66 params (6.6 obs/param, feasible); 42 = 861 (hopeless). Live 42→9 = 36 params, matches gate output.
Formulas: `params = N(N−1)/2 ≤ ~5·T` rule of thumb; `drop if |ρ_raw| > 0.7`.
Refs: DeMiguel, Garlappi & Uppal (2009) 1/N vs optimized (estimation error dominance); Jagannathan & Ma (2003) constraints as shrinkage.
Pitfall: filtering on the same correlations you later estimate (selection bias) — filter on scores/duplicates, never on "nice CI" pairs.

## 6. Fundamental-Distance Proxy Correlation

Core idea: when history is 10 points, fundamentals carry independent signal: pairs with similar beta/score/sector get higher prior correlation. `ρ_proxy = f(|β_i−β_j|, |score_i−score_j|, same_sector)` blended with shrunk sample ρ. Sector grouping is the coarsest fallback (block-constant ρ within sector).
Why at T=10: pure-sample ρ has huge SE; proxy is T-independent — anchors pairs the bootstrap flags unstable.
Formulas: `ρ_final = λ·ρ_shrunk + (1−λ)·ρ_proxy`; e.g. `ρ_proxy = ρ0·exp(−d²/ℓ²)`, `d` = normalized fundamental distance.
Refs: Chan, Karceski & Lakonishok (1999) Factor-model covariances; Fama & French (1997) Industry Costs of Equity (sector grouping).
Pitfall: double-counting — scores already filtered the universe (§5); reusing the same score gap as correlation signal without down-weighting λ overstates confidence.

## 7. Black-Litterman Mapping

Core idea: BL blends market equilibrium `Π` with views `P·μ = Q + ε, ε ~ N(0,Ω)`. Oxossi: sparse P (one row per stable pair-spread), Q from XANGO score differentials, Ω diagonal from bootstrap CI widths (wide CI → high Ω → low view weight), τ ≈ 0.025–0.05 scales prior uncertainty.
Why at T=10: unstable pairs must enter Ω→∞ (or be dropped) — 15/36 live pairs correctly get ~zero weight. CI-width Ω is the mechanism that converts §4 gating into portfolio caution.
Formulas: `μ_BL = [(τΣ)⁻¹ + P'Ω⁻¹P]⁻¹[(τΣ)⁻¹Π + P'Ω⁻¹Q]`; `Ω_kk ∝ CIwidth_k²`.
Refs: Black & Litterman (1992) Global Portfolio Optimization; Idzorek (2004) Step-by-Step Guide (Ω from confidence + τ calibration).
Pitfall: dense P (a view per unstable pair) or constant Ω — both let noise dictate tilts. Keep P sparse, Ω CI-driven.

---

## Implementation Checklist

| # | Item | File |
|---|---|---|
| 1 | Assert `N(N−1)/2` vs T; log `κ(Σ)`; refuse to invert if singular | `stable_corr_sample.py` |
| 2 | Shrink to const-corr target, fixed `δ=0.5`; log raw vs shrunk mean | `stable_corr_sample.py` |
| 3 | Per-channel log1p+detrend → pooled OLS `~M+S` → Spearman(eps) | `stable_corr_sample.py` |
| 4 | FIX: bootstrap resamples TIME rows (not stocks); Fisher-z CIs + stability gate | `prof-corr.py` |
| 5 | XANGO>60 + suffix-3 → DropCorrelated 0.7 → cap 10–12; log 42→9 | `prof-corr.py` |
| 6 | Blend `ρ = λ·ρ_shrunk + (1−λ)·ρ_proxy` (beta/score gaps; sector fallback) | `stable_corr_sample.py` |
| 7 | Build sparse P (stable pairs only), Q = score diffs | `black-litterman.py` |
| 8 | Ω diagonal from CI widths; unstable → drop or Ω→∞ | `black-litterman.py` |
| 9 | Calibrate τ (0.025–0.05); posterior `μ_BL` + sanity-check tilts vs 1/N | `black-litterman.py` |
| 10 | Repro gate: rerun → 21/36 stable, meanCI ≈0.178, verdict STABLE | `prof-corr.py` + `stable_corr_sample.py` |
