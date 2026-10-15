# Context Document (Part 3, REVISED): Final Architecture — Markov Persistence Removed

## Purpose
**This document replaces the earlier Part 3 (which included a first-order Markov
persistence layer).** Paste this version, not the earlier one, alongside Parts 1 and 2 to
restore full context. Core change: the Markov occurrence-persistence layer has been
**completely removed**. Temporal persistence (both occurrence spell-length and magnitude
day-to-day dependence) is now a fully acknowledged, documented limitation rather than a
partially-addressed feature. This also resolved a second, related loose end (see Part B)
that would otherwise have reintroduced persistence implicitly.

---

## Part A: The core decision that produced v3 (unchanged)

Two parallel products, not one merged framework — closing the loop on Ch. 10, Fig. 10.2,
deterministic vs. stochastic downscaling. Both built from the **same shared fitted core**,
read two different ways.

**Precedent, correctly scoped**: E-OBS ensemble (Cornes et al., 2018, JGR Atmospheres
123(17), 9391–9409) — real citation, valid precedent for the *dual-product architecture
pattern* only (deterministic + separately-generated stochastic ensemble, maintained side
by side). Not evidence it solves the same problem this project solves — E-OBS's ensemble
quantifies interpolation uncertainty from station-network sparsity, not sub-grid
representativeness.

---

## Part B: Shared core (fit once)

### Predictor stack (X)
- Gridded precip/temp product value — **native-resolution (nearest-neighbor) assignment**
  (Part E)
- Topographic covariates: elevation, slope, aspect, TPI (from DEM)
- MODIS LST/EVI covariates
- Day-of-year harmonics (sin/cos)
- **Previous-day wet/dry state (precip only) — resolved handling, see below.** This is a
  genuinely useful predictive feature (today's occurrence probability responds to recent
  wetness), kept in the shared predictor stack. **Critical resolution**: for the
  **stochastic** product, this feature is populated using the **deterministic backbone's**
  own previous-day value, identical across all 100 realizations on a given day — never
  each realization's own simulated previous-day draw. This avoids (a) reintroducing
  sequential-per-realization compute dependency, and (b) creating uncontrolled,
  un-acknowledged individual-level persistence through the back door. It remains an
  "inherited" signal (same category as the predictor's own synoptic autocorrelation, Part
  K), not a "designed" persistence mechanism — consistent with this document's full
  removal of designed persistence. For the **deterministic** product, this feature is
  simply that product's own previous-day value — unambiguous, since there's only one
  series.

### Two model components, fit once, read two ways
```
Classifier:  P(wet | X)          → probabilistic output, precipitation occurrence only
QRF:         F(amount | X, wet)  → precip magnitude, wet days only
QRF:         F(tmax | X)         → no wet/dry gate — continuous everywhere
QRF:         F(tmin | X)         → no wet/dry gate — continuous everywhere
```
QRF mechanics: for a given X, routes through the forest to training-data leaf nodes,
forms a weighted empirical distribution of observed y-values in those leaves. Median =
deterministic. Random sample from that weighted distribution = stochastic.
(`quantregForest` in R, `quantile-forest` in Python.)

---

## Part C: Deterministic product (unchanged)

```
p_wet = classifier(X)          # X includes this series' own previous-day state
if p_wet > threshold:
    precip = QRF_precip.predict_median(X)
else:
    precip = 0
tmax = QRF_tmax.predict_median(X)
tmin = QRF_tmin.predict_median(X)
```

**Reproducibility requirement**: median comes directly from the model's quantile
function (p=0.5), never computed by averaging stochastic realizations.

**Documentation caution**: single best point-guess given the predictors, not a verified
reconstruction of the true historical value — classifier misses ~33% of true wet days
(POD≈0.67) even at its best-tuned threshold.

---

## Part D: Stochastic product — full daily generation algorithm (SIMPLIFIED)

Run per day *t*, per realization *r* (r = 1...100), per pixel/station.
**No recursive dependency remains — fully parallelizable across t, r, and pixel/station.**

```
STEP 1 — occurrence (fully independent draws; no Markov nudge, no per-realization memory)
    p_cov,t = classifier(X_t)     # X_t's prev-day-wet feature = deterministic backbone's
                                   # own previous-day value, identical across all r
    wet_t(r) ~ Bernoulli(p_cov,t) # independent draw, same p_cov,t for every r

STEP 2 — raw magnitude, independently sampled per realization
    if wet_t(r): precip_raw_t(r) ~ QRF_precip.sample(X_t)
    tmax_raw_t(r) ~ QRF_tmax.sample(X_t)
    tmin_raw_t(r) ~ QRF_tmin.sample(X_t)

STEP 3 — Schaake shuffle, per day, across realizations, over
         {precip magnitude, tmax, tmin} — occurrence is NOT shuffled (see Part F)
    final_precip_t, final_tmax_t, final_tmin_t = schaake_shuffle_day(
        raw_ensemble = {precip_raw_t, tmax_raw_t, tmin_raw_t},
        wet_mask = wet_t (all r),
        target_date = t
    )
```

**What was removed**: the logit-shift Markov nudge (`logit(p_final,t) = logit(p_cov,t) +
Δ_state`) that previously modified Step 1 using each realization's own prior-day outcome.
See Part K for the accepted consequence, and the note below for why it was dropped.

**Why it was dropped (brief history, for context — not part of the active design)**: the
mechanism itself was correctly resolved before removal — per-realization recursive
tracking does deliver both correct population-level aggregate shifts *and* genuine
individual-level persistence simultaneously (these aren't competing properties; the
recursive design gets both). The decision to remove it regardless was a simplicity/scope
choice — avoiding the sequential-per-realization compute constraint and the added
implementation surface, in favor of a simpler system with one clearly documented gap
rather than a more complex system with a correctly-working but harder-to-verify
persistence mechanism.

---

## Part E: Interpolation decision (unchanged)

**Nearest-neighbor, for all products (precip and temperature), at all native→target
resolution steps.** Fidelity-preservation: assigns actual retrieved values; bilinear
manufactures synthetic sub-pixel values never part of any real measurement. True for
precipitation (convective-patchiness) and temperature (no elevation-awareness — blends
warm-valley and cold-ridge cells by geometric distance alone). Resolution earned via
topographic/LST covariates in the shared core, not the resampling method.

**Accepted artifact**: all 1km sub-pixels within one parent coarse cell share an
identical predictor time series and (where Y-proxy substitution applies) identical
Schaake Y-vector — a resolution-limit artifact, not a bug.

---

## Part F: Schaake shuffle — full specification

### Scope (unchanged conclusion, updated rationale)
Magnitude (precip amount given wet, tmax, tmin) and cross-variable coherence, **within a
single day only**. Occurrence is never shuffled.

**Updated rationale for excluding occurrence** (the Markov-protection reason no longer
applies, since Markov is gone — but exclusion still holds, for a more fundamental reason):
occurrence is binary. Rank-matching a binary variable via the standard continuous-variable
Schaake mechanism produces massive ties and degenerate ranks — sorting 100 zeros and ones
gives you blocks of identical values with no meaningful internal ordering to match against
historical rank. This is the same zero-inflation problem the precipitation
bias-correction literature handles by only rank-matching non-zero (wet-day) magnitudes
(see Singularity Stochastic Removal, Vrac et al., 2016, from earlier in this project) —
excluding occurrence from the shuffle is standard practice for this reason alone,
independent of whatever persistence architecture is or isn't present elsewhere.

### Mechanics (Clark et al., 2004, J. Hydrometeorology 5(1), 243–262)
For each station/pixel *j* and variable *k*, independently, against the **same shared
set of selected historical dates** across all *j* and *k*:
1. Pick N historical dates from a calendar window around the target date (other years only).
2. Build historical matrix Y using those same dates, for every station/variable.
3. Sort the raw stochastic ensemble x (Step 2, Part D) ascending.
4. Find the rank of each value in y.
5. Reassign: whichever ensemble slot corresponds to y's rank-r date now receives the
   r-th smallest value of sorted x.

### Y-proxy substitution
Station observations where available; predictor's own historical values elsewhere.
Weakest in convective/patchy precipitation regimes — same representativeness caveat as
elsewhere in this project.

### tmax/tmin physical consistency
Genuine correspondence created via shared historical rank-matching (real (tmax,tmin)
pairs are never inverted). Finite historical pool (~15–30 dates/window) means rare
inversions are plausible on low-diurnal-range days. No fallback built preemptively —
validate via post-implementation diurnal-range (tmax−tmin) distribution check; add a
fallback only if that check reveals a real problem.

### Python skeleton
```python
import numpy as np
from scipy.stats import rankdata

def schaake_shuffle_day(
    stochastic_ensemble,   # [n_stations, n_variables, n_realizations], raw & unordered
                            # variables = {precip_magnitude, tmax, tmin}; occurrence excluded
    historical_obs,        # indexed by date: [date, station, variable]
    target_date,
    wet_mask,               # [n_stations, n_realizations] — restricts precip Y-pool to
                             # historical wet days matching each realization's occurrence
    window_days=15,
    n_realizations=100,
    rng=None,
):
    rng = rng or np.random.default_rng()
    candidate_dates = get_window_dates(target_date, window_days, exclude_year=target_date.year)
    replace = len(candidate_dates) < n_realizations
    selected_dates = rng.choice(candidate_dates, size=n_realizations, replace=replace)

    Y = historical_obs.loc[selected_dates]
    shuffled = np.empty_like(stochastic_ensemble)

    n_stations, n_variables, _ = stochastic_ensemble.shape
    for j in range(n_stations):
        for k in range(n_variables):
            x = stochastic_ensemble[j, k, :]
            y = Y[:, j, k]
            x_sorted = np.sort(x)
            y_rank = rankdata(y, method="ordinal").astype(int) - 1
            shuffled[j, k, :] = x_sorted[y_rank]

    return shuffled   # occurrence (wet_t) untouched, passed through separately
```

---

## Part G: Markov persistence layer — REMOVED

Previously specified here: a per-season logit-shift nudge (`Δ_wet`, `Δ_dry`, estimated
from `p0`, `pww`, `pwd`) applied to the classifier's daily probability, conditioned on
each realization's own simulated previous-day state. **Fully removed from the active
architecture.** Not implemented, not part of the current design. See Part D for the
brief note on why, and Part K for the accepted consequence. (If this needs to be
revisited later, the formulas and worked example are preserved in the prior, superseded
version of this document, not reproduced here to avoid ambiguity about current scope.)

---

## Part H: Data structures (unchanged)

```
tmax_deterministic  (dims: time, y, x)
tmax_stochastic     (dims: time, y, x, realization=100)
# same pattern for tmin, precip
```
NetCDF4 or Zarr. Two separate variables, not one flat band stack.

**Storage sizing**: decide full-record vs. defined-subset archiving before implementation.

---

## Part I: Using the ensemble — critical usage rule (unchanged)

**Never average the 100 realizations to produce a final delivered value.**

**Correct pattern for any nonlinear derived statistic**:
```
for each of the 100 realizations:
    compute the statistic on that realization alone
→ 100 values → summarize (median + 5th-95th percentile) AFTER computing per-realization
```
`summarize(f(realization)) ≠ f(summarize(realizations))` for nonlinear f.

**Two legitimate final uses**: (1) hand the full ensemble to a downstream impact model
capable of running ensembles; (2) compute the statistic of interest per realization, then
report the resulting distribution.

---

## Part J: Evaluation framework

### Deterministic product
Unchanged: POD/FAR/CSI/FB, QQ-plots, point-wise RMSE/MAE.

### Stochastic product
- Realization-averaged POD/FAR/CSI/FB
- Reliability diagram / Brier score for P(wet) calibration
- PIT / rank histogram for QRF amount distributions
- Extreme-event coverage check (5th–95th percentile band vs. true station count; target
  ≈90% coverage)
- Trend-uncertainty workflow: per-realization Mann-Kendall + Theil-Sen, or Poisson/
  Negative-Binomial regression on annual counts (not OLS — count data)
- Threshold-definition leakage check: holdout discipline if using a percentile-based
  extreme threshold

### New, required given the persistence removal
- **Quantify, don't just acknowledge, the spell-length gap**: compute CDD/CWD (max
  consecutive dry/wet days) from the stochastic ensemble and compare directly against
  station-observed CDD/CWD. Report the actual magnitude of underestimation (expected,
  since iid daily Bernoulli draws produce a geometric-like spell-length distribution that
  underrepresents the true long tail) — this turns "spell length is a known limitation"
  into a measured, documented gap size, which matters for anyone downstream deciding
  whether this product is fit for their specific use case.
- Same treatment for Rx5day-style multi-day totals if relevant to target use cases.

### Trend caveat (unchanged)
Within-record trend detection, not formal climate-change attribution.

---

## Part K: Complete scoped-limitations list (final, compiled)

1. **Temporal persistence — not modeled at all, full acknowledgment.** This now covers
   both dimensions that were previously handled separately:
   - **Occurrence persistence (spell length: CDD/CWD, wet/dry spell duration)** — no
     longer addressed by any explicit mechanism (previously handled by the now-removed
     Markov layer). Each realization's daily wet/dry draw is independent given the day's
     covariates; spell lengths will read shorter than reality, consequence of iid
     Bernoulli sampling.
   - **Magnitude persistence (day-to-day amount dependence)** — never addressed; explored
     and set aside earlier (copula/AR-on-transformed-magnitude, threshold-exceedance/
     r-Pareto, generative alternatives — none justified by a named downstream requirement).
   - What *is* still present, and should not be confused with designed persistence: the
     shared predictor's own real synoptic-scale autocorrelation flows through into
     P(wet|X_t) naturally, and the previous-day-state feature (Part B) provides an
     aggregate, non-individualized signal via the deterministic backbone. Neither
     constitutes designed, individual-realization persistence — both are "inherited,"
     consistent with how this category has been treated throughout this project.
2. **Cross-variable temporal consistency beyond same-day** — not modeled.
3. **Occurrence spatial coherence is inherited-only, not explicitly corrected** — cost of
   scoping the shuffle away from occurrence. Cheap partial fix remains on the table if
   needed: a shared Bernoulli random seed per day across stations, not yet built.
4. **Multi-day accumulated totals** (Rx5day-style) — variance understated.
5. **Y-proxy substitution weakest in convective/patchy precip regimes.**
6. **Sub-pixel collapse within parent coarse cells** — accepted resolution-limit artifact.
7. **QRF tail thinning** for rare/extreme predictor combinations — Gamma+GPD hybrid-tail
   remains the natural, not-yet-built upgrade path.
8. **Trend detection is within-record only**, not formal attribution.
9. **Deterministic product's single-day value is a best point-guess**, not verified
   ground truth for that specific day.
10. **Rare tmin > tmax violations possible post-shuffle** — pending post-hoc validation.
11. **Random cascades** — rejected entirely; not part of architecture.
12. **Storage/compute scaling** for full N=100 realization archives — sizing decision
    required before implementation.

---

## Part L: Open items for next session / scripting phase
- Implement `schaake_shuffle_day` (Part F) with wet-day-filtered Y-pool for precip magnitude.
- Decide storage subset vs. full-record archiving (Part H).
- Decide whether to build the shared-Bernoulli-seed partial fix for occurrence spatial
  coherence (limitation #3) or leave fully out of scope.
- **Run the CDD/CWD quantification** (Part J) as soon as a first implementation exists —
  turns limitation #1 from a qualitative acknowledgment into a measured, documented gap.
- Build and run the full evaluation stack (Part J) on the composed system output.
- **Engineering note (simplification from this revision)**: occurrence generation for the
  stochastic product is now fully parallelizable across time, realizations, and pixels —
  no sequential-per-realization constraint remains, since both the Markov layer and the
  per-realization-recursive previous-day feature have been removed/resolved.
