# %%
# %%
# Daily In-Situ Temperature Kriging — V4: Hofstra/Haylock Climatology+Anomaly-KED Architecture
#
# V4 changelog vs V3 (full discussion in chat; summary here for anyone picking this
# notebook up cold):
#   D1  Anisotropy scaling/angle (locked in Section 9b) is now actually threaded
#       into the CV evaluator and the production kriging call. In V3 these were
#       computed and printed but never passed to PyKrige -- every krige, including
#       CV, was silently running isotropic regardless of the anisotropy lock.
#   D2  Bounded-variogram-fit machinery removed. It was dead code in V3 (computed
#       a `bounds` tuple but `fit_variogram_lm` never accepted/used it -- the LM
#       fit was always unconstrained). Kept unconstrained on purpose: empirically
#       stable (tmax ~100-190km, tmin ~85km range), no degenerate-range behavior
#       observed once the binning bug (below) was fixed.
#   D3  `run_ked_global` now uses `.filled(np.nan)` instead of `.data` on PyKrige's
#       masked-array kriging result, so failed/masked grid points come through as
#       real NaN instead of exposing whatever sentinel fill value sits under the
#       mask.
#   D4  `reproject_utm_to_template` now explicitly burns `nodata_out` into invalid
#       (NaN) pixels before writing the intermediate GeoTIFF, and passes
#       `src_nodata=nodata_out` into `reproject()`, so GDAL treats the nodata
#       boundary correctly during bilinear resampling instead of relying on raw
#       NaN arithmetic happening to behave.
#   D5  ISOTROPIC_OK locked False/False, both at 90 deg major azimuth, for BOTH
#       tmax and tmin -- reviewed directly against each variable's own directional
#       variogram plot, not assumed from tmax.
#   D6  Search-radius concept dropped entirely. All stations are used for every
#       day's KED solve (global UniversalKriging, no local/blockwise
#       approximation). Domain extent here (~768km) is not far from Hofstra's own
#       literature search radius for temperature (~500km) despite Nepal's much
#       sparser ~40-station network, so a further cutoff bought little. This is
#       framed as directly analogous to Hofstra's own "tested anisotropy, found
#       isotropic more appropriate" simplification call -- a locked, spatially
#       and temporally FIXED single covariance model, applied without local
#       modification, in the same one-set-of-assumptions-for-the-whole-domain
#       spirit as the rest of Step 5's design.
#   D7  Removed with D6: this also eliminates the mismatch where CV evaluated the
#       radius cutoff exactly per held-out point but production only approximated
#       it per block-center -- there is no cutoff and no block-wise step left, so
#       CV and production are now the literal same unrestricted-KED code path.
#   D8  MIN_STATIONS_FOR_CV_FOLD left shared between the KED and OK (no-drift)
#       comparison arms in Section 12 -- reviewed, judged not worth the added
#       complexity of separate thresholds at threshold=4.
#   D9  [NEW] All preprocessing/processing (climatology TPS prediction, UTM grid
#       construction, KED kriging grid, UTM<->WGS84 reprojection) now runs on a
#       slightly larger/buffered WGS84 grid (template_wgs84_buffered_path) instead
#       of the original delivery-extent template. Every raster that gets written
#       to disk is cropped back down to the ORIGINAL template's extent
#       (template_wgs84_path) only at the point of saving, via the new
#       clip_to_output_grid() helper (Section 4). This addresses reprojection
#       blemishes that were showing up near the domain edge: both the TPS
#       climatology fit (which can misbehave right at the edge of the station
#       convex hull) and the UTM->WGS84 bilinear resampling step (D4) now have
#       real interior coverage on all sides of what used to be the boundary,
#       and that boundary itself is discarded before anything is saved.
#
# Method (architecture doc Section 2), unchanged by the above:
#   1. Climatology stage (once): station monthly climatological means -> trivariate
#      (lon, lat, elevation) thin-plate spline -> 12 monthly climatology grids.
#   2. Anomaly stage (daily): anomaly = raw value - station's climatological monthly
#      mean. Anomalies kriged with Kriging-with-External-Drift (KED, elevation as
#      drift), using ONE variogram calibrated once on the full pooled anomaly record.
#   3. Reconstruction: daily grid = climatology grid (that month) + kriged anomaly grid.
#
# ─────────────────────────────────────────────────────────────────────────────────
# ASSUMPTIONS / OPEN ITEMS RESOLVED WITH DEFAULTS (flagged "ASSUMPTION:" inline).
# Search this file for "ASSUMPTION:" to find every place a judgment call was made
# in the absence of a spec in the architecture doc. The big ones:
#
#   - Data paths/schema (open item 1-2): reused directly from the V2 script. Checked
#   - Fallback threshold for Step 7 (open item 5): MIN_STATIONS_FOR_KED below is a
#     placeholder — TBD per the architecture doc itself. A diagnostic print of the
#     real per-day station-count distribution is produced so you can set this
#     properly before the first full production run.
#   - Search radius (Step 5a, architecture doc's own open concept) is NOT
#     implemented at all (see D6 above) -- every day's KED solve uses all
#     reporting stations, unrestricted. This is a deliberate simplification, not
#     an oversight; see D6/D7 in the changelog above for the reasoning.
#   - The pooled global variogram (Step 5) is NOT built by dumping all ~9,000-day x
#     ~40-station rows into a single naive pairwise semivariogram (that both mixes
#     different dates' observations into "spatial" pairs, which is wrong, and is
#     computationally O(N^2) over ~365k rows, which is infeasible). Instead it's
#     built as a proper pooled/temporally-replicated semivariogram: for each day,
#     same-day station pairs contribute a squared-difference at their (fixed) spatial
#     lag; these accumulate across all days into per-lag-bin sums before dividing.
#     This is the standard estimator for a variogram pooled across temporal
#     replicates of one (assumed) stationary spatial process.
#   - [NEW, D9] Buffered processing grid (open item, your request): the DEM at
#     `dem_path` is assumed to already cover the FULL extent of the buffered
#     template (template_wgs84_buffered_path), not just the original template's
#     extent -- otherwise the DEM reprojection onto the buffered grid will itself
#     introduce a new ring of NaN right at the old boundary, defeating the point.
#     Worth a quick visual check before a full run.
#
# ─────────────────────────────────────────────────────────────────────────────────
# V4.1 -- LINUX / LOGGING PORT (this pass). Infrastructure only -- no science,
# parameters, thresholds, or kriging logic touched anywhere below. Full list:
#   L1  Added a real `logging` setup (console handler + a timestamped run-log
#       file under output_dir, Section 1) and swapped every print() for
#       logger.info/.warning/.error. This also fixes a pre-existing bug: Section
#       12's "skip, file already exists" branch already called
#       `logger.info(...)`, but `logger` was never defined anywhere in V4 -- that
#       line would have raised NameError the first time a day was actually
#       skipped on a re-run. It now works as originally intended.
#   L2  Added log_step() banners marking the start of each major section, and
#       tqdm progress bars (optional dependency -- falls back to a plain
#       pass-through iterator if tqdm isn't installed, so nothing breaks either
#       way) on the loops that actually run long: the pooled-semivariogram
#       day-by-day accumulation (Section 9 / compute_pooled_semivariogram), the
#       LOOCV date loop (Section 10 / evaluate_loo_ked), the elevation-screen
#       date loop (Section 8), and -- the big one -- the daily production
#       kriging loop (Section 12).
#   L3  Removed all matplotlib plotting: Section 6's climatology QA imshow, and
#       Section 9's experimental-vs-fitted-model plot plus its directional/
#       anisotropy-check plot -- including the anisotropy check's own repeated
#       compute_pooled_semivariogram() calls (one per azimuth in
#       ANISOTROPY_AZIMUTHS_DEG) that existed only to feed that second plot.
#       Section 9b's ISOTROPIC_OK / ANISOTROPY_SCALING / ANISOTROPY_ANGLE_
#       PYKRIGE_DEG are hardcoded there directly, not read from these plots, so
#       nothing downstream depends on the removed block -- it also meaningfully
#       speeds up Section 9 on an unattended run, since it was otherwise
#       recomputing the full pooled semivariogram 4 extra times per variable
#       solely to draw a chart no one would see. `import matplotlib.pyplot as
#       plt` removed as a result. ANISOTROPY_AZIMUTHS_DEG / ANISOTROPY_ATOL_DEG
#       in Section 1 are left defined (now unused) in case you want to reinstate
#       the visual check interactively later.
#   L4  Section 9's end-of-loop summary no longer says "review the plots above"
#       (there are none now); reworded to reflect that this is an unattended,
#       already-locked run -- Section 9b's LOCKED_MODEL/LOCKED_PARAMS remain the
#       single source of truth, unchanged.
#   L5  Section 12's `except Exception as e:` around the daily KED solve
#       previously discarded `e` entirely (fell back to climatology-only with
#       only a bare method-string flag). It now also logs the actual exception,
#       so a multi-day unattended run leaves a diagnosable trail. The fallback
#       behavior itself is byte-for-byte unchanged.
#   No path changes were needed for Linux -- every path in Section 1 already
#   uses forward slashes (e.g. r"../Data/..."), which work fine on Linux;
#   Windows-style backslashes were never used here to begin with.
#
# ─────────────────────────────────────────────────────────────────────────────────
# V4.2 -- SECTION 12 PARALLELIZATION (this pass). Only Section 12 (the daily KED
# kriging + raster-write loop) and its two small dependencies are touched;
# per-task computational logic is byte-for-byte identical to V4.1, just
# dispatched across worker processes instead of run one day-variable pair at a
# time in a single process.
#   P1  Section 12's daily loop is now driven by a fork-based
#       concurrent.futures.ProcessPoolExecutor (config: N_WORKERS, Section 1),
#       not threads. Why processes, not threads: the per-task work here is
#       CPU-bound Python -- PyKrige's UniversalKriging matrix build/solve,
#       numpy reconstruction arithmetic, GDAL reprojection via rasterio -- code
#       that either runs under the GIL directly or only inconsistently
#       releases it. Threads would leave most of that serialized on one core
#       no matter how many you start; separate processes sidestep the GIL
#       entirely and actually use every core. Each (date, variable) pair is a
#       fully independent task (its own output file, no shared mutable state)
#       -- an ideal, embarrassingly-parallel fit for a process pool. `fork`
#       (not `spawn`) is used deliberately: forked workers inherit the
#       parent's already-built anomalies_df / climatology grids / DEM grids /
#       locked variogram via copy-on-write, so none of that gets re-pickled or
#       recomputed per worker. This assumes Linux/macOS (the point of this
#       whole port); if `fork` isn't available the script logs a warning and
#       runs Section 12 sequentially in one process instead of risking a
#       broken `spawn`-based pool in a plain top-to-bottom script with no
#       `if __name__ == "__main__":` guard.
#   P2  Fixed a race condition this parallelization would otherwise have
#       introduced: `reproject_utm_to_template`'s intermediate GeoTIFF used to
#       be one fixed, shared filename (`_anomaly_utm_tmp.tif`). That was fine
#       sequentially but multiple worker processes writing/reading that same
#       path at the same time could silently corrupt each other's output.
#       Each call now uses its own uniquely-named temp file (pid + a random
#       token) and removes it afterward.
#   P3  Worker processes no longer call `logger` directly -- a
#       logging.FileHandler shared across forked processes can interleave or
#       garble writes to the run-log file with no extra synchronization. Each
#       worker instead returns a small result dict; the parent process (the
#       only one holding the real logger) does all the logging -- including
#       the KED-failure exception message (same message as V4.1's L5) -- once
#       results come back, in the same order the tasks were submitted.
#   P4  OMP_NUM_THREADS / OPENBLAS_NUM_THREADS / MKL_NUM_THREADS /
#       VECLIB_MAXIMUM_THREADS / NUMEXPR_NUM_THREADS are all set to "1" (via
#       os.environ.setdefault, so your own environment's choices win if
#       already set) before numpy is imported in Section 0. Without this,
#       each of the N_WORKERS processes' own numpy/BLAS calls could ALSO try
#       to spawn several internal threads, oversubscribing the machine and
#       making the parallel run slower than the original, not faster.
#   P5  Everything else in Section 12 -- the KED-vs-climatology-fallback
#       decision, the kriging call itself, clip_to_output_grid, the output
#       profile/dtype/nodata handling, and the log-row schema written to
#       LOG_CSV -- is unchanged.

# %%
# ── Section 0: Imports ────────────────────────────────────────────────────────────
 
import os

# [PARALLEL, P4] Must be set before numpy/scipy are imported to reliably take
# effect -- see the V4.2 changelog above.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import sys
import math
import logging
import warnings
import uuid
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
import numpy as np
import pandas as pd
from pyproj import Proj
from scipy.optimize import curve_fit
from scipy.spatial.distance import pdist
from scipy.spatial import cKDTree
from scipy.interpolate import RBFInterpolator
import rasterio
from rasterio.enums import Resampling
from rasterio.warp import calculate_default_transform, reproject
from pykrige.uk import UniversalKriging
from pykrige.ok import OrdinaryKriging  # only used for the no-drift CV comparison, Step 5b

# [LINUX PORT, L2] tqdm is an optional dependency, used only for progress bars on
# the genuinely long-running loops (Sections 8, 9, 10/11, 12). If it isn't
# installed, tqdm(...) below just falls back to returning its iterable
# unchanged -- step-level logging (log_step / logger.info) works either way.
# Install with: pip install tqdm
try:
    from tqdm import tqdm
    _TQDM_AVAILABLE = True
except ImportError:
    _TQDM_AVAILABLE = False
    def tqdm(iterable, *args, **kwargs):
        return iterable

# %%
# ── Section 1: User Configuration ─────────────────────────────────────────────────
 
# ── Per-variable input files ─────────────────
# ASSUMPTION: "train" stations only. The architecture doc scopes held-out
# test-parquet validation to a separate script (Decisions Log #13),
 
VARIABLE_CONFIGS = {
    "tmax": {
        "insitu_pq": r"../Data_Temp/Insitu/Cleaned/tmax_cleaned.parquet",
        "train_csv": r"../Data_Temp/Processed/Split_stations/tmax_train_stations.csv",
        "value_col": "max_temp",
    },
    "tmin": {
        "insitu_pq": r"../Data_Temp/Insitu/Cleaned/tmin_cleaned.parquet",
        "train_csv": r"../Data_Temp/Processed/Split_stations/tmin_train_stations.csv",
        "value_col": "min_temp",
    },
}
 
# ── Grid template & DEM (reused verbatim from the V2 script) ──────────────────────
# template_wgs84_path is the ORIGINAL, delivery-extent grid. It now does double
# duty as the CLIP TARGET only (see clip_to_output_grid, Section 4) -- every
# raster written to disk is cropped back down to this extent right before saving.
template_wgs84_path = r"../Data/Datasets/DEM_1km/DEM_1km_elevation_max.tif"

# [NEW, D9] template_wgs84_buffered_path is a slightly larger/padded version of
# the same grid (same CRS/resolution/pixel alignment as template_wgs84_path,
# just extended on all sides) -- supply your buffered raster's path here. ALL
# preprocessing/processing (climatology TPS prediction, the UTM kriging grid,
# the KED anomaly grid, the UTM<->WGS84 reprojection) now runs on this larger
# grid instead of the original template, so edge effects from both the TPS fit
# and the reprojection step land in the buffer region that gets discarded
# before saving, not in the delivered raster.
template_wgs84_buffered_path = r"../Data/Datasets/DEM_1km/DEM_1km_elevation_mean_bigger_grid.tif"

dem_path = r"../Data/Datasets/DEM_1km/DEM_1km_elevation_mean.tif"
 
# ── Calibration & processing periods (reused verbatim) ────────────────────────────
calibration_start_year = 2001
calibration_end_year = 2025
krig_start_day = pd.Timestamp("2001-01-01")
krig_end_day = pd.Timestamp("2025-12-31")
 
# ── Step 2: monthly climatology completeness gate ──────────────────────────────────
MONTH_COMPLETENESS_THRESHOLD = 0.80  # architecture doc's stated default
# ASSUMPTION: the architecture doc references this threshold but not the per-year
# "sufficient within-month reporting" cutoff it's applied to. Defined here as: a
# given station-year-month counts as "reporting" if it has at least this many days
# of data. threshold is then applied to the FRACTION of calendar years (within
# calibration_start_year..calibration_end_year) that qualify.
MIN_DAYS_PER_MONTH_YEAR = 24
# If a month has fewer than this many qualifying stations after the completeness
# gate, the gate is relaxed for that month (see Section 6) rather than leaving a
# hole in the climatology surface — flagged loudly when it happens.
MIN_STATIONS_FOR_CLIMATOLOGY_TPS = 10
 
# ── Step 3: thin-plate-spline smoothing selection ──────────────────────────────────
# ASSUMPTION: candidate smoothing values are generated per (variable, month) as
# multiples of that month's climatological-value variance, since RBFInterpolator's
# `smoothing` parameter is in squared-value units and an absolute grid would not
# transfer sensibly between tmax and tmin or between wet/dry-season variance levels.
RBF_SMOOTHING_MULTIPLIERS = np.array([0.0, 1e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0, 3.0, 10.0])
 
# ── Step 4.5: elevation screen (diagnostic only, non-gating) ───────────────────────
ELEV_SCREEN_SAMPLE_FRAC = 0.20
MIN_STATIONS_FOR_ELEV_SCREEN_DAY = 5
 
# ── Step 5: pooled variogram ────────────────────────────────────────────────────────
# NOTE (D2): variogram fit is deliberately UNCONSTRAINED (matches Hofstra's method
# literally, per architecture Step 5 -- "Range is not pre-fixed as a separate
# step"). An earlier version of this notebook computed a `bounds` tuple intending
# to cap the fitted range at a multiple of domain_extent, but the underlying fit
# function never actually accepted/used it -- it was dead code. Since fixing the
# real bug in compute_pooled_semivariogram's binning (`valid_dist`, see below),
# unconstrained LM has produced stable, physically reasonable ranges (tmax
# ~100-190km, tmin ~85km on a ~768km domain), so no bounding is applied.
MIN_PAIRS_PER_BIN = 30  # Webster & Oliver (2007) stable-gamma(h) convention, reused from V2
MODEL_TRIAL_ORDER = ["exponential", "gaussian", "spherical"]
ANISOTROPY_AZIMUTHS_DEG = [0, 45, 90, 135]
# ASSUMPTION: 22.5 deg is half of the 45 deg azimuth spacing above (i.e. adjacent
# directional bins meet edge-to-edge with no gap/overlap). Not specified in the doc.
ANISOTROPY_ATOL_DEG = 22.5
 
# ── Step 5a/5b: embedded LOOCV ──────────────────────────────────────────────────────
# ASSUMPTION: literal leave-one-station-out CV on all ~9,125 days x ~40 stations is
# unnecessary for a stable RMSE estimate and expensive. A stratified random sample
# of days (even across calendar months, matching the Step 4.5 convention) is used
# instead. Raise this if you want a slower but more exhaustive CV.
CV_N_SAMPLE_DAYS = 200
MIN_STATIONS_FOR_CV_FOLD = 4  # shared by both the KED and no-drift/OK CV arms (D8):
                               # reviewed and left as-is rather than adding a separate,
                               # stricter threshold for KED's extra drift parameter.
RANDOM_SEED = 64
 
# ── Step 7: fallback threshold ──────────────────────────────────────────────────────
# ASSUMPTION / PLACEHOLDER — explicitly "TBD" in the architecture doc (open item 5).
# This is a genuine minimum for a solvable KED system (nugget+range+1 drift
# coefficient), not a statistically-tuned value. Section 4's diagnostic print of the
# real per-day station-count distribution is meant to inform a better number before
# the first full production run.
MIN_STATIONS_FOR_KED = 4
 
# ── Step 5a: search radius — REMOVED (D6) ───────────────────────────────────────
# ASSUMPTION / DECISION: the architecture doc's Step 5a ("set search radius
# directly, Hofstra-style, via CV comparison") is deliberately NOT implemented.
# Domain extent here (~768km, computed in Section 9) is not far below Hofstra's
# own literature search radius for temperature (~500km), despite Nepal's network
# being far sparser (~40 stations vs. hundreds in Europe) -- restricting the
# neighborhood further bought little in testing. All stations are used for every
# day's KED solve; at n~40 the linear solve is trivial regardless of the
# neighborhood size, so there is no performance reason to restrict it either.
# This also removes PyKrige's UniversalKriging-has-no-n_closest_points limitation
# as an issue entirely, and eliminates the CV-vs-production radius-cutoff
# mismatch that an earlier version of this notebook flagged (no cutoff exists in
# either place anymore, so there is nothing left to mismatch).

# ── Step 6-8: Section 12 parallel execution ─────────────────────────────────────
# [PARALLEL] Number of worker PROCESSES for the daily KED kriging + raster-write
# loop (Section 12) -- see the V4.2 changelog at the top of this file for why
# processes (not threads) were chosen, and why BLAS thread pools are capped to 1
# per process right above the imports in Section 0. Defaults to "all cores but
# one", leaving some headroom for the OS/other work; override directly, or set
# to 1 to reproduce the original strictly-sequential behavior (e.g. for
# debugging, or on a platform without the `fork` start method).
N_WORKERS = 10
 
# ── Output ────────────────────────────────────────────────────────────────────────
# ASSUMPTION (open item 4): new sibling folder so this run never overwrites V2's
# "Daily" outputs. Rename freely.
output_dir = r"../Data_Temp/Insitu/KrigingDaily_1Km"
temp_dir = r"../Data_Temp/Insitu/KrigingDaily_1Km/Temp"
diag_dir = r"../Data_Temp/Insitu/KrigingDaily_1Km/Diagnostics"
nodata_out = -9999
 
UTM_PROJ = "+proj=utm +zone=45 +ellps=WGS84 +datum=WGS84 +units=m +no_defs"
DST_CRS = "EPSG:32645"
MONTH_NAMES = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
 
os.makedirs(output_dir, exist_ok=True)
os.makedirs(temp_dir, exist_ok=True)
os.makedirs(diag_dir, exist_ok=True)
for v in VARIABLE_CONFIGS:
    os.makedirs(os.path.join(output_dir, v), exist_ok=True)
 
LOG_CSV = os.path.join(output_dir, "daily_kriging_log_KED.csv")
VARIOGRAM_SUMMARY_CSV = os.path.join(diag_dir, "locked_global_variogram.csv")
CLIMATOLOGY_DIAG_CSV = os.path.join(diag_dir, "climatology_tps_diagnostics.csv")
CV_FINAL_CSV = os.path.join(diag_dir, "final_loocv_validation.csv")
ELEV_SCREEN_CSV = os.path.join(diag_dir, "elevation_screen_diagnostic.csv")

# [LINUX PORT, L1] Logging setup -- console + a timestamped run-log file under
# output_dir. Placed here (not Section 0) because it needs output_dir to exist
# already. Nothing above this point changes; from here on, print() is replaced
# by logger.info/.warning/.error throughout the rest of the notebook.
RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE_TXT = os.path.join(output_dir, f"pipeline_run_{RUN_TIMESTAMP}.log")

logger = logging.getLogger("ked_kriging")
logger.setLevel(logging.INFO)
logger.propagate = False
if logger.hasHandlers():
    logger.handlers.clear()  # avoid duplicate lines if this cell is re-run in one kernel

_log_fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

_console_handler = logging.StreamHandler(sys.stdout)
_console_handler.setFormatter(_log_fmt)
logger.addHandler(_console_handler)

_file_handler = logging.FileHandler(LOG_FILE_TXT, mode="a", encoding="utf-8")
_file_handler.setFormatter(_log_fmt)
logger.addHandler(_file_handler)


def log_step(title):
    """Stepwise progress banner -- marks the start of a major pipeline section
    in both the console and the run-log file."""
    logger.info("=" * 88)
    logger.info(title)
    logger.info("=" * 88)


logger.info(f"Run log file -> {LOG_FILE_TXT}")
if not _TQDM_AVAILABLE:
    logger.warning("tqdm not installed -- progress bars will be skipped (step-level "
                    "logging still works). Install with: pip install tqdm")

rng = np.random.default_rng(RANDOM_SEED)

log_step("SECTION 1 — Configuration loaded, logging initialized")

# %%
# ── Section 2: Helper Functions ───────────────────────────────────────────────────

proj_utm = Proj(UTM_PROJ)

def coords_to_utm_km(lons, lats):
    x_m, y_m = proj_utm(lons, lats)
    return np.asarray(x_m) / 1000.0, np.asarray(y_m) / 1000.0


def coords_to_utm_m(lons, lats):
    x_m, y_m = proj_utm(lons, lats)
    return np.asarray(x_m), np.asarray(y_m)


def nearest_neighbor_distances_km(x_km, y_km):
    """True per-station nearest-neighbor distance via KD-tree (reused from V2)."""
    coords = np.column_stack([x_km, y_km])
    tree = cKDTree(coords)
    dists, _ = tree.query(coords, k=2)
    return dists[:, 1]


def network_geometry_params(x_km, y_km, min_pairs=MIN_PAIRS_PER_BIN):
    """
    Auto-derive LAG_DIST / N_LAG / domain_extent from station geometry (reused
    logic from V2 Section 5, run ONCE per variable here since there is only one
    pooled variogram, not one per season).
    """
    coords_km = np.column_stack([x_km, y_km])
    domain_extent = float(pdist(coords_km).max())
    nn_dists = nearest_neighbor_distances_km(x_km, y_km)
    median_nn = float(np.median(nn_dists))

    lag_dist = max(10.0, round(median_nn * 1.5 / 5) * 5)
    max_lag = domain_extent / 2.0
    n_lag = max(6, int(np.floor(max_lag / lag_dist)))

    return {
        "lag_dist": lag_dist,
        "n_lag": n_lag,
        "max_lag": max_lag,
        "domain_extent": domain_extent,
        "median_nn": median_nn,
    }


# ── Variogram model functions — verified against PyKrige's own internal formulas ──
# (PyKrige's internal parameter is called "psill" and IS the partial sill, matching
# what's fit here; see Section 10's note on the 'sill' vs 'psill' dict key.)
def spherical(h, nugget, psill, range_):
    h = np.asarray(h, dtype=float)
    sph = np.where(
        h <= range_,
        nugget + psill * (1.5 * (h / range_) - 0.5 * (h / range_) ** 3),
        nugget + psill,
    )
    return np.where(h == 0, nugget, sph)


def exponential(h, nugget, psill, range_):
    h = np.asarray(h, dtype=float)
    return np.where(h == 0, nugget, nugget + psill * (1 - np.exp(-h / (range_ / 3))))


def gaussian_vgm(h, nugget, psill, range_):
    h = np.asarray(h, dtype=float)
    eff_range = range_ * 4.0 / 7.0
    return np.where(h == 0, nugget, nugget + psill * (1 - np.exp(-(h ** 2) / eff_range ** 2)))


MODEL_FUNCTIONS = {"spherical": spherical, "exponential": exponential, "gaussian": gaussian_vgm}


def fit_variogram_lm(model_name, lag, gamma, n_pairs, p0):
    """
    Joint nugget/psill/range fit via Levenberg-Marquardt (architecture Step 5).
    Deliberately unconstrained (D2) -- see the note at MIN_PAIRS_PER_BIN above.

    Weighting: inverse-square distance (sigma = lag), so near-origin bins
    dominate the fit -- MIN_PAIRS_PER_BIN already floors every fitted bin's
    pair count, so weighting further by n_pairs was double-penalizing the
    near-origin bins the nugget depends on. chi2 below is computed under this
    same weighting.
    """
    func = MODEL_FUNCTIONS[model_name]
    sigma = np.maximum(np.asarray(lag, dtype=float), 1e-3)  # weight = 1/lag^2, eps avoids div-by-zero at h=0
    popt, _ = curve_fit(func, lag, gamma, p0=p0, sigma=sigma, absolute_sigma=True,
                         method="lm", maxfev=20000)
    gamma_model = func(lag, *popt)
    chi2 = float(np.sum(((gamma - gamma_model) / sigma) ** 2))
    return {"nugget": float(popt[0]), "psill": float(popt[1]), "range_km": float(popt[2]),
            "chi2": chi2, "gamma_model": gamma_model}


def build_station_pair_table(station_xy_km_df):
    """
    Precompute the (time-invariant) upper-triangular distance & azimuth for every
    unique station pair, once. Reused by the pooled-variogram accumulator below for
    both the omnidirectional fit and every directional anisotropy check.
    """
    ids = station_xy_km_df.index.to_numpy()
    xs = station_xy_km_df["x_km"].to_numpy()
    ys = station_xy_km_df["y_km"].to_numpy()
    iu, ju = np.triu_indices(len(ids), k=1)
    dx = xs[ju] - xs[iu]
    dy = ys[ju] - ys[iu]
    dist_km = np.sqrt(dx ** 2 + dy ** 2)
    azimuth_deg = np.degrees(np.arctan2(dx, dy)) % 180.0  # 0-180, direction-agnostic
    return {
        "id_i": ids[iu], "id_j": ids[ju],
        "pos_i": iu, "pos_j": ju,          # positions into `ids`, for fast reindexing
        "ids": ids,
        "dist_km": dist_km, "azimuth_deg": azimuth_deg,
    }


def compute_pooled_semivariogram(long_df, pair_table, lag_dist, n_lag,
                                  azimuth_deg=None, atol_deg=None):
    """
    Pooled/temporally-replicated experimental semivariogram (architecture Step 5:
    "all daily anomaly residuals ... pooled into one dataset").

    NOT implemented as a single giant pairwise gamv() call across every
    station-day row: that would (a) be O(N^2) over ~365,000 rows, infeasible, and
    (b) silently pair one date's station A against a DIFFERENT date's station B as
    if they were a valid spatial pair, which is not meaningful.

    Instead: station-pair distance/azimuth are fixed (from `pair_table`); for each
    day, only SAME-DAY reporting pairs contribute a squared difference into their
    pair's lag bin; sums accumulate across all days before dividing — the standard
    estimator for a variogram pooled across temporal replicates of one (assumed)
    stationary spatial process.

    long_df: columns ['date', 'index_no', 'anomaly'] (or 'value' for non-anomaly use)
    """
    value_col = "anomaly" if "anomaly" in long_df.columns else "value"
    ids = pair_table["ids"]
    id_pos = {sid: i for i, sid in enumerate(ids)}
    n_stations = len(ids)
    pos_i, pos_j = pair_table["pos_i"], pair_table["pos_j"]
    dist_km = pair_table["dist_km"]

    edges = np.arange(0, n_lag + 1) * lag_dist
    pair_bin = np.digitize(dist_km, edges[1:-1])  # 0..n_lag-1 valid; n_lag = beyond range
    # FIX (binning bug): this must filter on true distance vs. the outer edge, not
    # on the bin index -- `pair_bin < n_lag` never actually excluded anything due
    # to how np.digitize assigns the last bin, silently merging all far-distance
    # pairs into one oversized last bin.
    valid_dist = dist_km < edges[-1]

    if azimuth_deg is not None:
        az = pair_table["azimuth_deg"]
        target = azimuth_deg % 180.0
        ang_diff = np.abs(((az - target + 90.0) % 180.0) - 90.0)
        valid_dir = ang_diff <= atol_deg
    else:
        valid_dir = np.ones_like(valid_dist, dtype=bool)

    keep = valid_dist & valid_dir
    pos_i_k, pos_j_k = pos_i[keep], pos_j[keep]
    bin_k = pair_bin[keep]
    dist_k = dist_km[keep]

    sumsq = np.zeros(n_lag)
    sumdist = np.zeros(n_lag)
    npairs = np.zeros(n_lag, dtype=np.int64)

    # [LINUX PORT, L2] Progress bar over the per-day accumulation loop -- this is
    # the part of the pooled-variogram build that scales with the full
    # ~9,000+ day record. Loop body below is untouched.
    n_days_total = long_df["date"].nunique()
    day_iterator = tqdm(long_df.groupby("date"), total=n_days_total,
                         desc="Pooled semivariogram (days)", leave=False)
    for _, day_rows in day_iterator:
        dense = np.full(n_stations, np.nan)
        idx = day_rows["index_no"].map(id_pos).dropna().astype(int)
        dense[idx.to_numpy()] = day_rows.loc[idx.index, value_col].to_numpy()

        diffs = dense[pos_j_k] - dense[pos_i_k]
        present = ~np.isnan(diffs)
        if not present.any():
            continue
        b = bin_k[present]
        np.add.at(sumsq, b, diffs[present] ** 2)
        np.add.at(npairs, b, 1)
        np.add.at(sumdist, b, dist_k[present])

    with np.errstate(invalid="ignore", divide="ignore"):
        gamma = 0.5 * sumsq / np.maximum(npairs, 1)
        lag_center = sumdist / np.maximum(npairs, 1)

    return lag_center, gamma, npairs


def weighted_chi_square(gamma_exp, gamma_model, n_pairs):
    sigma = 1.0 / np.sqrt(np.maximum(n_pairs, 1))
    return float(np.sum(((gamma_exp - gamma_model) / sigma) ** 2))


def sample_stratified_dates(dates, frac, seed=RANDOM_SEED, n_target=None):
    """Random sample of dates, stratified evenly across calendar month."""
    dates = pd.DatetimeIndex(sorted(pd.unique(dates)))
    local_rng = np.random.default_rng(seed)
    out = []
    for m in range(1, 13):
        month_dates = dates[dates.month == m]
        if len(month_dates) == 0:
            continue
        if n_target is not None:
            k = max(1, int(round(n_target / 12)))
        else:
            k = max(1, int(round(len(month_dates) * frac)))
        k = min(k, len(month_dates))
        picked = local_rng.choice(month_dates, size=k, replace=False)
        out.extend(picked)
    return pd.DatetimeIndex(sorted(out))


def standardize(coords):
    mu = coords.mean(axis=0)
    sd = coords.std(axis=0)
    sd = np.where(sd < 1e-9, 1.0, sd)
    return (coords - mu) / sd, mu, sd


def loocv_rbf_smoothing(coords_std, values, candidates):
    """Manual leave-one-out CV grid search over TPS smoothing (architecture Step 3:
    'Selected via leave-one-out CV grid search over candidate smoothing values')."""
    n = len(values)
    best_smoothing, best_rmse = candidates[0], np.inf
    diagnostics = []
    for s in candidates:
        errs = np.empty(n)
        for i in range(n):
            mask = np.arange(n) != i
            try:
                rbf_i = RBFInterpolator(coords_std[mask], values[mask],
                                         kernel="thin_plate_spline", smoothing=s)
                errs[i] = rbf_i(coords_std[i:i + 1])[0] - values[i]
            except Exception:
                errs[i] = np.nan
        rmse = float(np.sqrt(np.nanmean(errs ** 2)))
        diagnostics.append({"smoothing": s, "loocv_rmse": rmse})
        if rmse < best_rmse:
            best_rmse, best_smoothing = rmse, s
    return best_smoothing, best_rmse, pd.DataFrame(diagnostics)

# %%
# ── Section 3: Load & Preprocess Daily Station Data (Step 1) ──────────────────────

log_step("SECTION 3 / STEP 1 — Load & Preprocess Daily Station Data")


def load_variable_data(var_name, cfg):
    df = pd.read_parquet(cfg["insitu_pq"])

    rename_map = {cfg["value_col"]: "insitu_temp"}
    if "lat" in df.columns:
        rename_map["lat"] = "y"
    if "long" in df.columns:
        rename_map["long"] = "x"
    if "elevation" in df.columns:
        rename_map["elevation"] = "Elevation"
    df = df.rename(columns=rename_map)

    required_cols = {"x", "y", "Elevation", "insitu_temp", "index_no", "year", "month", "day"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"[{var_name}] parquet at {cfg['insitu_pq']} is missing expected "
                          f"column(s) after rename: {sorted(missing)}. Available columns: "
                          f"{sorted(df.columns)}")

    df["date"] = pd.to_datetime(df[["year", "month", "day"]])

    n_before = len(df)
    df = df.dropna(subset=["insitu_temp", "Elevation"]).copy()
    if len(df) < n_before:
        logger.info(f"[{var_name}] Dropped {n_before - len(df)} row(s) with missing insitu_temp/Elevation.")

    train_ids = set(pd.read_csv(cfg["train_csv"])["index_no"])
    train_daily = df[df["index_no"].isin(train_ids)].copy()

    n_train_stations = train_daily["index_no"].nunique()
    logger.info(f"[{var_name}] Total unique train stations: {n_train_stations}")

    return {"train_daily": train_daily, "n_train_stations": n_train_stations}


insitu_data = {v: load_variable_data(v, cfg) for v, cfg in VARIABLE_CONFIGS.items()}

# %%
# ── Section 4: Grid Setup ──────────────────────────────────────────────────────────
# [D9] Two WGS84 grids now exist, not one:
#   (a) tmpl_* / grid_x / grid_y / dem_grid_wgs84 / dem_grid_utm -- the BUFFERED
#       processing grid (template_wgs84_buffered_path). ALL interpolation/
#       reprojection work in Sections 5-12 runs on this grid, unchanged from
#       before except that it's now built from the buffered raster instead of
#       the original one. Every downstream section still references these exact
#       variable names, so nothing else in the notebook needed to change.
#   (b) tmpl_*_out -- the ORIGINAL, delivery-extent grid (template_wgs84_path).
#       Used ONLY as the clip target in clip_to_output_grid(), called right
#       before each raster is written to disk (Sections 6 and 12).
#
# Two output grids were already needed and are NOT the same thing (unchanged from
# before):
#   (a) WGS84 template grid (CHIRTS resolution/extent) — climatology TPS operates
#       directly in (lon, lat, elevation), per the architecture's literal
#       trivariate-spline description, and the FINAL daily product is written on
#       this grid (now: clipped down to it from the buffered grid).
#   (b) UTM grid (meters) — the pooled variogram / KED anomaly stage needs a
#       projected, equal-distance CRS for a meaningful Euclidean-distance
#       covariance model; this is the same UTM-45N convention V2 used.
# The DEM is reprojected (bilinear, reused convention from V2) onto BOTH (a) and (b),
# both now built from the buffered raster.

log_step("SECTION 4 — Grid Setup")

with rasterio.open(template_wgs84_buffered_path) as tmp:
    tmpl_profile = tmp.profile.copy()
    # Output profile always float32 with our own nodata value -- NOT necessarily
    # the template's native dtype/nodata (e.g. if CHIRTS is stored as int16, using
    # it as-is for our OWN float output would silently truncate every value).
    tmpl_profile.update(dtype="float32", nodata=nodata_out, count=1)
    tmpl_transform = tmp.transform
    tmpl_crs = tmp.crs
    tmpl_width, tmpl_height = tmp.width, tmp.height

# [NEW, D9] Original (un-buffered) extent -- used ONLY as the clip target right
# before saving, via clip_to_output_grid() below. Nothing upstream of Section 6/12
# touches these.
with rasterio.open(template_wgs84_path) as tmp_out:
    tmpl_profile_out = tmp_out.profile.copy()
    tmpl_profile_out.update(dtype="float32", nodata=nodata_out, count=1)
    tmpl_transform_out = tmp_out.transform
    tmpl_crs_out = tmp_out.crs
    tmpl_width_out, tmpl_height_out = tmp_out.width, tmp_out.height

# Pixel-center lon/lat grids for the WGS84 template (regular north-up grid, so the
# coordinate axes are separable). Buffered grid, per D9.
_lon_1d = tmpl_transform.c + (np.arange(tmpl_width) + 0.5) * tmpl_transform.a
_lat_1d = tmpl_transform.f + (np.arange(tmpl_height) + 0.5) * tmpl_transform.e
tmpl_lon2d, tmpl_lat2d = np.meshgrid(_lon_1d, _lat_1d)  # shape (tmpl_height, tmpl_width)

# DEM reprojected onto the WGS84 template grid (for climatology TPS prediction).
# Buffered grid, per D9 -- see the ASSUMPTION note in Section 0 about dem_path
# needing to cover the buffered extent.
dem_grid_wgs84 = np.full((tmpl_height, tmpl_width), np.nan, dtype="float32")
with rasterio.open(dem_path) as dem_ds:
    reproject(
        source=rasterio.band(dem_ds, 1), destination=dem_grid_wgs84,
        src_transform=dem_ds.transform, src_crs=dem_ds.crs,
        dst_transform=tmpl_transform, dst_crs=tmpl_crs,
        resampling=Resampling.bilinear,
    )

# UTM kriging grid (meters), same construction as V2 Section 7. Buffered grid,
# per D9 -- built from template_wgs84_buffered_path so the KED grid itself, and
# the reprojection back to WGS84 in Section 12, both have buffer coverage.
with rasterio.open(template_wgs84_buffered_path) as tmp:
    transform_utm, width_utm, height_utm = calculate_default_transform(
        tmp.crs, DST_CRS, tmp.width, tmp.height, *tmp.bounds
    )

grid_x = np.linspace(transform_utm.c, transform_utm.c + width_utm * transform_utm.a, width_utm)
grid_y = np.linspace(transform_utm.f + height_utm * transform_utm.e, transform_utm.f, height_utm)
# grid_y is ascending south->north; PyKrige preserves input order row-for-row, so the
# result needs np.flipud before writing to a north-up raster — same convention as V2.

# DEM reprojected onto the UTM kriging grid (external-drift raster for KED).
# Buffered grid, per D9.
dem_grid_utm = np.full((height_utm, width_utm), np.nan, dtype="float32")
with rasterio.open(dem_path) as dem_ds:
    reproject(
        source=rasterio.band(dem_ds, 1), destination=dem_grid_utm,
        src_transform=dem_ds.transform, src_crs=dem_ds.crs,
        dst_transform=transform_utm, dst_crs=DST_CRS,
        resampling=Resampling.bilinear,
    )

utm_meta = {"driver": "GTiff", "height": height_utm, "width": width_utm, "count": 1,
            "dtype": "float32", "crs": DST_CRS, "transform": transform_utm, "nodata": nodata_out}

logger.info(f"Buffered processing grid : {tmpl_height} x {tmpl_width}  ({tmpl_crs})")
logger.info(f"Original output grid     : {tmpl_height_out} x {tmpl_width_out}  ({tmpl_crs_out})")
logger.info(f"UTM grid                 : {height_utm} x {width_utm}  ({DST_CRS})")


def clip_to_output_grid(arr_buffered):
    """
    [NEW, D9] Crop/resample an array computed on the buffered processing grid
    (tmpl_transform/tmpl_crs/tmpl_width/tmpl_height) down to the ORIGINAL,
    delivery extent (tmpl_transform_out/tmpl_crs_out/tmpl_width_out/
    tmpl_height_out).

    Called right before every raster is written to disk (Sections 6 and 12).
    Both grids share the same CRS; if the buffered raster you supplied is
    pixel-aligned with the original (same resolution, same origin phase, just
    padded), this reduces to an exact crop with no resampling blur, since every
    output pixel center coincides exactly with a buffered-grid pixel center. If
    it's not perfectly aligned, bilinear resampling is used -- same convention
    as reproject_utm_to_template's nodata handling (D4), reused here for
    consistency.
    """
    src = np.where(np.isnan(arr_buffered), nodata_out, arr_buffered).astype("float32")
    out = np.full((tmpl_height_out, tmpl_width_out), np.nan, dtype="float32")
    reproject(
        source=src, destination=out,
        src_transform=tmpl_transform, src_crs=tmpl_crs,
        dst_transform=tmpl_transform_out, dst_crs=tmpl_crs_out,
        src_nodata=nodata_out, dst_nodata=np.nan,
        resampling=Resampling.bilinear,
    )
    return out

# %%
# ── Section 5: Step 2 — Long-Term Monthly Station Climatology ─────────────────────

log_step("SECTION 5 / STEP 2 — Long-Term Monthly Station Climatology")


def compute_station_climatology(var_name, train_daily):
    calib = train_daily[
        (train_daily["year"] >= calibration_start_year) & (train_daily["year"] <= calibration_end_year)
    ].copy()
    n_years_total = calibration_end_year - calibration_start_year + 1

    rows = []
    station_meta = calib.drop_duplicates("index_no").set_index("index_no")[["x", "y", "Elevation"]]

    for (sid, month), grp in calib.groupby(["index_no", "month"]):
        per_year_counts = grp.groupby("year").size()
        qualifying_years = per_year_counts[per_year_counts >= MIN_DAYS_PER_MONTH_YEAR].index
        n_qualifying = len(qualifying_years)
        completeness = n_qualifying / n_years_total

        clim_mean = (grp[grp["year"].isin(qualifying_years)]["insitu_temp"].mean()
                     if n_qualifying > 0 else np.nan)
        # Ungated fallback: mean of every reported day that station/month has, with
        # no per-year completeness restriction at all. Used only if the gated
        # criterion leaves too few stations for a usable spline (see
        # fit_climatology_grids) — without this, the relaxation there has nothing
        # real to relax TO (clim_mean is NaN for every station in a sparse month,
        # e.g. a partial-year run), and would silently produce an empty/failed fit.
        clim_mean_raw = grp["insitu_temp"].mean()
        n_days_raw = len(grp)

        rows.append({
            "index_no": sid, "month": month,
            "x": station_meta.loc[sid, "x"], "y": station_meta.loc[sid, "y"],
            "Elevation": station_meta.loc[sid, "Elevation"],
            "clim_mean": clim_mean, "clim_mean_raw": clim_mean_raw,
            "n_years_used": n_qualifying, "n_days_raw": n_days_raw, "completeness": completeness,
        })

    clim_df = pd.DataFrame(rows)
    clim_df["qualifies"] = clim_df["completeness"] >= MONTH_COMPLETENESS_THRESHOLD

    logger.info(f"[{var_name}] Station climatology — qualifying stations per month "
                f"(completeness >= {MONTH_COMPLETENESS_THRESHOLD:.0%}, "
                f">= {MIN_DAYS_PER_MONTH_YEAR} days/month/year):")
    for m in range(1, 13):
        sub = clim_df[clim_df["month"] == m]
        n_q = int(sub["qualifies"].sum())
        logger.info(f"    {MONTH_NAMES[m-1]:<4} {n_q:>3} / {sub['index_no'].nunique():>3} stations qualify")

    return resolve_climatology_relaxation(var_name, clim_df)


def resolve_climatology_relaxation(var_name, clim_df):
    """
    Per (variable, month): if fewer than MIN_STATIONS_FOR_CLIMATOLOGY_TPS stations
    pass the completeness gate, relax to the ungated raw mean for that month so the
    TPS fit (Section 6) and the anomaly construction (Section 7) both see the same,
    consistent set of usable stations -- rather than each section independently
    guessing at a fallback and potentially disagreeing with one another.
    """
    clim_df = clim_df.copy()
    clim_df["clim_mean_used"] = clim_df["clim_mean"]
    clim_df["qualifies_final"] = clim_df["qualifies"]

    for m in range(1, 13):
        month_mask = clim_df["month"] == m
        n_qualify = int((month_mask & clim_df["qualifies"]).sum())
        if n_qualify >= MIN_STATIONS_FOR_CLIMATOLOGY_TPS:
            continue
        relaxed_mask = month_mask & clim_df["clim_mean_raw"].notna()
        n_relaxed = int(relaxed_mask.sum())
        logger.warning(f"[{var_name}] {MONTH_NAMES[m-1]}: only {n_qualify} stations pass the "
                        f"completeness gate — relaxing to {n_relaxed} station(s) with ANY reported "
                        f"day this month (ungated raw mean).")
        clim_df.loc[relaxed_mask, "clim_mean_used"] = clim_df.loc[relaxed_mask, "clim_mean_raw"]
        clim_df.loc[relaxed_mask, "qualifies_final"] = True
        if n_relaxed < MIN_STATIONS_FOR_CLIMATOLOGY_TPS:
            logger.warning(f"  still only {n_relaxed} — climatology grid for "
                            f"{MONTH_NAMES[m-1]} will be unreliable regardless.")

    return clim_df


climatology_stations = {v: compute_station_climatology(v, insitu_data[v]["train_daily"])
                         for v in VARIABLE_CONFIGS}

# %%
# ── Section 6: Step 3 — Climatology Grids (trivariate thin-plate spline) ───────────

log_step("SECTION 6 / STEP 3 — Climatology Grids (Thin-Plate Spline)")


def fit_climatology_grids(var_name, clim_df):
    grids = {}
    diag_rows = []

    for month in range(1, 13):
        # qualifies_final / clim_mean_used already resolved (with relaxation applied
        # where needed) once, in resolve_climatology_relaxation -- Section 7's
        # anomaly construction uses the exact same resolved column, so the set of
        # "usable" stations can't disagree between the climatology fit and the
        # anomaly stage.
        sub = clim_df[(clim_df["month"] == month) & clim_df["qualifies_final"]]
        if len(sub) < MIN_STATIONS_FOR_CLIMATOLOGY_TPS:
            logger.warning(f"[{var_name}] {MONTH_NAMES[month-1]}: only {len(sub)} usable "
                            f"stations even after relaxation — climatology grid for this month "
                            f"will be unreliable regardless of TPS smoothing choice.")
        if len(sub) < 4:
            logger.error(f"[{var_name}] {MONTH_NAMES[month-1]}: fewer than 4 usable stations "
                         f"-- skipping this month's climatology grid (left as NaN everywhere).")
            grids[month] = np.full((tmpl_height, tmpl_width), np.nan, dtype="float32")
            diag_rows.append({"variable": var_name, "month": month, "n_stations": len(sub),
                               "chosen_smoothing": np.nan, "loocv_rmse": np.nan})
            continue

        coords = sub[["x", "y", "Elevation"]].to_numpy(dtype=float)
        values = sub["clim_mean_used"].to_numpy(dtype=float)
        coords_std, mu, sd = standardize(coords)

        candidates = RBF_SMOOTHING_MULTIPLIERS * max(np.var(values), 1e-6)
        best_smoothing, best_rmse, cv_table = loocv_rbf_smoothing(coords_std, values, candidates)

        rbf = RBFInterpolator(coords_std, values, kernel="thin_plate_spline", smoothing=best_smoothing)

        pred_coords = np.column_stack([tmpl_lon2d.ravel(), tmpl_lat2d.ravel(), dem_grid_wgs84.ravel()])
        valid_pred = np.isfinite(pred_coords).all(axis=1)
        pred_coords_std = (pred_coords - mu) / sd

        grid_flat = np.full(pred_coords.shape[0], np.nan, dtype="float32")
        grid_flat[valid_pred] = rbf(pred_coords_std[valid_pred]).astype("float32")
        grids[month] = grid_flat.reshape(tmpl_height, tmpl_width)

        diag_rows.append({"variable": var_name, "month": month, "n_stations": len(sub),
                           "chosen_smoothing": best_smoothing, "loocv_rmse": best_rmse})
        logger.info(f"[{var_name}] {MONTH_NAMES[month-1]:<4} n={len(sub):>3}  "
                    f"smoothing={best_smoothing:.4g}  LOOCV RMSE={best_rmse:.3f}")

    return grids, pd.DataFrame(diag_rows)


climatology_grid_wgs84 = {}
climatology_diag_rows = []
for v in VARIABLE_CONFIGS:
    grids, diag = fit_climatology_grids(v, climatology_stations[v])
    climatology_grid_wgs84[v] = grids
    climatology_diag_rows.append(diag)

climatology_diag_df = pd.concat(climatology_diag_rows, ignore_index=True)
climatology_diag_df.to_csv(CLIMATOLOGY_DIAG_CSV, index=False)
logger.info(f"Climatology TPS diagnostics saved -> {CLIMATOLOGY_DIAG_CSV}")

# [LINUX PORT, L3] Quick visual QA plot (one month per variable, buffered grid,
# pre-clip) removed for headless/unattended execution. The per-month diagnostics
# saved above (CLIMATOLOGY_DIAG_CSV) and the delivered GeoTIFFs below cover the
# same ground for offline review.

# ── Save Climatology Grids to GeoTIFF ──────────────────────────────────────────────
# [D9] Clipped from the buffered processing grid down to the original output
# extent (clip_to_output_grid) right before writing -- everything upstream of
# this loop is unchanged.
logger.info("Saving climatology grids to GeoTIFF...")
clim_out_dir = os.path.join(output_dir, "Climatology")
os.makedirs(clim_out_dir, exist_ok=True)

for v in VARIABLE_CONFIGS:
    var_clim_dir = os.path.join(clim_out_dir, v)
    os.makedirs(var_clim_dir, exist_ok=True)

    for month, grid in climatology_grid_wgs84[v].items():
        out_path = os.path.join(var_clim_dir, f"{v}_climatology_{month:02d}.tif")

        # [NEW, D9] crop back to the original delivery extent before writing.
        grid_clipped = clip_to_output_grid(grid)

        # Replace NaNs with the designated nodata value before writing
        grid_out = np.where(np.isnan(grid_clipped), nodata_out, grid_clipped)

        with rasterio.open(out_path, 'w', **tmpl_profile_out) as dst:
            dst.write(grid_out, 1)

logger.info(f"Done saving climatology grids to -> {clim_out_dir}")

# %%
# ── Section 7: Step 4 — Daily Anomaly Construction ─────────────────────────────────

log_step("SECTION 7 / STEP 4 — Daily Anomaly Construction")


def build_anomalies(var_name, train_daily, clim_df):
    # Uses clim_mean_used (post-relaxation) so a station/month kept in the TPS fit
    # via the relaxed fallback isn't then silently dropped here for having a NaN
    # gated clim_mean -- the two stages must agree on which station-months exist.
    clim_lookup = clim_df.set_index(["index_no", "month"])["clim_mean_used"]
    merged = train_daily.merge(
        clim_lookup.rename("clim_mean_used"), left_on=["index_no", "month"], right_index=True, how="left"
    )
    n_before = len(merged)
    merged = merged.dropna(subset=["clim_mean_used"]).copy()
    if len(merged) < n_before:
        logger.info(f"[{var_name}] Dropped {n_before - len(merged)} station-day(s) whose "
                    f"station/month never had a usable climatological mean (gated or relaxed).")
    merged["anomaly"] = merged["insitu_temp"] - merged["clim_mean_used"]
    return merged[["index_no", "x", "y", "Elevation", "date", "year", "month", "day", "anomaly"]]


anomalies_df = {v: build_anomalies(v, insitu_data[v]["train_daily"], climatology_stations[v])
                for v in VARIABLE_CONFIGS}

# Station -> UTM (meters) lookup, computed once and reused throughout (not
# reprojected fresh inside the daily loop).
station_utm_lookup = {}
for v in VARIABLE_CONFIGS:
    st = anomalies_df[v].drop_duplicates("index_no").set_index("index_no")[["x", "y"]]
    x_m, y_m = coords_to_utm_m(st["x"].to_numpy(), st["y"].to_numpy())
    station_utm_lookup[v] = pd.DataFrame({"x_m": x_m, "y_m": y_m}, index=st.index)

for v in VARIABLE_CONFIGS:
    logger.info(f"[{v}] anomaly table: {len(anomalies_df[v]):,} station-days across "
                f"{anomalies_df[v]['index_no'].nunique()} stations, "
                f"{anomalies_df[v]['date'].nunique():,} unique dates")

# %%
# ── Section 8: Step 4.5 — Elevation Screen (diagnostic only, non-gating) ──────────
# Architecture doc: "Not a gate — KED is a locked design choice regardless of this
# screen's outcome. Kept only because it's cheap and useful context."

log_step("SECTION 8 / STEP 4.5 — Elevation Screen (diagnostic only)")


def elevation_screen(var_name, anomalies):
    sample_dates = sample_stratified_dates(anomalies["date"], ELEV_SCREEN_SAMPLE_FRAC)
    rows = []
    # [LINUX PORT, L2] progress bar only -- loop body/logic unchanged.
    for d in tqdm(sample_dates, desc=f"Elevation screen ({var_name})", leave=False):
        day = anomalies[anomalies["date"] == d]
        if len(day) < MIN_STATIONS_FOR_ELEV_SCREEN_DAY:
            continue
        elev = day["Elevation"].to_numpy(dtype=float)
        anom = day["anomaly"].to_numpy(dtype=float)
        if np.std(elev) < 1e-6:
            continue
        r = np.corrcoef(elev, anom)[0, 1]
        rows.append({"date": d, "n_stations": len(day), "r2": r ** 2})
    return pd.DataFrame(rows)


elev_screen_results = {v: elevation_screen(v, anomalies_df[v]) for v in VARIABLE_CONFIGS}

elev_screen_all = pd.concat(
    [df.assign(variable=v) for v, df in elev_screen_results.items()], ignore_index=True
)
elev_screen_all.to_csv(ELEV_SCREEN_CSV, index=False)

for v in VARIABLE_CONFIGS:
    r2 = elev_screen_results[v]["r2"]
    if len(r2) > 0:
        logger.info(f"[{v}] elevation screen (diagnostic, non-gating): n_days={len(r2)}  "
                    f"median R^2={r2.median():.3f}  pct(R^2>0.20)={100*(r2>0.20).mean():.1f}%")
    else:
        logger.info(f"[{v}] elevation screen: no sampled days had enough stations to evaluate.")

# %%
# ── Section 9: Step 5 — Global Pooled Variogram + Anisotropy Check ────────────────
# ONE variogram per variable, fit once on the full pooled anomaly record, used for
# all 25 years (architecture doc, Decisions Log #6). No per-day/month/season
# refitting anywhere in this notebook. Fit is unconstrained (D2, see note near
# MIN_PAIRS_PER_BIN in Section 1).
#
# [LINUX PORT, L3] The interactive diagnostic plot (experimental vs fitted models)
# and the directional/anisotropy-check plot -- along with the extra
# compute_pooled_semivariogram() calls (one per azimuth in ANISOTROPY_AZIMUTHS_DEG)
# that existed only to feed that second plot -- have been removed for unattended
# execution. Section 9b's ISOTROPIC_OK / ANISOTROPY_SCALING / ANISOTROPY_ANGLE_
# PYKRIGE_DEG are hardcoded there directly (not read from these plots), so
# nothing downstream depends on the removed block.

log_step("SECTION 9 / STEP 5 — Global Pooled Variogram")

geom_params = {}
pair_tables = {}
pooled_omni = {}          # (lag, gamma, npairs) per variable
variogram_fit_results = {}  # per variable: {model_name: {...}}
AUTO_RECOMMENDED_MODEL = {}


# ── Manual variogram-fit overrides (optional) ──────────────────────────────────────
# Leave any field as None to keep the auto-derived value (from network_geometry_params
# for lag_dist/n_lag, or the 0.1/0.8-var heuristic for p0). Fill in a value to override
# just that piece for one variable, then re-run this cell and inspect the plot again --
# all before anything gets locked in Section 9b.
MANUAL_VARIOGRAM_OVERRIDE = {
    "tmax": {"lag_dist": None, "n_lag": None, "p0": None},  # p0 = [nugget, psill, range_km]
    "tmin": {"lag_dist": None, "n_lag": None, "p0": None},
}



for v in VARIABLE_CONFIGS:
    station_xy_km = station_utm_lookup[v].copy()
    station_xy_km["x_km"] = station_xy_km["x_m"] / 1000.0
    station_xy_km["y_km"] = station_xy_km["y_m"] / 1000.0

    geom = network_geometry_params(station_xy_km["x_km"].to_numpy(), station_xy_km["y_km"].to_numpy())

    # ── Apply manual override, if set ──────────────────────────────────────────
    override = MANUAL_VARIOGRAM_OVERRIDE.get(v, {})
    override_applied = []
    if override.get("lag_dist") is not None:
        geom["lag_dist"] = override["lag_dist"]
        override_applied.append(f"lag_dist={geom['lag_dist']}")
    if override.get("n_lag") is not None:
        geom["n_lag"] = override["n_lag"]
        override_applied.append(f"n_lag={geom['n_lag']}")

    geom_params[v] = geom
    pair_table = build_station_pair_table(station_xy_km[["x_km", "y_km"]])
    pair_tables[v] = pair_table

    tag = f"  ⚙ MANUAL OVERRIDE: {', '.join(override_applied)}" if override_applied else ""
    logger.info(f"[{v}] pooled-variogram geometry: domain_extent={geom['domain_extent']:.0f} km  "
                f"median_nn={geom['median_nn']:.1f} km  LAG_DIST={geom['lag_dist']:.0f} km  "
                f"N_LAG={geom['n_lag']}{tag}")

    lag, gamma, npairs = compute_pooled_semivariogram(
        anomalies_df[v][["date", "index_no", "anomaly"]], pair_table, geom["lag_dist"], geom["n_lag"]
    )
    valid = (npairs >= MIN_PAIRS_PER_BIN) & np.isfinite(gamma)
    lag_fit, gamma_fit, npairs_fit = lag[valid], gamma[valid], npairs[valid]
    pooled_omni[v] = (lag, gamma, npairs)

    pooled_var = float(anomalies_df[v]["anomaly"].var())
    range_init = (float(lag_fit[np.argmax(gamma_fit >= 0.8 * pooled_var)])
                  if (gamma_fit >= 0.8 * pooled_var).any() else float(lag_fit.mean()))
    p0_auto = [0.1 * pooled_var, 0.8 * pooled_var, range_init]

    # ── Apply p0 override, if set ──────────────────────────────────────────────
    if override.get("p0") is not None:
        p0 = list(override["p0"])
        logger.info(f"[{v}] MANUAL OVERRIDE p0: nugget={p0[0]:.4f} psill={p0[1]:.4f} range={p0[2]:.1f}km")
    else:
        p0 = p0_auto

    fit_results = {}
    for model_name in MODEL_TRIAL_ORDER:
        try:
            r = fit_variogram_lm(model_name, lag_fit, gamma_fit, npairs_fit, p0)
            # SAFETY NET: the architecture doc asks for an UNCONSTRAINED LM fit
            # ("Range is not pre-fixed"), which this is — no bounds are passed to
            # curve_fit (D2). But unconstrained LM can in principle walk into a
            # degenerate solution (e.g. range orders of magnitude beyond the
            # domain, which just fits a near-linear trend through the lag window
            # and can still "win" on raw chi^2). These flags don't change the fit
            # itself, only whether it's eligible for auto-selection below — same
            # spirit as V2's sill/range flags, and exactly the kind of thing the
            # doc's "deliberate, inspected, not an auto-accept" language is
            # warning about. In practice (post binning-bug fix) fitted ranges
            # have landed well inside the domain for both variables.
            r["degenerate"] = (
                r["range_km"] <= 0 or r["psill"] <= 0 or r["nugget"] < 0
                or r["range_km"] > 3 * geom["domain_extent"]
                or r["range_km"] < geom["lag_dist"] / 2
            )
            fit_results[model_name] = r
        except (RuntimeError, ValueError) as e:
            logger.error(f"[{v}] {model_name:<12} FAILED — {e}")
    variogram_fit_results[v] = fit_results

    non_degenerate = {m: r for m, r in fit_results.items() if not r["degenerate"]}
    selection_pool = non_degenerate if non_degenerate else fit_results
    if not non_degenerate:
        logger.warning(f"[{v}] ALL candidate fits look degenerate (range far outside the "
                        f"domain, or non-physical nugget/psill) — falling back to lowest chi^2 "
                        f"among all of them anyway. Treat this variogram as provisional.")
    best_model = min(selection_pool, key=lambda m: selection_pool[m]["chi2"])
    AUTO_RECOMMENDED_MODEL[v] = best_model

    logger.info(f"[{v}] Pooled anomaly variance: {pooled_var:.3f}")
    for m, r in fit_results.items():
        tag = "  <- lowest chi^2 (auto-recommended)" if m == best_model else ""
        tag += "  ⚠ DEGENERATE (excluded from auto-pick unless all are)" if r["degenerate"] else ""
        logger.info(f"  {m:<12} nugget={r['nugget']:.4f}  psill={r['psill']:.4f}  "
                    f"range={r['range_km']:.1f} km  chi2={r['chi2']:.2f}{tag}")

    # [LINUX PORT, L3] The experimental-vs-fitted-model diagnostic plot and the
    # directional/anisotropy-check plot (plus its extra per-azimuth
    # compute_pooled_semivariogram() recomputation) were here and have been
    # removed for headless/unattended execution -- see the V4.1 changelog (L3)
    # at the top of this file. All of the fitted values are still logged above.

logger.info("Auto-recommended model per variable (unattended run — diagnostic plots skipped, "
            "see V4.1 changelog L3; Section 9b's LOCKED_MODEL is the actual source of truth):")
for v in VARIABLE_CONFIGS:
    logger.info(f"  {v}: {AUTO_RECOMMENDED_MODEL[v]}")

# %%
# ── Section 9b: Lock the variogram (manual review checkpoint) ─────────────────────

log_step("SECTION 9b — Lock the Global Variogram (manual review checkpoint)")

# 1. Pick the model (default to auto-recommended, or uncomment to override):
#LOCKED_MODEL = dict(AUTO_RECOMMENDED_MODEL)
LOCKED_MODEL = {'tmax': 'exponential', 'tmin': 'exponential'}


# 2. Extract the exact fitted parameters from Section 9:
LOCKED_PARAMS = {
    v: {
        "nugget": variogram_fit_results[v][LOCKED_MODEL[v]]["nugget"],
        "psill": variogram_fit_results[v][LOCKED_MODEL[v]]["psill"],
        "range_km": variogram_fit_results[v][LOCKED_MODEL[v]]["range_km"],
    }
    for v in VARIABLE_CONFIGS
}

# 3. Anisotropy settings
# Set to True if you want to use the omnidirectional variogram as-is:
ISOTROPIC_OK = {"tmax": True, "tmin": True}
ANISOTROPY_MAJOR_AZIMUTH_DEG = {"tmax": 90, "tmin": 90}

# If you want anisotropy, define scaling manually or fit with safety bounds:
ANISOTROPY_SCALING = {"tmax": 2, "tmin": 4}
ANISOTROPY_ANGLE_PYKRIGE_DEG = {"tmax": 0.0, "tmin": 0.0}

if not all(ISOTROPIC_OK.values()):
    for v in VARIABLE_CONFIGS:
        if not ISOTROPIC_OK[v]:
            major_az = ANISOTROPY_MAJOR_AZIMUTH_DEG[v] % 180.0
            ANISOTROPY_ANGLE_PYKRIGE_DEG[v] = (90.0 - major_az) % 360.0
            # Set a physically realistic scaling ratio (e.g., 1.5 - 2.0) based on visual inspection of Sec 9 plot
            # instead of unconstrained LM on sparse directional bins:
            ANISOTROPY_SCALING[v] = 1.5  

# Save summary
variogram_summary_rows = []
for v in VARIABLE_CONFIGS:
    variogram_summary_rows.append({
        "variable": v, "model": LOCKED_MODEL[v], **LOCKED_PARAMS[v],
        "isotropic_ok": ISOTROPIC_OK[v],
        "anisotropy_scaling": ANISOTROPY_SCALING[v],
        "anisotropy_angle_pykrige_deg": ANISOTROPY_ANGLE_PYKRIGE_DEG[v],
        "anisotropy_major_azimuth_compass_deg": ANISOTROPY_MAJOR_AZIMUTH_DEG.get(v),
    })
pd.DataFrame(variogram_summary_rows).to_csv(VARIOGRAM_SUMMARY_CSV, index=False)
logger.info(f"Locked global variogram saved -> {VARIOGRAM_SUMMARY_CSV}")
for v in VARIABLE_CONFIGS:
    logger.info(f"  {v}: {LOCKED_MODEL[v]}  {LOCKED_PARAMS[v]}  aniso_scale={ANISOTROPY_SCALING[v]:.2f}  "
                f"aniso_angle={ANISOTROPY_ANGLE_PYKRIGE_DEG[v]:.1f}°")

# %%
# ── Section 10: Step 5b helper — LOOCV-KED evaluator (reused by Step 5b below) ───
# No search-radius / local-neighbor restriction (D6) -- both this evaluator and
# the production kriging call in Section 12 run unrestricted UniversalKriging
# over ALL reporting stations for the day, using the SAME code path. This also
# means CV here and production are literally the same kriging call, with the
# same anisotropy locked in (D1) -- no CV/production mismatch is possible.

def krige_one_point_ked(x_others, y_others, v_others, x_pt, y_pt, model_name, params,
                         anisotropy_scaling=1.0, anisotropy_angle=0.0):
    uk = UniversalKriging(
        x_others, y_others, v_others,
        variogram_model=model_name,
        variogram_parameters={"psill": params["psill"], "range": params["range_km"] * 1000.0,
                               "nugget": params["nugget"]},
        drift_terms=["external_Z"],
        external_drift=dem_grid_utm, external_drift_x=grid_x, external_drift_y=grid_y,
        anisotropy_scaling=anisotropy_scaling, anisotropy_angle=anisotropy_angle,
        verbose=False, enable_plotting=False,
    )
    pred, var = uk.execute("points", np.array([x_pt]), np.array([y_pt]))
    return float(pred[0])


def krige_one_point_ok(x_others, y_others, v_others, x_pt, y_pt, model_name, params,
                        anisotropy_scaling=1.0, anisotropy_angle=0.0):
    ok = OrdinaryKriging(
        x_others, y_others, v_others,
        variogram_model=model_name,
        variogram_parameters={"psill": params["psill"], "range": params["range_km"] * 1000.0,
                               "nugget": params["nugget"]},
        anisotropy_scaling=anisotropy_scaling, anisotropy_angle=anisotropy_angle,
        verbose=False, enable_plotting=False,
    )
    pred, var = ok.execute("points", np.array([x_pt]), np.array([y_pt]))
    return float(pred[0])


def evaluate_loo_ked(var_name, sample_dates, model_name, params, use_drift=True,
                      anisotropy_scaling=1.0, anisotropy_angle=0.0):
    """
    Leave-one-station-out CV, unrestricted (D6: no radius_km parameter -- every
    fold uses all OTHER reporting stations for that day, matching production's
    Section 12 kriging call exactly).
    """
    anomalies = anomalies_df[var_name]
    lookup = station_utm_lookup[var_name]
    errors = []
    n_folds = 0

    drift_tag = "KED" if use_drift else "OK(no-drift)"
    # [LINUX PORT, L2] progress bar only -- loop body/logic unchanged.
    for d in tqdm(sample_dates, desc=f"LOOCV {var_name} [{drift_tag}]", leave=False):
        day = anomalies[anomalies["date"] == d]
        if len(day) < MIN_STATIONS_FOR_CV_FOLD:
            continue
        day = day.merge(lookup, left_on="index_no", right_index=True)
        xs, ys, vs = day["x_m"].to_numpy(), day["y_m"].to_numpy(), day["anomaly"].to_numpy()

        for i in range(len(day)):
            others = np.arange(len(day)) != i
            if others.sum() < MIN_STATIONS_FOR_CV_FOLD - 1:
                continue
            try:
                if use_drift:
                    pred = krige_one_point_ked(xs[others], ys[others], vs[others], xs[i], ys[i],
                                                model_name, params,
                                                anisotropy_scaling=anisotropy_scaling,
                                                anisotropy_angle=anisotropy_angle)
                else:
                    pred = krige_one_point_ok(xs[others], ys[others], vs[others], xs[i], ys[i],
                                               model_name, params,
                                               anisotropy_scaling=anisotropy_scaling,
                                               anisotropy_angle=anisotropy_angle)
            except Exception:
                continue
            errors.append(pred - vs[i])
            n_folds += 1

    errors = np.array(errors)
    if len(errors) == 0:
        return {"rmse": np.nan, "mae": np.nan, "n_folds": 0}
    return {"rmse": float(np.sqrt(np.mean(errors ** 2))), "mae": float(np.mean(np.abs(errors))),
            "n_folds": n_folds}

# %%
# ── Section 11: Step 5b — Final Embedded LOOCV (+ drift-term confirmation) ───────
# Two purposes (architecture doc): (1) unbiased RMSE/MAE for the locked method,
# (2) empirical with-vs-without-drift comparison. Per the Decisions Log, KED is
# "Confirmed, unconditional" — this comparison is reported for context only and
# does not gate anything. (Step 5a's search-radius CV has been removed per D6 --
# there is no radius to select.)

log_step("SECTION 11 / STEP 5b — Final Embedded LOOCV")

cv_sample_dates = {}
cv_final_rows = []
for v in VARIABLE_CONFIGS:
    dates_v = anomalies_df[v]["date"]
    sample_dates = sample_stratified_dates(dates_v, frac=None, n_target=CV_N_SAMPLE_DAYS)
    cv_sample_dates[v] = sample_dates

    res_drift = evaluate_loo_ked(v, sample_dates, LOCKED_MODEL[v], LOCKED_PARAMS[v], use_drift=True,
                                  anisotropy_scaling=ANISOTROPY_SCALING[v],
                                  anisotropy_angle=ANISOTROPY_ANGLE_PYKRIGE_DEG[v])
    res_nodrift = evaluate_loo_ked(v, sample_dates, LOCKED_MODEL[v], LOCKED_PARAMS[v], use_drift=False,
                                    anisotropy_scaling=ANISOTROPY_SCALING[v],
                                    anisotropy_angle=ANISOTROPY_ANGLE_PYKRIGE_DEG[v])
    cv_final_rows.append({"variable": v, "with_drift_rmse": res_drift["rmse"],
                           "with_drift_mae": res_drift["mae"], "with_drift_n": res_drift["n_folds"],
                           "no_drift_rmse": res_nodrift["rmse"], "no_drift_mae": res_nodrift["mae"],
                           "no_drift_n": res_nodrift["n_folds"]})
    logger.info(f"[{v}] FINAL LOOCV (unrestricted, n_days={len(sample_dates)}):  "
                f"KED RMSE={res_drift['rmse']:.3f} (n={res_drift['n_folds']})   "
                f"no-drift RMSE={res_nodrift['rmse']:.3f} (n={res_nodrift['n_folds']})")

pd.DataFrame(cv_final_rows).to_csv(CV_FINAL_CSV, index=False)
logger.info(f"Final LOOCV validation saved -> {CV_FINAL_CSV}")
logger.info("(Held-out test-parquet validation is out of scope here per the architecture "
            "doc — handled by the separate validation script.)")

# %%
# ── Section 12: Steps 6-8 — Daily KED Kriging Loop + Reconstruction ──────────────
# D6/D3/D4 all land here: unrestricted global KED (no blockwise approximation),
# masked-array fill fixed, and nodata handled explicitly through reprojection.
# [D9] The KED grid and reconstruction still happen on the buffered grid exactly
# as before; the only new step is the clip_to_output_grid() call right before
# each day's raster is written to disk.

log_step("SECTION 12 / STEPS 6-8 — Daily KED Kriging Loop + Reconstruction")


def run_ked_global(day_df_utm, model_name, params, anisotropy_scaling=1.0, anisotropy_angle=0.0):
    uk = UniversalKriging(
        day_df_utm["x_m"].to_numpy(), day_df_utm["y_m"].to_numpy(), day_df_utm["anomaly"].to_numpy(),
        variogram_model=model_name,
        variogram_parameters={"psill": params["psill"], "range": params["range_km"] * 1000.0,
                               "nugget": params["nugget"]},
        drift_terms=["external_Z"],
        external_drift=dem_grid_utm, external_drift_x=grid_x, external_drift_y=grid_y,
        anisotropy_scaling=anisotropy_scaling, anisotropy_angle=anisotropy_angle,
        verbose=False, enable_plotting=False,
    )
    zgrid, _ = uk.execute("grid", grid_x, grid_y, backend="vectorized")
    # FIX (D3): `.data` exposes the masked array's raw underlying buffer, which for
    # points PyKrige marked invalid/unsolved can be a sentinel fill value rather
    # than NaN. `.filled(np.nan)` respects the mask so failed points come through
    # as real NaN, which downstream reconstruction correctly treats as "no
    # anomaly info here, fall back to climatology-only" via np.nan_to_num.
    return np.flipud(zgrid.filled(np.nan))


def reproject_utm_to_template(utm_array):
    """UTM anomaly grid -> WGS84 (buffered) template grid, bilinear (same
    convention as V2's final reprojection step). Still lands on the buffered
    grid -- the crop to delivery extent happens separately, in Section 12's
    save loop, via clip_to_output_grid()."""
    # [PARALLEL, P2] Used to be one fixed shared filename
    # (os.path.join(temp_dir, "_anomaly_utm_tmp.tif")). That's fine when this
    # runs strictly sequentially in one process, but under Section 12's
    # multiprocessing pool, multiple worker processes could call this at the
    # same time and stomp on each other's temp file mid-write/mid-read,
    # silently corrupting output. Each call now gets its own uniquely-named
    # temp file (pid + a random token) and removes it when done.
    tmp_name = f"_anomaly_utm_tmp_{os.getpid()}_{uuid.uuid4().hex}.tif"
    tmp_path_utm = os.path.join(temp_dir, tmp_name)
    # FIX (D4): explicitly burn nodata_out into invalid pixels before writing, and
    # tell reproject() what the source nodata value is (src_nodata=nodata_out)
    # rather than writing raw NaN and relying on IEEE NaN-arithmetic happening to
    # propagate correctly through GDAL's bilinear resampling across the
    # valid/invalid boundary.
    array_with_nodata = np.where(np.isnan(utm_array), nodata_out, utm_array).astype("float32")
    with rasterio.open(tmp_path_utm, "w", **utm_meta) as dst:
        dst.write(array_with_nodata, 1)

    out = np.full((tmpl_height, tmpl_width), np.nan, dtype="float32")
    try:
        with rasterio.open(tmp_path_utm) as src:
            reproject(
                source=rasterio.band(src, 1), destination=out,
                src_transform=src.transform, src_crs=src.crs,
                dst_transform=tmpl_transform, dst_crs=tmpl_crs,
                src_nodata=nodata_out, dst_nodata=np.nan,
                resampling=Resampling.bilinear,
            )
    finally:
        # [PARALLEL, P2] Clean up so a 25-year x 2-variable parallel run doesn't
        # leave tens of thousands of stray temp GeoTIFFs behind.
        try:
            os.remove(tmp_path_utm)
        except OSError:
            pass
    return out


target_dates = pd.date_range(start=krig_start_day, end=krig_end_day, freq="D")
log_rows = []

# Diagnostic: real per-day station-count distribution, to sanity-check
# MIN_STATIONS_FOR_KED (Section 1's flagged placeholder) before trusting the run.
for v in VARIABLE_CONFIGS:
    counts = anomalies_df[v].groupby("date").size()
    logger.info(f"[{v}] per-day station-count distribution: min={counts.min()}  "
                f"p10={counts.quantile(.1):.0f}  median={counts.median():.0f}  max={counts.max()}  "
                f"days below MIN_STATIONS_FOR_KED({MIN_STATIONS_FOR_KED})={int((counts < MIN_STATIONS_FOR_KED).sum())}"
                f" / {len(counts)}")


def _krige_and_write_one(date, v):
    """
    [PARALLEL, P1/P3] Worker: does exactly what one (date, v) pass of the
    original sequential loop body did -- decide climatology-only vs KED, krige,
    clip, write the GeoTIFF -- and returns a small result dict instead of
    touching `logger` or `log_rows` directly.

    Runs in a forked child process, so it can see every read-only object
    already built at module scope in the parent (anomalies_df,
    station_utm_lookup, climatology_grid_wgs84, dem_grid_utm, grid_x, grid_y,
    LOCKED_MODEL, LOCKED_PARAMS, ANISOTROPY_SCALING, ANISOTROPY_ANGLE_PYKRIGE_DEG,
    tmpl_profile_out, output_dir, MIN_STATIONS_FOR_KED, nodata_out) via
    copy-on-write -- none of that is re-pickled or recomputed per task, only the
    (date, v) arguments and this small return dict cross the process boundary.

    Per-process logging is intentionally NOT done in here (P3): a shared
    logging.FileHandler across forked processes can interleave/garble writes,
    so the parent process (which owns the one real logger) does all logging
    once results come back.
    """
    date_str = date.strftime("%Y%m%d")
    month = date.month
    final_path = os.path.join(output_dir, v, f"{v}_{date_str}.tif")

    if os.path.exists(final_path):
        return {"date": date_str, "variable": v, "status": "skipped_exists"}

    day_df = anomalies_df[v][anomalies_df[v]["date"] == date]
    n_available = len(day_df)
    clim_grid = climatology_grid_wgs84[v][month]

    error_info = None
    if n_available == 0:
        recon = clim_grid
        method = "no_data"
    elif n_available < MIN_STATIONS_FOR_KED:
        recon = clim_grid
        method = "fallback_zero_anomaly"
    else:
        day_utm = day_df.merge(station_utm_lookup[v], left_on="index_no", right_index=True)
        try:
            anomaly_grid_utm = run_ked_global(day_utm, LOCKED_MODEL[v], LOCKED_PARAMS[v],
                                               anisotropy_scaling=ANISOTROPY_SCALING[v],
                                               anisotropy_angle=ANISOTROPY_ANGLE_PYKRIGE_DEG[v])
            anomaly_grid_wgs84 = reproject_utm_to_template(anomaly_grid_utm)
            recon = clim_grid + np.nan_to_num(anomaly_grid_wgs84, nan=0.0)
            method = "KED"
        except Exception as e:
            recon = clim_grid
            method = "KED_failed_fallback_zero_anomaly"
            # [LINUX PORT, L5 / PARALLEL, P3] fallback itself is unchanged --
            # this just carries the actual exception back to the parent so it
            # still ends up in the run log (the parent logs it, not us).
            error_info = f"{type(e).__name__}: {e}"

    # [NEW, D9] crop from the buffered processing grid back down to the
    # original delivery extent right before writing.
    recon_clipped = clip_to_output_grid(recon)

    out_profile = tmpl_profile_out.copy()
    with rasterio.open(final_path, "w", **out_profile) as dst:
        dst.write(np.nan_to_num(recon_clipped, nan=nodata_out).astype(out_profile["dtype"]), 1)

    result = {"date": date_str, "variable": v, "n_stations_used": n_available,
              "method": method, "model": LOCKED_MODEL[v],
              "neighbor_selection": "unrestricted_global", "status": "written"}
    if error_info is not None:
        result["error"] = error_info
    return result


# [PARALLEL, P1] Flat, ordered (date-major, variable-minor -- same order as the
# original nested loop) task list. Because executor.map() below preserves
# input order in its output regardless of actual completion order, the day==1
# progress-ping grouping further down can safely assume that every
# n_vars-sized consecutive run of results belongs to one date.
task_dates, task_vars = [], []
for date in target_dates:
    for v in VARIABLE_CONFIGS:
        task_dates.append(date)
        task_vars.append(v)
n_tasks = len(task_dates)
n_vars = len(VARIABLE_CONFIGS)

logger.info(f"Starting daily kriging loop: {len(target_dates)} days x {n_vars} "
            f"variable(s) = {n_tasks} total day-variable rasters to produce "
            f"(existing outputs are skipped, see above).")

# [PARALLEL, P1] fork-based process pool. See the V4.2 changelog at the top of
# this file for why processes (not threads), why `fork` specifically, and the
# sequential fallback below if `fork` isn't available on this platform.
_fork_available = "fork" in mp.get_all_start_methods()
_use_parallel = N_WORKERS > 1 and _fork_available

if N_WORKERS > 1 and not _fork_available:
    logger.warning("This platform doesn't support the 'fork' multiprocessing start "
                    "method (Linux/macOS only), and this script has no "
                    "`if __name__ == \"__main__\":` guard, so a 'spawn'-based pool "
                    "isn't safe here. Falling back to sequential execution in this "
                    "one process -- run on Linux/macOS for the parallel speedup.")

executor = None
if _use_parallel:
    logger.info(f"Parallelizing across {N_WORKERS} worker process(es) (fork-based "
                f"ProcessPoolExecutor).")
    _mp_ctx = mp.get_context("fork")
    _chunksize = max(1, n_tasks // (N_WORKERS * 4))
    executor = ProcessPoolExecutor(max_workers=N_WORKERS, mp_context=_mp_ctx)
    results_iter = executor.map(_krige_and_write_one, task_dates, task_vars, chunksize=_chunksize)
else:
    logger.info("Running sequentially in this process (N_WORKERS=1 or fork unavailable).")
    results_iter = (_krige_and_write_one(d, v) for d, v in zip(task_dates, task_vars))

day_rows_buffer = []
try:
    for result in tqdm(results_iter, total=n_tasks, desc="Daily KED kriging", unit="raster"):
        if result["status"] == "skipped_exists":
            logger.info(f"{result['variable']} for {result['date']} - skipping (exists)")
        else:
            if "error" in result:
                logger.error(f"[{result['variable']}] {result['date']}: KED failed, falling "
                              f"back to climatology-only ({result['error']})")
            row = {k: val for k, val in result.items() if k not in ("status", "error")}
            log_rows.append(row)

        # Same monthly progress ping as the original sequential version, just
        # re-grouped: every n_vars consecutive results (guaranteed contiguous
        # per date by the ordered task list above) correspond to one date.
        day_rows_buffer.append(result)
        if len(day_rows_buffer) == n_vars:
            if day_rows_buffer[0]["date"][6:8] == "01":
                written_only = [r for r in day_rows_buffer if r["status"] == "written"]
                if written_only:
                    summary = "  ".join(f"{r['variable']}: n={r['n_stations_used']} {r['method']}"
                                         for r in written_only)
                    logger.info(f"{day_rows_buffer[0]['date']}  {summary}")
            day_rows_buffer = []
finally:
    if executor is not None:
        executor.shutdown(wait=True)

log_df = pd.DataFrame(log_rows)
log_df.to_csv(LOG_CSV, index=False)

logger.info(f"Daily KED kriging complete — {len(log_df)} day-variable rows")
for v in VARIABLE_CONFIGS:
    sub = log_df[log_df["variable"] == v]
    logger.info(f"{v} method breakdown:")
    for method, count in sub["method"].value_counts().items():
        logger.info(f"  {method:<28} {count:>6}  ({count/len(sub)*100:.1f}%)")
logger.info(f"Log saved -> {LOG_CSV}")

log_step("PIPELINE COMPLETE")