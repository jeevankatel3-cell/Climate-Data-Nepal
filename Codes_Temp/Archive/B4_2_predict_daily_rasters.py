# %%
# %%
# Use the QRF regressor (fitting script v3) to make two daily prediction
# products per variable, per pixel, over the full grid:
#   1. Deterministic  -- the model's median read, one band per day
#   2. Stochastic     -- 100 realizations per day, Schaake-shuffled
#                        (Clark et al. 2004) against a CHIRTS historical
#                        archive to restore cross-pixel / cross-variable
#                        rank coherence that independent per-pixel QRF
#                        sampling doesn't have on its own
#
# Daily GeoTIFFs only for now (one 1-band file for product 1, one 100-band
# file for product 2, per day) -- yearly NetCDF aggregation is a separate
# follow-up script.

# V2 - 2026-08-12 - reworked for the QRF shared-core architecture (was:
#   single point-estimate RF/XGB prediction); added the Schaake-shuffled
#   stochastic product - jpk

# Workflow
# 1. Load config, QRF predictor bundles, spatial template
# 2. Preload the CHIRTS historical archive ONCE per variable (the Schaake
#    reference pool) -- NOT per target day, see note below
# 3. Iterate over days in the period range
# 4. Per day:
#    a. Read that day's predictor rows from parquet
#    b. Deterministic: bundle.predict_median() -> 1-band GeoTIFF
#    c. Stochastic: bundle.sample(n=100) -> pick this day's 100 reference
#       dates from the historical archive -> Schaake-shuffle -> 100-band
#       GeoTIFF
# 5. Align every output to the template extent via reproject_match, save
#    as float32 GeoTIFF

# %%
# libraries
import os
import gc
import glob
import joblib
import json
import numpy as np
import pandas as pd
import xarray as xr
import rioxarray as rxr
from rasterio.enums import Resampling
from scipy.stats import rankdata
from tqdm import tqdm

# QRFPredictor lives in its own module so this script and the fitting
# script both import the SAME class definition -- required for
# joblib/pickle to unpickle *_qrf_predictor.pkl bundles correctly. Needs
# qrf_predictor.py to sit alongside this script (same folder / importable
# path).
from qrf_predictor import QRFPredictor

start_year, start_month, start_day = 2001, 1, 1
end_year,   end_month, end_day   = 2025, 12, 31     # trial run; final run: end_year = 2025

N_REALIZATIONS = 100
SHUFFLE_WINDOW_DAYS = 7          # Clark et al. (2004) -- +/- 7 days
random_state_num = 64            # matches the fitting script's convention

# paths
out_base = fr"..\Data_Temp\Processed\Daily_bias_correction"
model_path = os.path.join(out_base, r"Saved_regressors")
template = rxr.open_rasterio(r"..\Data_Temp\Datasets\CHIRTS\CHIRTS_tmax_20010101.tif")

tmax_temp_dir = os.path.join(out_base, 'Temp_tmax')
tmax_yearly_parquet_dir = os.path.join(tmax_temp_dir, 'Tmax_yearly_parquet')
tmax_final_pred_dir = os.path.join(out_base, "Final_products", "Tmax")
tmax_deterministic_dir = os.path.join(tmax_final_pred_dir, "Deterministic")
tmax_stochastic_dir = os.path.join(tmax_final_pred_dir, "Stochastic")

tmin_temp_dir = os.path.join(out_base, 'Temp_tmin')
tmin_yearly_parquet_dir = os.path.join(tmin_temp_dir, 'Tmin_yearly_parquet')
tmin_final_pred_dir = os.path.join(out_base, "Final_products", "Tmin")
tmin_deterministic_dir = os.path.join(tmin_final_pred_dir, "Deterministic")
tmin_stochastic_dir = os.path.join(tmin_final_pred_dir, "Stochastic")

for d in [tmax_temp_dir, tmax_yearly_parquet_dir, tmax_deterministic_dir, tmax_stochastic_dir,
          tmin_temp_dir, tmin_yearly_parquet_dir, tmin_deterministic_dir, tmin_stochastic_dir]:
    os.makedirs(d, exist_ok=True)


def get_model_features(t_var: str) -> list:
    """Predictor feature list straight from the fitting script's metadata
    JSON -- only one model family (qrf) now, so no glob/model_type lookup
    needed like the old RF/XGB version."""
    metadata_file = os.path.join(model_path, f"{t_var}_qrf_feature_metadata.json")
    with open(metadata_file, 'r') as f:
        metadata = json.load(f)
    return metadata['input_features']

predictor_features_tmax = get_model_features("tmax")
predictor_features_tmin = get_model_features("tmin")

# %%
# Helpers -- Schaake shuffle (Clark et al. 2004)
#
# Verified against the paper's own worked numerical example before use here
# (10-member ensemble, exact match) and against a vectorized multi-pixel
# version of the same example -- see conversation history for the checks.

def get_window_dates(archive_dates: pd.DatetimeIndex, target_date: pd.Timestamp,
                      window_days: int = SHUFFLE_WINDOW_DAYS) -> pd.DatetimeIndex:
    """
    All archive dates within +/- window_days of target_date's day-of-year,
    circularly (so a window around Jan 2 correctly reaches back into late
    December), excluding the target date's own YEAR entirely -- not just
    the date itself. Full-year exclusion avoids a realization ever drawing
    from t+/-1 within the same year, which would be near-identical weather
    from short-range serial correlation, not an independent historical
    analog.

    Circular distance uses a fixed 365-day year, so there's an accepted
    ~1-day fuzz around Feb 29 in leap years -- consistent with the window's
    own +/-7-day coarseness, not worth special-casing.
    """
    target_doy = target_date.dayofyear
    doy = archive_dates.dayofyear
    diff = np.abs(doy - target_doy)
    circ_diff = np.minimum(diff, 365 - diff)
    mask = (circ_diff <= window_days) & (archive_dates.year != target_date.year)
    return archive_dates[mask]


def select_reference_dates(candidate_dates: pd.DatetimeIndex, n_realizations: int,
                            rng: np.random.Generator) -> np.ndarray:
    """
    Pick n_realizations historical dates -- ONE shared set, used identically
    across every pixel and both variables for this target day. That sharing
    is what creates the cross-pixel / cross-variable coherence (Clark et al.
    2004) -- it is NOT an independent draw per pixel.

    Falls back to sampling WITH replacement if the candidate pool is
    smaller than n_realizations. Expected in the 2001-2005 trial window:
    ~4 other years x 15 candidate days =~ 60 candidates < 100, so
    replace=True will engage most/every day in the trial run, and
    replace=False will engage for the full 2001-2025 run (~24 other years
    x 15 =~ 360 candidates).
    """
    replace = len(candidate_dates) < n_realizations
    return rng.choice(candidate_dates, size=n_realizations, replace=replace)


def schaake_shuffle_grid(raw_ensemble: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """
    Vectorized Clark et al. (2004) rank-reordering across every pixel row
    at once (no Python loop over pixels/stations).

    raw_ensemble, Y : shape (n_pixels, n_realizations), same pixel row
    order. For each pixel: sort the raw ensemble ascending, rank the
    historical Y-vector, and reassign so the ensemble slot whose historical
    analog was the day's r-th smallest value receives the r-th smallest raw
    ensemble value.

    Pixels where either the raw ensemble or Y contains a NaN (a CHIRTS gap
    on one or more of the day's selected historical dates) are passed
    through UNSHUFFLED rather than shuffled or dropped -- can't rank-match
    against missing history, and a gap at one pixel shouldn't corrupt
    neighboring pixels or force NaN into an otherwise-valid ensemble.
    """
    out = raw_ensemble.copy()
    valid = ~np.isnan(Y).any(axis=1) & ~np.isnan(raw_ensemble).any(axis=1)
    if valid.any():
        x_sorted = np.sort(raw_ensemble[valid], axis=1)
        y_rank = rankdata(Y[valid], method="ordinal", axis=1).astype(int) - 1
        out[valid] = np.take_along_axis(x_sorted, y_rank, axis=1)
    return out


def load_chirts_historical_archive(t_variable: str, start_year: int, end_year: int,
                                    template) -> xr.DataArray:
    """
    Preload every available daily CHIRTS_{t_variable} raster in
    [start_year, end_year], aligned to `template`, into ONE in-memory array
    (dims: date, y, x). CHIRTS only -- see the shuffle-source discussion:
    it's the one product every other raster in this pipeline is already
    reprojected onto, so this needs no extra resampling step of its own.

    This preload is the single biggest new I/O cost the shuffle
    introduces, and it's loaded ONCE up front rather than per target day on
    purpose: a naive per-day reload (up to 100 distinct historical rasters
    x every target day across a multi-year run) would mean hundreds of
    thousands of raster reads. Missing days are skipped entirely (not
    NaN-filled), so they simply never become shuffle candidates rather than
    being candidates with empty data.

    Memory: at CHIRTS' native ~5km resolution over a Nepal-sized domain
    this should be small (low hundreds of MB for a 25-year run) -- the
    printed footprint below is the actual number for your domain, worth a
    glance the first time this runs on the real archive rather than my
    estimate.
    """
    dates = pd.date_range(f"{start_year}-01-01", f"{end_year}-12-31", freq="D")
    arrays, valid_dates = [], []
    for d in tqdm(dates, desc=f"Preloading CHIRTS {t_variable} historical archive"):
        path = fr"..\Data_Temp\Datasets\CHIRTS\CHIRTS_{t_variable}_{d.year}{d.month:02d}{d.day:02d}.tif"
        if not os.path.exists(path):
            continue
        da = rxr.open_rasterio(path, masked=True).rio.write_crs("EPSG:4326")
#        da = da.rio.reproject_match(template, resampling=Resampling.nearest).squeeze("band", drop=True)
        da = da.squeeze("band", drop=True)
        arrays.append(da.values.astype("float32"))
        valid_dates.append(d)
        da.close()

    archive = xr.DataArray(
        np.stack(arrays, axis=0),
        dims=("date", "y", "x"),
        coords={"date": pd.DatetimeIndex(valid_dates), "y": template.y.values, "x": template.x.values},
    )
    print(f"  [{t_variable}] historical archive: {archive.shape} "
          f"({len(valid_dates)}/{len(dates)} days found), "
          f"{archive.nbytes / 1e9:.3f} GB in memory")
    return archive


def build_reference_Y(archive: xr.DataArray, selected_dates: np.ndarray,
                       pixel_y: np.ndarray, pixel_x: np.ndarray) -> np.ndarray:
    """Pull the historical Y-matrix for one target day's selected dates, at
    exactly the pixel set present in that day's dataframe (vectorized
    pointwise lookup, not a per-pixel loop) -- shape (n_pixels, n_realizations)."""
    Y = archive.sel(date=selected_dates).sel(
        y=xr.DataArray(pixel_y, dims="pixel"),
        x=xr.DataArray(pixel_x, dims="pixel"),
        method="nearest",
    )
    return Y.transpose("pixel", "date").values


def build_multiband_geotiff(df: pd.DataFrame, value_cols: list, template, outpath: str,
                             nodata: float = -9999) -> None:
    """
    Pivot df[value_cols] (indexed by df['y'], df['x']) into a (band, y, x)
    DataArray -- one band per column, in the given column order -- aligned
    and gap-filled to `template` via reproject_match, and written as a
    float32 GeoTIFF. Used for both products: pass a 1-item list for the
    deterministic file, a 100-item list for the stochastic file. Same
    tested pivot/align/write path for both, so there's no risk of the two
    output types silently diverging in behavior.
    """
    ds = df.set_index(["y", "x"])[value_cols].to_xarray()
    da = ds.to_array(dim="band")
    da = (
        da.rio.write_crs(template.rio.crs)
          .rio.write_nodata(nodata)
          .rio.write_transform(template.rio.transform())
    )
    da = da.rio.reproject_match(template)
    da.rio.to_raster(outpath, dtype="float32")

# %%
#  Main prediction loop

day_range = pd.period_range(
    start=f"{start_year}-{start_month:02d}-{start_day:02d}",
    end=f"{end_year}-{end_month:02d}-{end_day:02d}",
    freq="D",
)

for t_variable in ['tmax']:
    print(f"\n Predicting daily rasters for {t_variable} ==========================================")

    predictor_features = predictor_features_tmax if t_variable == 'tmax' else predictor_features_tmin
    deterministic_dir = tmax_deterministic_dir if t_variable == 'tmax' else tmin_deterministic_dir
    stochastic_dir = tmax_stochastic_dir if t_variable == 'tmax' else tmin_stochastic_dir
    yearly_parquet_dir = tmax_yearly_parquet_dir if t_variable == 'tmax' else tmin_yearly_parquet_dir

    bundle_file = os.path.join(model_path, f"{t_variable}_qrf_predictor.pkl")
    bundle: QRFPredictor = joblib.load(bundle_file)

    # One preload per variable, reused across every day in the loop below --
    # NOT reloaded per day. See load_chirts_historical_archive's docstring.
    chirts_archive = load_chirts_historical_archive(t_variable, start_year, end_year, template)
    archive_dates = chirts_archive.date.to_index()

    for p in tqdm(day_range, desc="Predict rasters"):
        year, month, day = p.year, p.month, p.day
        target_date = pd.Timestamp(year, month, day)

        det_outpath = os.path.join(deterministic_dir, f"{t_variable}_deterministic_{year}{month:02d}{day:02d}.tif")
        sto_outpath = os.path.join(stochastic_dir, f"{t_variable}_stochastic_{year}{month:02d}{day:02d}.tif")

        # Skip only the products that already exist -- each is checked
        # independently, unlike the old single-output version.
        if os.path.exists(det_outpath) and os.path.exists(sto_outpath):
            continue

        df_year_path = os.path.join(yearly_parquet_dir, f"full_{year}.parquet")
        df_day = pd.read_parquet(
            df_year_path,
            filters=[("year", "==", year), ("month", "==", month), ("day", "==", day)],
        )

        if df_day.empty:
            tqdm.write(f"  [{year}-{month:02d}{day:02d}] no rows found, skipping.")
            continue

        # ---- Product 1: deterministic ----
        if not os.path.exists(det_outpath):
            df_day[f"{t_variable}_deterministic"] = bundle.predict_median(df_day[predictor_features])
            build_multiband_geotiff(df_day, [f"{t_variable}_deterministic"], template, det_outpath)

        # ---- Product 2: stochastic, 100 realizations, Schaake-shuffled ----
        if not os.path.exists(sto_outpath):
            # Two independent-but-reproducible RNG streams derived from one
            # per-day seed (SeedSequence.spawn), rather than reusing a single
            # seed for both the QRF draw and the date-selection draw.
            seed_seq = np.random.SeedSequence([random_state_num, p.ordinal])
            qrf_seed, shuffle_seed = seed_seq.spawn(2)

            # bundle.sample() now draws n_realizations random quantile levels
            # and evaluates predict() at all of them in one batched call --
            # same predict() path predict_median() uses (see qrf_predictor.py).
            raw_samples = bundle.sample(
                df_day[predictor_features],
                n_realizations=N_REALIZATIONS,
                random_state=qrf_seed,
            )

            candidates = get_window_dates(archive_dates, target_date)
            if len(candidates) == 0:
                tqdm.write(f"  [{year}-{month:02d}{day:02d}] no historical candidate dates "
                           f"in the CHIRTS archive -- writing raw (unshuffled) ensemble.")
                shuffled = raw_samples
            else:
                rng = np.random.default_rng(shuffle_seed)
                selected_dates = select_reference_dates(candidates, N_REALIZATIONS, rng)
                Y = build_reference_Y(chirts_archive, selected_dates, df_day['y'].values, df_day['x'].values)
                shuffled = schaake_shuffle_grid(raw_samples, Y)

            realization_cols = [f"r{i:03d}" for i in range(N_REALIZATIONS)]
            # Single concat rather than 100 sequential column assignments --
            # the latter measurably fragments the DataFrame at this width.
            df_day = pd.concat(
                [df_day, pd.DataFrame(shuffled, columns=realization_cols, index=df_day.index)],
                axis=1,
            )
            
            build_multiband_geotiff(df_day, realization_cols, template, sto_outpath)

    del chirts_archive
    gc.collect()
    print(f"\nDone. All rasters saved for {t_variable}")


