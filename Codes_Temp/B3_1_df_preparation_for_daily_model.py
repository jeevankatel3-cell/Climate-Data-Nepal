# %%
# Prepare dataframe for bias correcting tmax and tmin on a daily scale
# V1 - 2026-07-27 - create code - jpk
# 2026-08-25 - Adapted for linux


# %%

# libraries
import os
import gc
import sys
import logging
import numpy as np
import pandas as pd
import xarray as xr
import rioxarray as rxr
import pyarrow as pa
import pyarrow.parquet as pq
from scipy.spatial import cKDTree
from tqdm import tqdm
from rasterio.enums import Resampling
import shutil

# --- Configure Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("data_processing.log", mode="w")
    ]
)

logging.info("Starting Data Preparation Pipeline...")

# configurations
start_year, start_month, start_day = 2001, 1, 1
end_year,   end_month, end_day   = 2025, 12, 31

# paths
static_paths = {
    "cluster":         r"../Data/Processed/Spatial_clustering/Temperature_clusters.tif",
    "elev_mean":       r"../Data/Datasets/DEM_1km/DEM_1km_elevation_mean.tif",
    "elev_std":        r"../Data/Datasets/DEM_1km/DEM_1km_elevation_std.tif",
    "elev_max":        r"../Data/Datasets/DEM_1km/DEM_1km_elevation_max.tif",
    "elev_min":        r"../Data/Datasets/DEM_1km/DEM_1km_elevation_min.tif",
    "slope_mean":      r"../Data/Datasets/DEM_1km/DEM_1km_slope_mean.tif",
    "aspect_mean":     r"../Data/Datasets/DEM_1km/DEM_1km_aspect_mean.tif",
    "elev_smoothed":   r"../Data/Processed/Fetch/Preprocessing/dem_smoothed.tif",
    "fetch_1000m_dist":r"../Data/Processed/Fetch/Preprocessing/fetch_dist_1000m.tif",
    "fetch_1000m_dir": r"../Data/Processed/Fetch/Preprocessing/fetch_dir_1000m.tif",
    "fetch_500m_dist": r"../Data/Processed/Fetch/Preprocessing/fetch_dist_500m.tif",
    "fetch_500m_dir":  r"../Data/Processed/Fetch/Preprocessing/fetch_dir_500m.tif",
    "local_relief":    r"../Data/Processed/Fetch/Preprocessing/local_relief.tif",
}

# In-situ
insitu_tmax_raw = pd.read_parquet("../Data/Insitu/Cleaned/tmax_cleaned.parquet")
insitu_tmin_raw = pd.read_parquet("../Data/Insitu/Cleaned/tmin_cleaned.parquet")

tmax_train_df = r"../Data/Processed/Split_stations/tmax_train_stations.csv"
tmax_test_df = r"../Data/Processed/Split_stations/tmax_test_stations.csv"

tmin_train_df = r"../Data/Processed/Split_stations/tmin_train_stations.csv"
tmin_test_df = r"../Data/Processed/Split_stations/tmin_test_stations.csv"

# Output
out_base = r"../Data/Processed/Daily_bias_correction"

tmax_daily_train_dir  = os.path.join(out_base, "tmax_daily_dynamic_train_df")
tmin_daily_train_dir  = os.path.join(out_base, "tmin_daily_dynamic_train_df")

tmax_daily_test_dir  = os.path.join(out_base, "tmax_daily_dynamic_test_df")
tmin_daily_test_dir  = os.path.join(out_base, "tmin_daily_dynamic_test_df")

tmax_parquet_train_dir = os.path.join(out_base, "tmax_yearly_parquet_train")
tmin_parquet_train_dir = os.path.join(out_base, "tmin_yearly_parquet_train")

tmax_parquet_test_dir = os.path.join(out_base, "tmax_yearly_parquet_test")
tmin_parquet_test_dir = os.path.join(out_base, "tmin_yearly_parquet_test")

static_dir = os.path.join(out_base, "static_df")

logging.info("Creating output directories if they don't exist...")
for d in [tmax_daily_train_dir, tmin_daily_train_dir, tmax_daily_test_dir,
    tmin_daily_test_dir, tmax_parquet_train_dir, tmin_parquet_train_dir,
    tmax_parquet_test_dir, tmin_parquet_test_dir, static_dir]:
    os.makedirs(d, exist_ok=True)

logging.info("Loading template raster for alignment...")
template = rxr.open_rasterio(r"../Data/Datasets/CHIRTS/CHIRTS_tmax_20010101.tif", masked = True)
template.rio.write_crs("EPSG:4326", inplace=True)

# %%
# Helpers
def doy_cyclical(year: int, doy: int):
    """Return (sin, cos) encoding for the day of the year."""
    days_in_year = 366 if pd.Timestamp(year, 1, 1).is_leap_year else 365
    angle = 2 * np.pi * doy / days_in_year
    return np.sin(angle), np.cos(angle)

def day_label(year: int, month: int, day: int) -> str:
    return f"{year}{month:02d}{day:02d}"

def build_daily_paths(year: int, month: int, day: int, t_variable: str) -> dict:
    """
    Return dict of {feature_name: filepath} for all daily rasters
    for the given year-month-day.
    """
    return {
        #Gridded daily temperature products
        f"chirts_{t_variable}": fr"../Data/Datasets/CHIRTS/CHIRTS_{t_variable}_{year}{month:02d}{day:02d}.tif",
        f"har_{t_variable}": fr"../Data/Datasets/HAR/HAR_{t_variable}_{year}{month:02d}{day:02d}.tif",
        f"era5l_{t_variable}": fr"../Data/Datasets/ERA5L/ERA5L_{t_variable}_{year}{month:02d}{day:02d}.tif", 

        #GLDAS weather variables
        "gldas_air_temp": fr"../Data/Datasets/GLDAS/air_temp/GLDAS_air_temp_{year}{month:02d}{day:02d}.tif",
        "gldas_lwdown": fr"../Data/Datasets/GLDAS/lwdown/GLDAS_lw_down_{year}{month:02d}{day:02d}.tif",
        "gldas_specific_humidity": fr"../Data/Datasets/GLDAS/specific_humidity/GLDAS_specific_humidity_{year}{month:02d}{day:02d}.tif",
        "gldas_surface_pressure": fr"../Data/Datasets/GLDAS/surface_pressure/GLDAS_surface_pressure_{year}{month:02d}{day:02d}.tif",
        "gldas_swdown": fr"../Data/Datasets/GLDAS/swdown/GLDAS_sw_down_{year}{month:02d}{day:02d}.tif",
        "gldas_wind_speed": fr"../Data/Datasets/GLDAS/wind_speed/GLDAS_wind_speed_{year}{month:02d}{day:02d}.tif"
    }    

# %%
# 1 Build a static csv
static_csv = os.path.join(static_dir, "static_df.csv")
 
if os.path.exists(static_csv):
    logging.info("Loading cached static df...")
    static_df = pd.read_csv(static_csv)
else:
    logging.info("Building static df from rasters...")
    aligned = []
    for name, path in static_paths.items():
        da = rxr.open_rasterio(path, masked=True).rio.write_crs("EPSG:4326")
        da = da.rio.reproject_match(template, resampling=Resampling.nearest).squeeze("band", drop=True)
        da.name = name
        aligned.append(da)
    static_xr = xr.merge(aligned, compat="override")
    static_df  = static_xr.to_dataframe().reset_index()
    static_df.to_csv(static_csv, index=False)
    del aligned, static_xr
    gc.collect()
 
static_df[["x", "y"]] = static_df[["x", "y"]].round(5)
logging.info(f"Static df prepared: {len(static_df):,} pixels")

# %%
# 2 Read insitu and separate train-test
logging.info("Preparing in-situ data (Train/Test splits)...")

# process tmax ==
insitu_tmax_raw["date"]  = pd.to_datetime(insitu_tmax_raw[["year", "month", "day"]])
insitu_tmax_raw  = insitu_tmax_raw.rename(columns={"lat": "y", "long": "x", "max_temp": "tmax"})

insitu_tmax_daily = insitu_tmax_raw[
    (insitu_tmax_raw["year"]  >= start_year) &
    (insitu_tmax_raw["year"]  <= end_year)
].copy()
insitu_tmax_daily[["x", "y"]] = insitu_tmax_daily[["x", "y"]].round(5)

train_ids_tmax = set(pd.read_csv(tmax_train_df)["index_no"])
test_ids_tmax = set(pd.read_csv(tmax_test_df)["index_no"])

insitu_tmax_daily_train = insitu_tmax_daily[insitu_tmax_daily["index_no"].isin(train_ids_tmax)].copy()
insitu_tmax_daily_test = insitu_tmax_daily[insitu_tmax_daily["index_no"].isin(test_ids_tmax)].copy()

# process tmin ==
insitu_tmin_raw["date"]  = pd.to_datetime(insitu_tmin_raw[["year", "month", "day"]])
insitu_tmin_raw  = insitu_tmin_raw.rename(columns={"lat": "y", "long": "x", "min_temp": "tmin"})

insitu_tmin_daily = insitu_tmin_raw[
    (insitu_tmin_raw["year"]  >= start_year) &
    (insitu_tmin_raw["year"]  <= end_year)
].copy()
insitu_tmin_daily[["x", "y"]] = insitu_tmin_daily[["x", "y"]].round(5)

train_ids_tmin = set(pd.read_csv(tmin_train_df)["index_no"])
test_ids_tmin = set(pd.read_csv(tmin_test_df)["index_no"])

insitu_tmin_daily_train = insitu_tmin_daily[insitu_tmin_daily["index_no"].isin(train_ids_tmin)].copy()
insitu_tmin_daily_test = insitu_tmin_daily[insitu_tmin_daily["index_no"].isin(test_ids_tmin)].copy()

# %%
# 3 Daily loop to aggregate daily dataframes
daily_range = pd.period_range(
    start = f"{start_year}-{start_month:02d}-{start_day:02d}",
    end = f"{end_year}-{end_month:02d}-{end_day:02d}",
    freq= "D"
)

logging.info(f"============ Processing {len(daily_range)} days for Tmax ============")

for period in tqdm(daily_range, desc = "Tmax Daily Extraction"):
    year, month, day = period.year, period.month, period.day
    label = day_label(year, month, day)

    tmax_train_out = os.path.join(tmax_daily_train_dir, f"tmax_daily_{label}.csv")
    tmax_test_out = os.path.join(tmax_daily_test_dir, f"tmax_daily_{label}.csv")

    if os.path.exists(tmax_train_out) and os.path.exists(tmax_test_out):
        continue

    paths = build_daily_paths(year, month, day, "tmax")
    aligned = []
    missing_vars = []

    for name, path in paths.items():
        if not os.path.exists(path):
            missing_vars.append(name)
            continue
        da = rxr.open_rasterio(path, masked = True).rio.write_crs("EPSG:4326")
        da = da.rio.reproject_match(template, resampling = Resampling.nearest).squeeze("band", drop = True)
        da.name =  name
        aligned.append(da.load())
        da.close()
    
    if missing_vars:
        logging.warning(f"[{label}] Missing rasters: {missing_vars}")

    if not aligned:
        logging.warning(f"[{label}] No rasters found - skipping day.")
        continue
    
    dynamic_xr = xr.merge(aligned, compat = 'override')
    dynamic_df = dynamic_xr.to_dataframe().reset_index()
    dynamic_df[['x', 'y']] = dynamic_df[['x', 'y']].round(5)

    for name in missing_vars:
        dynamic_df[name] = np.nan
    
    del aligned, dynamic_xr
    gc.collect()

    daily_grid = dynamic_df.merge(static_df, on = ['x', 'y'], how = 'inner')
    doy = pd.Timestamp(year, month, day).dayofyear
    sin_doy, cos_doy = doy_cyclical(year, doy)

    daily_grid['doy_sin'] = sin_doy
    daily_grid['doy_cos'] = cos_doy
    daily_grid['year'] = year
    daily_grid['month'] = month
    daily_grid['day'] = day

    del dynamic_df
    gc.collect()

    insitu_train_day = insitu_tmax_daily_train[
        (insitu_tmax_daily_train['year'] == year) &
        (insitu_tmax_daily_train['month'] == month) &
        (insitu_tmax_daily_train['day'] == day)
    ].copy()

    insitu_test_day = insitu_tmax_daily_test[
        (insitu_tmax_daily_test['year'] == year) &
        (insitu_tmax_daily_test['month'] == month) &
        (insitu_tmax_daily_test['day'] == day) 
    ].copy()

    if insitu_train_day.empty and insitu_test_day.empty:
        del daily_grid
        gc.collect()
        continue
    
    tree = cKDTree(daily_grid[['x', 'y']].values)

    if not insitu_train_day.empty:
        dist_tr, idx_tr = tree.query(insitu_train_day[['x', 'y']].values, k=1)
        mask_tr = dist_tr <= 0.035
        if mask_tr.any():
            station_rows = insitu_train_day.iloc[mask_tr].reset_index(drop=True)
            grid_rows = daily_grid.iloc[idx_tr[mask_tr]].reset_index(drop=True)
            final_train = pd.concat([
                station_rows[["index_no", "y", "x", "year", "month", "day", "tmax"]],
                grid_rows.drop(columns= ['x', 'y', 'year', 'month', 'day'])
            ], axis=1)
            final_train.to_csv(tmax_train_out, index = False)
    
    if not insitu_test_day.empty:
        dist_te, idx_te = tree.query(insitu_test_day[['x', 'y']].values, k=1)
        mask_te = dist_te <= 0.035
        if mask_te.any():
            station_rows = insitu_test_day.iloc[mask_te].reset_index(drop=True)
            grid_rows = daily_grid.iloc[idx_te[mask_te]].reset_index(drop=True)
            final_test = pd.concat([
                station_rows[["index_no", "y", "x", "year", "month", "day", "tmax"]],
                grid_rows.drop(columns=  ['x', 'y', 'year', 'month', 'day'])
            ], axis = 1)
            final_test.to_csv(tmax_test_out, index = False)
    
    del daily_grid, insitu_train_day, insitu_test_day
    gc.collect()

# %%
logging.info(f"============ Processing {len(daily_range)} days for Tmin ============")

for period in tqdm(daily_range, desc = "Tmin Daily Extraction"):
    year, month, day = period.year, period.month, period.day
    label = day_label(year, month, day)

    tmin_train_out = os.path.join(tmin_daily_train_dir, f"tmin_daily_{label}.csv")
    tmin_test_out = os.path.join(tmin_daily_test_dir, f"tmin_daily_{label}.csv")

    if os.path.exists(tmin_train_out) and os.path.exists(tmin_test_out):
        continue

    paths = build_daily_paths(year, month, day, "tmin")
    aligned = []
    missing_vars = []

    for name, path in paths.items():
        if not os.path.exists(path):
            missing_vars.append(name)
            continue
        da = rxr.open_rasterio(path, masked = True).rio.write_crs("EPSG:4326")
        da = da.rio.reproject_match(template, resampling = Resampling.nearest).squeeze("band", drop = True)
        da.name =  name
        aligned.append(da.load())
        da.close()
    
    if missing_vars:
        logging.warning(f"[{label}] Missing rasters: {missing_vars}")

    if not aligned:
        logging.warning(f"[{label}] No rasters found - skipping day.")
        continue
    
    dynamic_xr = xr.merge(aligned, compat = 'override')
    dynamic_df = dynamic_xr.to_dataframe().reset_index()
    dynamic_df[['x', 'y']] = dynamic_df[['x', 'y']].round(5)

    for name in missing_vars:
        dynamic_df[name] = np.nan
    
    del aligned, dynamic_xr
    gc.collect()

    daily_grid = dynamic_df.merge(static_df, on = ['x', 'y'], how = 'inner')
    doy = pd.Timestamp(year, month, day).dayofyear
    sin_doy, cos_doy = doy_cyclical(year, doy)

    daily_grid['doy_sin'] = sin_doy
    daily_grid['doy_cos'] = cos_doy
    daily_grid['year'] = year
    daily_grid['month'] = month
    daily_grid['day'] = day

    del dynamic_df
    gc.collect()

    insitu_train_day = insitu_tmin_daily_train[
        (insitu_tmin_daily_train['year'] == year) &
        (insitu_tmin_daily_train['month'] == month) &
        (insitu_tmin_daily_train['day'] == day) 
    ].copy()

    insitu_test_day = insitu_tmin_daily_test[
        (insitu_tmin_daily_test['year'] == year) &
        (insitu_tmin_daily_test['month'] == month) &
        (insitu_tmin_daily_test['day'] == day)
    ].copy()

    if insitu_train_day.empty and insitu_test_day.empty:
        del daily_grid
        gc.collect()
        continue
    
    tree = cKDTree(daily_grid[['x', 'y']].values)

    if not insitu_train_day.empty:
        dist_tr, idx_tr = tree.query(insitu_train_day[['x', 'y']].values, k=1)
        mask_tr = dist_tr <= 0.035
        if mask_tr.any():
            station_rows = insitu_train_day.iloc[mask_tr].reset_index(drop=True)
            grid_rows = daily_grid.iloc[idx_tr[mask_tr]].reset_index(drop=True)
            final_train = pd.concat([
                station_rows[["index_no", "y", "x", "year", "month", "day", "tmin"]],
                grid_rows.drop(columns= ['x', 'y', 'year', 'month', 'day'])
            ], axis=1)
            final_train.to_csv(tmin_train_out, index = False)
    
    if not insitu_test_day.empty:
        dist_te, idx_te = tree.query(insitu_test_day[['x', 'y']].values, k=1)
        mask_te = dist_te <= 0.035
        if mask_te.any():
            station_rows = insitu_test_day.iloc[mask_te].reset_index(drop=True)
            grid_rows = daily_grid.iloc[idx_te[mask_te]].reset_index(drop=True)
            final_test = pd.concat([
                station_rows[["index_no", "y", "x", "year", "month", "day", "tmin"]],
                grid_rows.drop(columns=  ['x', 'y', 'year', 'month', 'day'])
            ], axis = 1)
            final_test.to_csv(tmin_test_out, index = False)
    
    del daily_grid, insitu_train_day, insitu_test_day
    gc.collect()

# %%
# 5 Aggregate monthly CSVs to yearly parquet --> final combined df
def aggregate_to_parquet(daily_dir: str, parquet_dir: str, t_variable: str, split_label: str) -> str:
    logging.info(f"Aggregating {t_variable} ({split_label}) to Parquet...")
    parquet_paths = []

    for year in range(start_year, end_year + 1):
        yearly_path = os.path.join(parquet_dir, f"full_{year}.parquet")
        daily_periods = pd.period_range(
            start=f"{year}-01-01", end=f"{year}-12-31", freq="D"
        )
        expected = [
            os.path.join(daily_dir, f"{t_variable}_daily_{day_label(p.year, p.month, p.day)}.csv")
            for p in daily_periods
        ]
        found = [f for f in expected if os.path.exists(f)]

        if not found:
            logging.warning(f"{t_variable} [{split_label}] {year} - No daily files found, skipping.")
            continue

        yearly_df = pd.concat([pd.read_csv(fpath) for fpath in found], ignore_index=True)
        yearly_df.to_parquet(yearly_path, index=False)
        
        parquet_paths.append(yearly_path)
        logging.info(f"{t_variable} [{split_label}] {year} - {len(found)} days written to parquet.")
    
    if not parquet_paths:
        logging.warning(f"{t_variable} [{split_label}] No data - Final dataframe not written.")
        return None
    
    final_df = pd.concat([pd.read_parquet(p) for p in parquet_paths], ignore_index=True)
    final_path = os.path.join(out_base, f"final_{t_variable}_{split_label}_df.parquet")
    final_df.to_parquet(final_path, index=False)

    logging.info(f"{t_variable} [{split_label}] Final dataframe compiled: {len(final_df):,} rows × {len(final_df.columns)} cols")
    logging.info(f"Saved to: {final_path}")
    return final_path

tmax_train_final_path = aggregate_to_parquet(tmax_daily_train_dir, tmax_parquet_train_dir, "tmax", "train")
tmax_test_final_path = aggregate_to_parquet(tmax_daily_test_dir, tmax_parquet_test_dir, "tmax", "test")
tmin_train_final_path = aggregate_to_parquet(tmin_daily_train_dir, tmin_parquet_train_dir, "tmin", "train")
tmin_test_final_path = aggregate_to_parquet(tmin_daily_test_dir, tmin_parquet_test_dir, "tmin", "test")


# %%
# ── 7. Clean up intermediate files ──────────────────────────────────────────────

logging.info("=" * 60)
logging.info("Cleaning up intermediate files and directories...")
logging.info("=" * 60)

dirs_to_remove = [
    tmax_daily_train_dir,
    tmin_daily_train_dir,
    tmax_daily_test_dir,
    tmin_daily_test_dir,
    tmax_parquet_train_dir,
    tmin_parquet_train_dir,
    tmax_parquet_test_dir,
    tmin_parquet_test_dir,
    static_dir
]

for d in dirs_to_remove:
    if os.path.exists(d):
        shutil.rmtree(d)
        logging.info(f"Removed directory: {d}")

logging.info("Cleanup complete. Pipeline finished successfully.")
