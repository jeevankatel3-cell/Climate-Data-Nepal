# %%
# Prepare dataframe from daily rasters for prediction over the entire grid
# we use the fitted model from B3

# V1 - 2026-07-28 - create code - jpk
# V2 - 2026-08-11 - added har_tvar_lag1 as additional predictor variable
# V2A - 2026-08-27 - adapted for linux
# WORKFLOW
#  Load all predictor rasters for the day
#  Stack and create a df
#  Format df to fit column requirement of the model
#  Make predictions for tmax or tmin in the dataframe

# %%
# libraries
import os
import gc
import glob
import numpy as np
import pandas as pd 
import xarray as xr
import rioxarray as rxr
import pyarrow as pa    
import pyarrow.parquet as pq
from tqdm import tqdm
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

start_year, start_month, start_day = 2001, 1, 1
end_year,   end_month, end_day   = 2025, 12, 31


static_paths = {
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

template = rxr.open_rasterio(r"../Data/Datasets/CHIRTS/CHIRTS_tmax_20010101.tif")
out_base = fr"../Data/Processed/Daily_bias_correction"

tmax_temp_dir = os.path.join(out_base, 'Temp_tmax')
tmax_dynamic_pred_dir = os.path.join(tmax_temp_dir, 'Dynamic_pred')
tmax_yearly_parquet_dir = os.path.join(tmax_temp_dir, 'Tmax_yearly_parquet')

tmin_temp_dir = os.path.join(out_base, 'Temp_tmin')
tmin_dynamic_pred_dir = os.path.join(tmin_temp_dir, 'Dynamic_pred')
tmin_yearly_parquet_dir = os.path.join(tmin_temp_dir, 'Tmin_yearly_parquet')

static_dir = os.path.join(out_base, "Static_df")

for d in [tmax_temp_dir, tmax_dynamic_pred_dir, tmax_yearly_parquet_dir, 
    static_dir, tmin_temp_dir, tmin_dynamic_pred_dir, tmin_yearly_parquet_dir]:
    if not os.path.exists(d):
        os.makedirs(d)

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
    for the given year-month-day. Missing files are noted but not fatal —
    the loader handles absent paths with NaN fill.
    """
    prev_date = pd.Timestamp(year, month, day) - pd.Timedelta(days=1)
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
        "gldas_wind_speed": fr"../Data/Datasets/GLDAS/wind_speed/GLDAS_wind_speed_{year}{month:02d}{day:02d}.tif",
}    

# %%
SEASONS = {
    'winter':       [12, 1, 2],
    'pre_monsoon':  [3, 4, 5],
    'monsoon':      [6, 7, 8, 9],
    'post_monsoon': [10, 11],
}

def get_season(month):
    for s, months in SEASONS.items():
        if month in months:
            return s
    return None

SEASONS_NUM = {
    1: 'winter',
    2: 'pre_monsoon',
    3: 'monsoon',
    4: 'post_monsoon'
}

def get_season_num(season_str):
    for s_n, s_name in SEASONS_NUM.items():
        if s_name == season_str:
            return s_n
    return None

# %%
# 1. Static feature dataframe 

static_csv = os.path.join(static_dir, "static_df.csv")
 
if os.path.exists(static_csv):
    logging.info("Loading cached static df from CSV...")
    static_df = pd.read_csv(static_csv)
else:
    logging.info("Building static df from rasters. This may take a moment...")
    aligned = []
    for name, path in static_paths.items():
        da = rxr.open_rasterio(path, masked=True).rio.write_crs("EPSG:4326")
        da = da.rio.reproject_match(template).squeeze("band", drop=True)
        da.name = name
        if "spatial_ref" in da.coords:
            da = da.drop_vars("spatial_ref")
        aligned.append(da)
    static_xr = xr.merge(aligned, compat="override")
    static_df  = static_xr.to_dataframe().reset_index()
    static_df.to_csv(static_csv, index=False)
    del aligned, static_xr
    gc.collect()
 
logging.info(f"Static df prepared: {len(static_df):,} pixels")

# %%
# 2. Daily loop to build df

daily_range = pd.period_range(
    start = f"{start_year}-{start_month:02d}-{start_day:02d}",
    end = f"{end_year}-{end_month:02d}-{end_day:02d}",
    freq= "D"
)

for t_variable in ['tmax', 'tmin']:
    logging.info(f"============ Processing {len(daily_range)} days for {t_variable.upper()} ============")
    for period in tqdm(daily_range, desc = f"Daily feature extraction ({t_variable})"):
        year = period.year
        month = period.month
        day = period.day
        label = day_label(year, month, day)
        
        # Write directly to the dynamic pred directory to match Cell 5
        if t_variable == 'tmax':
            pred_out = os.path.join(tmax_dynamic_pred_dir, f"tmax_bc_{label}.parquet")
        else:
            pred_out = os.path.join(tmin_dynamic_pred_dir, f"tmin_bc_{label}.parquet")

        if os.path.exists(pred_out):
            continue

        paths = build_daily_paths(year, month, day, t_variable)
        aligned = []
        missing_vars = []

        for name, path in paths.items():
            if not os.path.exists(path):
                missing_vars.append(name)
                continue
            da = rxr.open_rasterio(path, masked=True).rio.write_crs("EPSG:4326")
            da = da.rio.reproject_match(template).squeeze("band", drop = True)
            da.name = name
            aligned.append(da.load())
            da.close()

        if missing_vars:
            tqdm.write(f" [{label}] WARNING: Missing rasters: {missing_vars}")
        
        if not aligned:
            tqdm.write(f" [{label}] WARNING: No rasters found - skipping")
            continue

        dynamic_xr = xr.merge(aligned, compat='override')
        dynamic_df = dynamic_xr.to_dataframe().reset_index()

        # variable without rasters gets Nan Column
        for name in missing_vars:
            dynamic_df[name] = np.nan

        del aligned, dynamic_xr
        gc.collect()

        #merge with static df
        daily_grid = dynamic_df.merge(static_df, on = ['x', 'y'], how = "inner")

        #Add temporal features - cyclical month columns
        sin_doy, cos_doy = doy_cyclical(year, period.day_of_year)
        daily_grid['doy_sin'] = sin_doy
        daily_grid['doy_cos'] = cos_doy
        daily_grid['year'] = year
        daily_grid['month'] = month
        daily_grid['day'] = day
        daily_grid['season_num'] = (daily_grid['month'].apply(get_season)).apply(get_season_num) 

        # Write directly to Parquet - saves time and space!
        daily_grid.to_parquet(pred_out, index=False)

        del daily_grid
        gc.collect()

# %%
# 3 aggregate monthly parquets to yearly parquet 

def aggregate_to_parquet(daily_dir: str, parquet_dir: str, t_variable: str) -> str:
    """
    Read daily parquets year by year, write to a single yearly parquet.
    Yields a list of yearly parquets.
    """
    parquet_paths = []

    for year in range(start_year, end_year + 1):
        yearly_path = os.path.join(parquet_dir, f"full_{year}.parquet")
        daily_periods = pd.period_range(
            start = f"{year}-01-01", end = f"{year}-12-31", freq = "D"
        )
        expected = [
            os.path.join(daily_dir, f"{t_variable}_bc_{day_label(p.year, p.month, p.day)}.parquet")
            for p in daily_periods
        ]
        found = [f for f in expected if os.path.exists(f)]

        if not found:
            logging.warning(f"[{t_variable}] | {year} - no daily files found, skipping")
            continue

        writer = None
        target_schema = None

        for fpath in found:
            chunk = pd.read_parquet(fpath)
            
            if writer is None:
                table = pa.Table.from_pandas(chunk)
                target_schema = table.schema
                writer = pq.ParquetWriter(yearly_path, target_schema)
            else:
                # 1. Re-order columns to match the first file's schema order
                chunk = chunk[[field.name for field in target_schema]]
                table = pa.Table.from_pandas(chunk)
                # 2. Cast data types (e.g. float vs double) to match the first file's schema
                table = table.cast(target_schema)

            writer.write_table(table)

        if writer:
            writer.close()
            parquet_paths.append(yearly_path)
            logging.info(f"[{t_variable}] {year} - {len(found)} daily files written to parquet")
    
    if not parquet_paths:
        logging.warning(f"[{t_variable}] No data in parquet paths - final df not written")
        return None
        
    logging.info(f"[{t_variable}] Yearly files aggregated successfully in {parquet_dir}.")
    logging.info(f"[{t_variable}] Load entire dataset memory-safely using: pd.read_parquet(r'{parquet_dir}')")
    
    return parquet_paths


logging.info("================ Aggregating to yearly parquet files ================")

tmax_pred_final_paths = aggregate_to_parquet(daily_dir = tmax_dynamic_pred_dir, 
    parquet_dir = tmax_yearly_parquet_dir, 
    t_variable= 'tmax')

tmin_pred_final_paths = aggregate_to_parquet(daily_dir = tmin_dynamic_pred_dir, 
    parquet_dir = tmin_yearly_parquet_dir, 
    t_variable= 'tmin')

logging.info("================ Process Complete! ================")
