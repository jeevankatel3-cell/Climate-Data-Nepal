# Prepare monthly rasters for monthly downscaling model
# V1 - for final run on local server - JPK - 2026-09-16

# libraries
import os
import gc
import glob
import re
import logging
import numpy as np
import pandas as pd
import xarray as xr
import rioxarray as rxr
from rasterio.enums import Resampling
from pathlib import Path

# Configuration
start = pd.Timestamp("2001-01-01")
end = pd.Timestamp("2025-12-31")

dirs = {
    "ERA5L": r"F:/ClimateDataWS/Data/Datasets/ERA5L",
    "HAR": r"F:/ClimateDataWS/Data/Datasets/HAR",
    "CHIRTS": r"F:/ClimateDataWS/Data/Datasets/CHIRTS",
    "CHELSA": r"F:/ClimateDataWS/Data/Datasets/CHELSA",
    "IK": r"F:/ClimateDataWS/Data/Insitu/KrigingDaily_1Km",
    "GLDAS_air_temp": r"F:/ClimateDataWS/Data/Datasets/GLDAS/air_temp",
    "GLDAS_lwdown": r"F:/ClimateDataWS/Data/Datasets/GLDAS/lwdown",
    "GLDAS_specific_humidity": r"F:/ClimateDataWS/Data/Datasets/GLDAS/specific_humidity",
    "GLDAS_surface_pressure": r"F:/ClimateDataWS/Data/Datasets/GLDAS/surface_pressure",
    "GLDAS_swdown": r"F:/ClimateDataWS/Data/Datasets/GLDAS/swdown",
    "GLDAS_wind_speed": r"F:/ClimateDataWS/Data/Datasets/GLDAS/wind_speed",
    "MODIS_evi": r"F:/ClimateDataWS/Data/Datasets/MODIS_2_evi",
    "MODIS_lst_d": r"F:/ClimateDataWS/Data/Datasets/MODIS_2_lst_d",
    "MODIS_lst_n": r"F:/ClimateDataWS/Data/Datasets/MODIS_2_lst_n",
    "MODIS_lst_diff": r"F:/ClimateDataWS/Data/Datasets/MODIS_2_lst_diff"
        
}

output_dir = r"F:/ClimateDataWS/Data/Processed/Monthly_downscaling/Input_rasters"
os.makedirs(output_dir, exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(output_dir, "processing.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

template = rxr.open_rasterio(r"F:/ClimateDataWS/Data/Datasets/DEM_1km/DEM_1km_elevation_max.tif", masked = True).rio.write_crs("EPSG:4326")

# Helpers
def resample_to_template(raster: xr.DataArray) -> xr.DataArray:
    """Reprojects and clips raster to match the 1km WGS84 template"""
    return raster.rio.reproject_match(template, resampling = Resampling.nearest)

def parse_tif_catalog(directory: str, t_variable: str = "", separate: bool = False) -> pd.DataFrame:
    """
    Build a dataframe of (filepath, date) for files in the directory.
    Handles variable subfolders (separate=True), variable prefixes/stems, or general raster directories.
    """
    if separate:
        pattern = os.path.join(directory, t_variable, "*.tif")
    elif t_variable:
        pattern = os.path.join(directory, f"*{t_variable}*.tif")
    else:
        pattern = os.path.join(directory, "*.tif")

    files = sorted(glob.glob(pattern))
    # Filter out auxiliary xml and non-tif files
    files = [f for f in files if f.lower().endswith(".tif") and not f.lower().endswith(".aux.xml")]

    if not files:
        raise FileNotFoundError(f"No .tif files found matching pattern '{pattern}'")

    rows = []
    for f in files:
        stem = Path(f).stem
        match = re.search(r"(\d{8})", stem)
        if match:
            date_str = match.group(1)
            try:
                date = pd.Timestamp(date_str)
                rows.append({"path": f, "date": date})
            except ValueError:
                continue

    if not rows:
        raise FileNotFoundError(f"No valid date-indexed .tif files found for '{t_variable}' in {directory}")

    return pd.DataFrame(rows).sort_values("date").reset_index(drop=True)

def filter_period(catalog: pd.DataFrame, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    return catalog[(catalog["date"] >= start) & (catalog["date"] <= end)].reset_index(drop=True)

def stack_month(catalog: pd.DataFrame, year: int, month: int) -> xr.DataArray | None:
    """
    Load all daily rasters for year/month, resample each, and stack along
    a new 'day' dimension. Returns None if no files found for that month.
    Memory note: opens one file at a time, closes after stacking.
    """
    month_files = catalog[
        (catalog['date'].dt.year == year) &
        (catalog['date'].dt.month == month)      
    ]
    if month_files.empty:
        return None
    
    slices = []
    for _, row in month_files.iterrows():
        da = rxr.open_rasterio(row["path"], masked = True)
        da = resample_to_template(da)
        if "band" in da.dims and da.sizes["band"] == 1:
            da = da.squeeze("band", drop = True)
        da = da.assign_coords(day = row["date"])
        slices.append(da.load())
        da.close()
    
    stack = xr.concat(slices, dim = "day")

    return stack

def save_raster(data: xr.DataArray, out_path: str, crs, nodata: float = -9999):
    """Write a 2D DataArray to GeoTiff, preserving CRS and nodata"""
    data = data.rio.write_crs(crs)
    data = data.rio.write_nodata(nodata)
    data.rio.to_raster(out_path)

def month_label(year: int, month: int) -> str:
    return f"{year}{month:02d}"

# Build a list of (year, month to process)
month_range = pd.period_range(start = start, end = end, freq = "M")


# PROCESSING STACK 1 - TEMPERATURE PRODUCTS
# 1. Temperature products
#  Variables to prepare: Tmax-average, Tmin-average, Tmax-max, Tmin-max, Tmax-min, Tmin-min, Tmax-std, Tmin-std, DTR-Std

temp_products = {
    'ERA5L': dirs['ERA5L'],
    'HAR':dirs['HAR'],
    'CHIRTS': dirs['CHIRTS'],
    'CHELSA': dirs['CHELSA']
}

logger.info("Processing temperature products")

for product, directory in temp_products.items():
    product_out = os.path.join(output_dir, product)
    os.makedirs(product_out, exist_ok= True)

    catalog_tmax = filter_period(parse_tif_catalog(directory, "tmax", False), start, end)
    catalog_tmin = filter_period(parse_tif_catalog(directory, "tmin", False), start, end)
    crs = rxr.open_rasterio(catalog_tmax['path'].iloc[0]).rio.crs
    logger.info("CRS: %s", crs)

    for period in month_range:
        year, month = period.year, period.month
        label = month_label(year, month)

        #prepare file paths
        tmax_average_path = os.path.join(product_out, f"{product}_tmax_average_{label}.tif")
        tmin_average_path = os.path.join(product_out, f"{product}_tmin_average_{label}.tif")


        extra_exist = True
        if product == "CHIRTS":
            
            tmax_max_path = os.path.join(product_out, f"chirts_tmax_max_{label}.tif")
            tmin_max_path = os.path.join(product_out, f"chirts_tmin_max_{label}.tif")

            tmax_min_path = os.path.join(product_out, f"chirts_tmax_min_{label}.tif")
            tmin_min_path = os.path.join(product_out, f"chirts_tmin_min_{label}.tif")

            tmax_std_path = os.path.join(product_out, f"chirts_tmax_std_{label}.tif")
            tmin_std_path = os.path.join(product_out, f"chirts_tmin_std_{label}.tif")

            dtr_std_path = os.path.join(product_out, f"chirts_dtr_std_{label}.tif")

            extra_exist = (os.path.exists(tmax_max_path) 
                            and os.path.exists(tmin_max_path)
                            and os.path.exists(tmax_min_path)
                            and os.path.exists(tmin_min_path)
                            and os.path.exists(tmax_std_path)
                            and os.path.exists(tmin_std_path)
                            and os.path.exists(dtr_std_path)
            )
        
        if os.path.exists(tmax_average_path) and os.path.exists(tmin_average_path) and extra_exist:
            logger.info(f" [{product}] {label} - skipping (exists)")
            continue

        stack_tmax = stack_month(catalog_tmax, year, month)
        stack_tmin = stack_month(catalog_tmin, year, month)
        
        if stack_tmax is None:
            logger.warning(f"No file found for[{product}] Tmax {label}")
            continue
        if stack_tmin is None:
            logger.warning(f"No file found for[{product}] Tmin {label}")
            continue
        
        tmax_monthly_average = stack_tmax.mean(dim = "day", skipna = True)
        tmin_monthly_average = stack_tmin.mean(dim = "day", skipna = True)
        save_raster(tmax_monthly_average, tmax_average_path, crs)
        save_raster(tmin_monthly_average, tmin_average_path, crs)

        #Chirts additional aggregates
        if product.upper() == 'CHIRTS':
            tmax_max = stack_tmax.max(dim = "day", skipna = True)
            tmin_max = stack_tmin.max(dim = "day", skipna = True)

            tmax_min = stack_tmax.min(dim = "day", skipna = True)
            tmin_min = stack_tmin.min(dim = "day", skipna = True)

            tmax_std = stack_tmax.std(dim = "day", skipna = True)
            tmin_std = stack_tmin.std(dim = "day", skipna = True)
            dtr_std = (stack_tmax - stack_tmin).std(dim="day", skipna=True)

#            
            save_raster(tmax_max, tmax_max_path, crs)
            save_raster(tmin_max, tmin_max_path, crs)
            save_raster(tmax_min, tmax_min_path, crs)
            save_raster(tmin_min, tmin_min_path, crs)
            save_raster(tmax_std, tmax_std_path, crs)
            save_raster(tmin_std, tmin_std_path, crs)
            save_raster(dtr_std, dtr_std_path, crs)

        del stack_tmax, stack_tmin
        gc.collect()
        logger.info(f" [{product}] {label} - done")
    

## PROCESSING STACK 2 - INSITU KRIGING DATA
logger.info("Processing insitu kriging data")

ik_tmax_catalog = filter_period(parse_tif_catalog(dirs['IK'], "tmax", True), start, end)
ik_tmin_catalog = filter_period(parse_tif_catalog(dirs['IK'], "tmin", True), start, end)
crs = rxr.open_rasterio(ik_tmax_catalog['path'].iloc[0]).rio.crs
ik_output_dir = os.path.join(output_dir, "IK")
os.makedirs(ik_output_dir, exist_ok=True)

logger.info("CRS: %s", crs)

for period in month_range:
    year = period.year
    month = period.month
    label = month_label(year, month)

    tmax_ik_average_path = os.path.join(ik_output_dir, f"ik_tmax_average_{label}.tif")
    tmin_ik_average_path = os.path.join(ik_output_dir, f"ik_tmin_average_{label}.tif")

    ik_tmax_monthly_average_exist = os.path.exists(tmax_ik_average_path)
    ik_tmin_monthly_average_exist = os.path.exists(tmin_ik_average_path)

    if ik_tmax_monthly_average_exist and ik_tmin_monthly_average_exist:
        logger.info(f" [IK] {label} - skipping (exists)")
        continue
    
    stack_ik_tmax = stack_month(ik_tmax_catalog, year, month)
    stack_ik_tmin = stack_month(ik_tmin_catalog, year, month)

    if stack_ik_tmax is None:
        logger.warning(f"No file found for[IK] Tmax {label}")
        continue
    if stack_ik_tmin is None:
        logger.warning(f"No file found for[IK] Tmin {label}")
        continue

    ik_tmax_monthly_average = stack_ik_tmax.mean(dim = "day", skipna = True)
    ik_tmin_monthly_average = stack_ik_tmin.mean(dim = "day", skipna = True)
    save_raster(ik_tmax_monthly_average, tmax_ik_average_path, crs)
    save_raster(ik_tmin_monthly_average, tmin_ik_average_path, crs)

    del stack_ik_tmax, stack_ik_tmin
    gc.collect()
    logger.info(f" [IK] {label} - done")



## PROCESSING STACK 3 - GLDAS
logger.info("Processing GLDAS data")

MET_PRODUCTS = {
    "GLDAS_air_temp":     (dirs["GLDAS_air_temp"],     "mean"),
    "GLDAS_lwdown":     (dirs["GLDAS_lwdown"],     "sum"),  
    "GLDAS_specific_humidity": (dirs["GLDAS_specific_humidity"], "mean"),
    "GLDAS_surface_pressure":     (dirs["GLDAS_surface_pressure"], "mean"),
    "GLDAS_swdown": (dirs["GLDAS_swdown"], "sum"),
    "GLDAS_wind_speed": (dirs["GLDAS_wind_speed"], "mean"),
}

for product, (directory, agg_type) in MET_PRODUCTS.items():
    product_out = os.path.join(output_dir, product)
    os.makedirs(product_out, exist_ok=True)
 
    catalog = filter_period(parse_tif_catalog(directory), start, end)
    crs = rxr.open_rasterio(catalog["path"].iloc[0]).rio.crs
    logger.info("CRS: %s", crs)

    for period in month_range:
        year, month = period.year, period.month
        label = month_label(year, month)

        if agg_type == 'mean':
            mean_path = os.path.join(product_out, f"{product}_monthly_mean_{label}.tif")
            if os.path.exists(mean_path):
                logger.info(f"  [{product}] {label} — skipping (exists)")
                continue
    
            stack = stack_month(catalog, year, month)
            if stack is None:
                logger.warning(f"  [{product}] {label} — WARNING: no files found, skipping")
                continue
    
            monthly_mean = stack.mean(dim="day", skipna=True)
            save_raster(monthly_mean, mean_path, crs)
    
        elif agg_type == "sum":
            sum_path = os.path.join(product_out, f"{product}_monthly_sum_{label}.tif")
            if os.path.exists(sum_path):
                logger.info(f"  [{product}] {label} — skipping (exists)")
                continue
            stack = stack_month(catalog, year, month)
            monthly_sum = stack.sum(dim="day", skipna=True)
            save_raster(monthly_sum, sum_path, crs)
        
        else:
            continue

        del stack
        gc.collect()
        logger.info(f"  [{product}] {label} — done")


# PROCESSING STACK 4 - MODIS EVI
logger.info("Processing MODIS EVI data")

MODIS_EVI = dirs["MODIS_evi"]
evi_out = os.path.join(output_dir, "MODIS_evi")
os.makedirs(evi_out, exist_ok=True)
evi_catalog = parse_tif_catalog(MODIS_EVI)
evi_catalog["end_date"] = evi_catalog["date"] + pd.Timedelta(days=15)

# Estimate expected composites per month (16-day → ~2 per month)
def expected_composites(year: int, month: int) -> int:
    """Approximate number of 16-day composites expected to overlap a month."""
    days_in_month = pd.Period(f"{year}-{month:02d}", freq="M").days_in_month
    return max(1, round(days_in_month / 16))

crs_evi = rxr.open_rasterio(evi_catalog["path"].iloc[0]).rio.crs
logger.info("CRS: %s", crs_evi)

for period in month_range:
    year, month = period.year, period.month
    label = month_label(year, month)
 
    mean_path    = os.path.join(evi_out, f"evi_monthly_mean_{label}.tif")
    missing_path = os.path.join(evi_out, f"evi_missing_fraction_{label}.tif")
 
    if os.path.exists(mean_path) and os.path.exists(missing_path):
        logger.info(f"  [EVI] {label} — skipping (exists)")
        continue
 
    month_start = pd.Timestamp(year=year, month=month, day=1)
    month_end   = month_start + pd.offsets.MonthEnd(0)
 
    # Composites that overlap this month
    overlapping = evi_catalog[
        (evi_catalog["date"]     <= month_end) &
        (evi_catalog["end_date"] >= month_start)
    ]
 
    n_expected  = expected_composites(year, month)
    n_available = len(overlapping)
    missing_frac = max(0.0, (n_expected - n_available) / n_expected)
 
    if n_available == 0:
        logger.warning(f"  [EVI] {label} — WARNING: no composites found, skipping")
        continue
 
    # Load, resample, and average overlapping composites
    slices = []
    for _, row in overlapping.iterrows():
        da = rxr.open_rasterio(row["path"], masked=True)
        da = resample_to_template(da)
        if "band" in da.dims and da.sizes["band"] == 1:
            da = da.squeeze("band", drop=True)
        slices.append(da.load())
        da.close()
 
    stack   = xr.concat(slices, dim="composite")
    evi_mean = stack.mean(dim="composite", skipna=True)
    save_raster(evi_mean, mean_path, crs_evi)
 
    # Missing fraction raster — spatially constant scalar stored as raster
    missing_arr = xr.full_like(evi_mean, fill_value=float(missing_frac))
    save_raster(missing_arr, missing_path, crs_evi, nodata=-9999)
 
    del stack, slices
    gc.collect()
    logger.info(f"  [EVI] {label} — {n_available}/{n_expected} composites, missing={missing_frac:.2f}")
 
logger.info("\nMonthly raster aggregation complete.")
logger.info(f"Outputs in: {output_dir}")


#PROCESSING STACK 5 - MODIS LST
logger.info("Processing MODIS LST data")
 
lst_out = os.path.join(output_dir, "MODIS_lst")
os.makedirs(lst_out, exist_ok=True)

def stack_month_lst(catalog: pd.DataFrame, year: int, month: int):
    """
    Load LST daily rasters for year/month. Returns (stack, n_expected, n_available).
    stack is None if no files found.
    Missing fraction is computed per-pixel from NaN distribution in the stack,
    not as a scalar — LST cloud gaps are spatially structured.
    """
    month_files = catalog[
        (catalog["date"].dt.year  == year) &
        (catalog["date"].dt.month == month)
    ]
    days_in_month = pd.Period(f"{year}-{month:02d}", freq="M").days_in_month
 
    if month_files.empty:
        return None, days_in_month, 0
 
    slices = []
    for _, row in month_files.iterrows():
        da = rxr.open_rasterio(row["path"], masked=True)
        da = resample_to_template(da)
        if "band" in da.dims and da.sizes["band"] == 1:
            da = da.squeeze("band", drop=True)
        da = da.assign_coords(day=row["date"])
        slices.append(da.load())
        da.close()
 
    stack = xr.concat(slices, dim="day")
    return stack, days_in_month, len(slices)

def process_lst_product(product: str, directory: str, month_range):
    """Process one LST product directory → monthly mean + per-pixel missing fraction."""
    if not os.path.isdir(directory):
        logger.warning(f"  [{product}] Directory not found: {directory} — skipping")
        return
 
    catalog = filter_period(parse_tif_catalog(directory, ""), start, end)
    if catalog.empty:
        logger.warning(f"  [{product}] No files found in period — skipping")
        return
 
    crs = rxr.open_rasterio(catalog["path"].iloc[0]).rio.crs
 
    for period in month_range:
        year, month = period.year, period.month
        label = month_label(year, month)
 
        mean_path    = os.path.join(lst_out, f"{product}_monthly_mean_{label}.tif")
        missing_path = os.path.join(lst_out, f"{product}_missing_fraction_{label}.tif")
 
        if os.path.exists(mean_path) and os.path.exists(missing_path):
            logger.info(f"  [{product}] {label} — skipping (exists)")
            continue
 
        stack, n_expected, n_available = stack_month_lst(catalog, year, month)
 
        if stack is None:
            logger.warning(f"  [{product}] {label} — WARNING: no files found, skipping")
            continue
 
        # Monthly mean — skipna so valid pixels use whatever days are available
        monthly_mean = stack.mean(dim="day", skipna=True)
        save_raster(monthly_mean, mean_path, crs)
 
        # Per-pixel missing fraction:
        #   For each pixel, fraction of the month's calendar days with no valid value.
        #   Two sources of missingness: (a) file missing for that day entirely,
        #   (b) file present but pixel is NaN (cloud/QC mask).
        #
        #   We model (a) by treating entirely absent days as all-NaN planes,
        #   then missing_fraction = (NaN count per pixel) / days_in_month.
        nan_count_in_stack = stack.isnull().sum(dim="day")          # NaN days in stack
        absent_days        = n_expected - n_available                # days with no file at all
        total_nan_days     = nan_count_in_stack + absent_days        # per-pixel total
        missing_fraction   = (total_nan_days / n_expected).clip(0, 1)
        missing_fraction   = missing_fraction.astype(np.float32)
        save_raster(missing_fraction, missing_path, crs, nodata=-9999.0)
 
        del stack
        gc.collect()
 
        mean_missing = float(missing_fraction.mean())
        logger.info(f"  [{product}] {label} — {n_available}/{n_expected} days, mean missing fraction={mean_missing:.2f}")


process_lst_product("lst_d", dirs["MODIS_lst_d"], month_range) 
process_lst_product("lst_n", dirs["MODIS_lst_n"], month_range)
process_lst_product("lst_diff", dirs["MODIS_lst_diff"], month_range)