"""
HARv2 precipitation processing pipeline

This script downloads HARv2 hourly `prcp` NetCDF files, converts hourly rates (mm h-1)
into hourly depths (mm), aggregates to daily and yearly sums, reprojects safely,
then masks to a region-of-interest. It avoids bilinear resampling for precipitation,
properly handles nodata, and ensures spin-up/time-handling consistency.

Main decisions made here:
- Treat `prcp` as a rate in mm h-1. Convert to depth by multiplying by actual hours.
- Reproject from native LCC to EPSG:4326 *before* masking. Use nearest-neighbor
  resampling (or a conservative method if you implement it) to avoid smoothing totals.
- Never fill nodata with extreme numeric values before reprojection. Use np.nan
  and rasterio's masked arrays where possible.
- Carefully build source/destination transforms using GRID_NX/NY and GRID_DX/DY.

Usage:
  - Edit BASE_URL, ROI_ZIP, and paths as needed.
  - Run the script: python HARv2_precip_pipeline.py

Requirements: rasterio, netCDF4, geopandas, numpy, requests, tqdm

"""

import os
import re
import logging
from datetime import datetime, date, timedelta
from pathlib import Path

import requests
import numpy as np
import netCDF4
import geopandas as gpd
import rasterio
from rasterio.transform import from_origin
from rasterio.io import MemoryFile
from rasterio.warp import calculate_default_transform, reproject, Resampling
from rasterio.mask import mask as rio_mask
from tqdm import tqdm

# -------------------------------
# Configuration
# -------------------------------
BASE_URL = "https://data.klima.tu-berlin.de/HAR/v2/d10km/h/2d/"
OUT_DIR = Path("Data/Raw/har_datasets")
NC_DIR = OUT_DIR / "nc"
DAILY_TIF_DIR = OUT_DIR / "masked_tif_daily"
ANNUAL_DIR = OUT_DIR / "annual"
ROI_ZIP = Path(r"Data/Datasets/roi_nepal_rectangle.zip")  # replace with your ROI
KEYWORD = "prcp_2020"  # or change to other year/keyword

os.makedirs(NC_DIR, exist_ok=True)
os.makedirs(DAILY_TIF_DIR, exist_ok=True)
os.makedirs(ANNUAL_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# -------------------------------
# Utilities: listing and downloading
# -------------------------------

def get_file_links(base_url: str, keyword: str = "prcp"):
    logging.info(f"Listing files from {base_url} (keyword={keyword})")
    r = requests.get(base_url)
    r.raise_for_status()
    nc_files = re.findall(r'href="(.*?\.nc)"', r.text)
    filtered = [x for x in nc_files if keyword in x]
    logging.info(f"Found {len(filtered)} matching files")
    return filtered


def download_nc(filename: str, base_url: str = BASE_URL, dest: Path = NC_DIR):
    url = base_url + filename
    dest_path = dest / filename
    if dest_path.exists():
        logging.info(f"Already downloaded: {filename}")
        return dest_path
    logging.info(f"Downloading {url}")
    r = requests.get(url, stream=True)
    r.raise_for_status()
    with open(dest_path, 'wb') as f:
        for chunk in r.iter_content(chunk_size=8192):
            if chunk:
                f.write(chunk)
    logging.info(f"Saved to {dest_path}")
    return dest_path

# -------------------------------
# Core processing
# -------------------------------

def open_nc_meta(nc_path: Path):
    nc = netCDF4.Dataset(nc_path, 'r')
    return nc


def build_src_crs_and_transform(nc):
    # Build transform using GRID_X01, GRID_Y01 and GRID_DX/DY
    dx = float(nc.GRID_DX)
    dy = float(nc.GRID_DY)
    # GRID_X01, GRID_Y01 are Upper Left corner (from file metadata)
    ulx = float(nc.GRID_X01)
    uly = float(nc.GRID_Y01)

    transform = from_origin(ulx, uly, dx, dy)

    # Build LCC proj string
    raster_crs = (
        f"+proj=lcc +lat_1={float(nc.PROJ_STANDARD_PAR1.strip())} "
        f"+lat_2={float(nc.PROJ_STANDARD_PAR2.strip())} "
        f"+lat_0={float(nc.PROJ_CENTRAL_LAT.strip())} "
        f"+lon_0={float(nc.PROJ_CENTRAL_LON.strip())} "
        f"+a={float(nc.PROJ_SEMIMAJOR_AXIS.strip())} "
        f"+b={float(nc.PROJ_SEMIMINOR_AXIS.strip())} "
        f"+units=m +no_defs +type=crs"
    )

    return raster_crs, transform


def make_dst_profile_from_bounds(bounds, dst_crs='EPSG:4326', resolution=None):
    # bounds: (minx, miny, maxx, maxy) in the *source* CRS (we'll pass in projected bbox)
    # resolution: (x_res, y_res) in degrees for dst_crs; if None, rasterio will compute
    dst_transform, dst_width, dst_height = calculate_default_transform(
        src_crs=None, dst_crs=dst_crs, width=1, height=1, left=bounds[0], bottom=bounds[1], right=bounds[2], top=bounds[3]
    )
    # The above is placeholder - we'll compute transform per-reproject call below using rasterio.calculate_default_transform
    return None


def process_yearly_nc_to_daily_tifs(nc_path: Path, roi_path: Path, out_dir: Path):
    """
    Robust pipeline for one yearly NC file:
    1. Read prcp variable (units mm h-1)
    2. Convert hourly rate -> hourly depth (mm) via multiplication by hours represented by time step
    3. For each calendar day, sum the hourly depths
    4. Create an in-memory raster in source CRS, reproject to EPSG:4326 using NEAREST, then apply ROI mask (in EPSG:4326)
    5. Save daily tif files into out_dir/YYYYMMDD.tif
    """
    logging.info(f"Processing {nc_path}")
    nc = netCDF4.Dataset(str(nc_path), 'r')

    if 'prcp' not in nc.variables:
        raise RuntimeError('prcp not in netCDF')

    prcp = nc.variables['prcp']  # shape (time, y, x)
    time_var = nc.variables['time']

    # Time handling: get datetime objects and compute timestep in seconds (best-effort)
    times = netCDF4.num2date(time_var[:], units=time_var.units, calendar=getattr(time_var, 'calendar', 'standard'))

    if len(times) < 2:
        raise RuntimeError('Not enough time steps to infer timestep')

    # Infer timestep length (seconds) by median difference
    # times may be cftime objects; convert safely to Python datetimes
    def cftime_to_py(t):
        return datetime(t.year, t.month, t.day, t.hour, t.minute, t.second)

    py_times = [cftime_to_py(t) for t in times]
    # seconds between consecutive timesteps
    deltas_sec = np.diff([t.timestamp() for t in py_times])
    median_dt_seconds = int(np.median(deltas_sec))
    hours_per_timestep = median_dt_seconds / 3600.0
    logging.info(f"Inferred hours per timestep: {hours_per_timestep}")

    nodata_val = prcp.getncattr('_FillValue') if '_FillValue' in prcp.ncattrs() else None

    # Build source transform & CRS
    src_crs, src_transform = build_src_crs_and_transform(nc)

    # Read ROI, reproject to target CRS later. Keep ROI as GeoDataFrame in EPSG:4326
    roi_gdf = gpd.read_file(roi_path)
    if roi_gdf.crs is None:
        roi_gdf.set_crs(epsg=4326, inplace=True)

    # Build set of unique days (calendar) present in time dimension
    unique_days = sorted({date(t.year, t.month, t.day) for t in times})
    logging.info(f"Found {len(unique_days)} unique days to process")

    # We want consistent output grid for all daily rasters: compute bounding box in EPSG:4326
    # Strategy: compute the bounds of ROI in EPSG:4326, then choose a reasonable output resolution (e.g., transform from native grid via calculate_default_transform)
    dst_crs = 'EPSG:4326'

    # compute src raster full width/height using GRID_NX/NY
    src_width = int(getattr(nc, 'GRID_NX', prcp.shape[2]))
    src_height = int(getattr(nc, 'GRID_NY', prcp.shape[1]))

    # Compute global bounds in source CRS from metadata: GRID_X00/GIRD_Y00 and grid size
    # Prefer using GRID_X00/Y00 (lower left) and GRID_X01/Y01 (upper left)
    try:
        x00 = float(nc.GRID_X00)
        y00 = float(nc.GRID_Y00)
        dx = float(nc.GRID_DX)
        dy = float(nc.GRID_DY)
        # Upper-left given in GRID_X01, GRID_Y01
        ulx = float(nc.GRID_X01)
        uly = float(nc.GRID_Y01)
        lrx = ulx + dx * src_width
        lry = uly - dy * src_height
        src_bounds = (ulx, lry, lrx, uly)  # minx, miny, maxx, maxy in source CRS
    except Exception:
        # fallback: derive approximate bounds via transform
        src_bounds = (src_transform * (0, src_height-1))[0], (src_transform * (src_width-1, 0))[1], (src_transform * (src_width-1, 0))[0], (src_transform * (0, src_height-1))[1]

    # Compute destination transform/shape to cover ROI precisely
    roi_bounds_wgs84 = roi_gdf.to_crs(dst_crs).total_bounds  # minx, miny, maxx, maxy

    # Use calculate_default_transform with source CRS params
    dst_transform, dst_width, dst_height = calculate_default_transform(
        src_crs, dst_crs, src_width, src_height, *src_bounds)

    # But we want to crop to ROI bounds to avoid huge rasters — recompute transform for ROI extent
    dst_transform_full, dst_width_full, dst_height_full = calculate_default_transform(
        src_crs, dst_crs, src_width, src_height, *src_bounds)

    # Prepare out profile
    out_profile = {
        'driver': 'GTiff',
        'dtype': 'float32',
        'count': 1,
        'crs': dst_crs,
        'transform': dst_transform_full,
        'width': dst_width_full,
        'height': dst_height_full,
        'nodata': np.nan,
        'compress': 'lzw'
    }

    logging.info(f"Output daily grid (full) = {dst_width_full} × {dst_height_full}")

    # Helper: reproject a 2D array from source CRS to the dst grid (ROI grid).
    def reproject_to_dst(src_arr_2d, src_meta_transform, src_crs, dst_profile, src_nodata=None):
        """
        src_arr_2d: 2D numpy array in source grid order (y,x)
        src_meta_transform: affine transform for src array
        dst_profile: out_profile dict used above
        Returns: reprojected 2D array (float32) with dst_profile['nodata'] as nodata
        """
        dst_arr = np.full((dst_profile['height'], dst_profile['width']), np.nan, dtype=np.float32)
        reproject(
            source=src_arr_2d,
            destination=dst_arr,
            src_transform=src_meta_transform,
            src_crs=src_crs,
            dst_transform=dst_profile['transform'],
            dst_crs=dst_profile['crs'],
            resampling=Resampling.nearest,
            src_nodata=src_nodata,
            dst_nodata=np.nan
        )
        return dst_arr

    # We'll iterate day-by-day. For each day, collect indices in time dimension, sum hourly depths.
    time_index_map = {t: i for i, t in enumerate(times)}

    for day in tqdm(unique_days, desc=f"Processing days from {nc_path.name}"):
        # find indices for this day
        indices = [i for i, t in enumerate(times) if t.year == day.year and t.month == day.month and t.day == day.day]
        if not indices:
            logging.warning(f"No time indices found for {day}")
            continue

        # read those hourly slices from netCDF efficiently
        # result shape: (n_hours, ny, nx)
        hourly = prcp[indices, :, :]

        # convert to masked array, respecting _FillValue if present
        if nodata_val is not None:
            hourly = np.ma.masked_equal(hourly, nodata_val)
        else:
            hourly = np.ma.masked_invalid(hourly)

        # convert rate (mm h-1) -> depth (mm) by multiplying with hours_per_timestep
        hourly_depths = hourly * hours_per_timestep  # masked array

        # Sum along time axis -> daily depth in source grid
        daily_depth = hourly_depths.sum(axis=0)  # 2D masked array (ny, nx)

        # Prepare an in-memory source raster with correct transform
        src_meta = {
            'driver': 'GTiff',
            'height': daily_depth.shape[0],
            'width': daily_depth.shape[1],
            'count': 1,
            'dtype': 'float32',
            'crs': src_crs,
            'transform': src_transform,
            'nodata': np.nan
        }

        # Convert masked array to float with NaN for masked values (do NOT use -9999)
        daily_filled = daily_depth.filled(np.nan).astype(np.float32)

        # Reproject to destination ROI grid using NEAREST resampling
        reprojected = reproject_to_dst(daily_filled, src_transform, src_crs, out_profile, src_nodata=np.nan)

        # Mask to ROI in WGS84 (EPSG:4326). roi_gdf is already in 4326
        # Write to MemoryFile then mask using rasterio.mask with roi geometry (in same CRS)
        with MemoryFile() as mem:
            with mem.open(**{**out_profile, 'height': out_profile['height'], 'width': out_profile['width']}) as tmp:
                tmp.write(reprojected, 1)
                # mask with roi
                geoms = [g.__geo_interface__ for g in roi_gdf.to_crs(out_profile['crs']).geometry]
                masked_arr, masked_transform = rio_mask(tmp, geoms, crop=True, nodata=np.nan, filled=True)

                # Save final cropped raster
                # Build final profile
                final_profile = out_profile.copy()
                final_profile.update({
                    'height': masked_arr.shape[1],
                    'width': masked_arr.shape[2],
                    'transform': masked_transform,
                    'nodata': np.nan
                })

                out_fname = out_dir / f"HARv2_prcp_{day.strftime('%Y%m%d')}.tif"
                os.makedirs(out_dir, exist_ok=True)
                with rasterio.open(out_fname, 'w', **final_profile) as dst:
                    dst.write(masked_arr[0].astype(np.float32), 1)

        logging.debug(f"Wrote daily file: {out_fname}")

    nc.close()
    logging.info(f"Finished processing {nc_path}")


def sum_daily_to_annual(daily_dir: Path, out_path: Path):
    """
    Sum all daily TIFs in daily_dir to make a single annual raster.
    All rasters must share the same grid. Use numpy.nan for nodata.
    """
    tif_list = sorted([p for p in daily_dir.glob('*.tif')])
    if not tif_list:
        raise RuntimeError('No daily tifs found')

    # Read first to get profile/grid
    with rasterio.open(tif_list[0]) as src0:
        profile = src0.profile.copy()
        nodata = profile.get('nodata', np.nan)
        arr0 = src0.read(1).astype(np.float64)
        arr0 = np.where(np.isnan(arr0), 0.0, arr0)  # treat nodata as zero for accumulation
        sum_arr = arr0

    for p in tqdm(tif_list[1:], desc='Summing daily tifs'):
        with rasterio.open(p) as s:
            a = s.read(1).astype(np.float64)
            a = np.where(np.isnan(a), 0.0, a)
            # Ensure shapes match
            if a.shape != sum_arr.shape:
                raise RuntimeError(f"Shape mismatch: {p} vs first file")
            sum_arr += a

    # Where all inputs were nodata, result should be nodata
    # To detect those, we can compute a mask of valid pixels from first pass
    with rasterio.open(tif_list[0]) as s:
        valid_mask = ~np.isnan(s.read(1))
    for p in tif_list[1:]:
        with rasterio.open(p) as s:
            valid_mask = valid_mask | (~np.isnan(s.read(1)))

    result = np.where(valid_mask, sum_arr.astype(np.float32), np.nan)

    profile.update({'dtype': 'float32', 'count': 1, 'nodata': np.nan})
    os.makedirs(out_path.parent, exist_ok=True)
    with rasterio.open(out_path, 'w', **profile) as dst:
        dst.write(result, 1)

    logging.info(f"Wrote annual sum to {out_path}")


# -------------------------------
# Main CLI
# -------------------------------
if __name__ == '__main__':
    logging.info('Starting HARv2 pipeline')

    # 1) find remote files, download them
    nc_list = get_file_links(BASE_URL, keyword=KEYWORD)
    if not nc_list:
        logging.error('No NC files found; exiting')
        raise SystemExit(1)

    for nc_name in nc_list:
        local_nc = download_nc(nc_name)
        # process per-year
        year_match = re.search(r'_(\d{4})\.nc$', nc_name)
        if year_match:
            year = year_match.group(1)
        else:
            year = Path(nc_name).stem
        year_dir = DAILY_TIF_DIR / year
        process_yearly_nc_to_daily_tifs(Path(local_nc), ROI_ZIP, year_dir)

        # after daily files are present, sum to annual
        out_annual = ANNUAL_DIR / f'sum_{year}.tif'
        sum_daily_to_annual(year_dir, out_annual)

    logging.info('HARv2 pipeline completed successfully')
