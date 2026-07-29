import os
import re
import logging
import functools
import concurrent.futures
from datetime import datetime
from pathlib import Path
import requests
import numpy as np
import xarray as xr
import rioxarray
import geopandas as gpd
from tqdm import tqdm
from colorama import Fore, Style

# ==========================================
# 1. Configuration & Paths
# ==========================================
har_crs = (
    "+proj=lcc "
    "+lat_1=32.0 "
    "+lat_2=38.0 "
    "+lat_0=31.99998856 "
    "+lon_0=83.0 "
    "+x_0=0.0 "
    "+y_0=0.0 "
    "+a=6370000.0 "
    "+b=6370000.0 "
    "+units=m +no_defs "
    "+type=crs"
)
base_url = "https://data.klima.tu-berlin.de/HAR/v2/d10km/h/2d/"

script_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.dirname(script_dir) 

out_dir_nc = os.path.join(base_dir, "Data_Temp", "Datasets", "Raw", "har_datasets", "1_nc_temp")
out_dir_hr = os.path.join(base_dir, "Data_Temp", "Datasets", "Raw", "har_datasets", "2_hr_temp")
out_dir_hr_clip_reproject = os.path.join(base_dir, "Data_Temp", "Datasets", "Raw", "har_datasets", "3_hr_clp_reprj")
out_dir_daily = os.path.join(base_dir, "Data_Temp", "Datasets", "HAR")

roi = gpd.read_file(r"C:\Users\A S U S\Documents\Study\ClimateData\CD-Workspace\Data\Datasets\roi_nepal_rectangle\nepal_rectangle_shape.shp")
roi.to_crs(crs=har_crs, inplace=True)

log_dir = os.path.join(base_dir, "Data_Temp", "Datasets", "Logs")
os.makedirs(log_dir, exist_ok=True)
log_filename = os.path.join(log_dir, f"har_pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler(log_filename, mode='w'), logging.StreamHandler()]
)

# ==========================================
# Step 1: Downloading (Includes Year 2000)
# ==========================================
def get_file_links(base_url: str, keyword: str):
    logging.info(Fore.BLUE + f"Listing files from {base_url} (keyword={keyword})")
    try:
        r = requests.get(base_url)
        r.raise_for_status()
        nc_files = re.findall(r'href="(.*?\.nc)"', r.text)
        
        filtered = []
        for x in nc_files:
            if keyword in x:
                match = re.search(r'_(\d{4})\.nc$', x)
                if match:
                    year = int(match.group(1))
                    # Download 2000 to cleanly capture overflow hours for Jan 1st 2001
                    if 2000 <= year <= 2005:
                        filtered.append(x)
        
        logging.info(Fore.BLUE + f"Found {len(filtered)} matching files (2000-2005)")
        return filtered
    except Exception as e:
        logging.error(Fore.RED + f"Failed to fetch file list: {e}")
        return []

def download_nc(filename, base_url, dest, idx=None, total=None):
    dest = Path(dest)
    os.makedirs(dest, exist_ok=True)
    url = base_url + filename
    dest_path = dest / filename
    progress_prefix = f"({idx}/{total}) " if idx and total else ""
    
    if dest_path.exists():
        logging.info(Style.RESET_ALL + f"{progress_prefix}Skipping (Already downloaded): {filename}")
        return dest_path

    logging.info(Style.RESET_ALL + f"{progress_prefix}Downloading {url}...")
    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            total_size = int(r.headers.get('content-length', 0))
            bar_desc = Fore.BLUE + f"{progress_prefix}{filename}"
            with tqdm(total=total_size, unit='iB', unit_scale=True, desc=bar_desc) as bar:
                with open(dest_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=10 * 1024 * 1024):
                        if chunk:
                            f.write(chunk)
                            bar.update(len(chunk))
        logging.info(Fore.GREEN + f"Saved to {dest_path}")
        return dest_path
    except Exception as e:
        logging.error(Fore.RED + f"Download failed for {filename}: {e}")
        return None

# ==========================================
# Step 2: Extraction
# ==========================================
def process_single_nc_file(nc, in_nc_dir, out_tif_dir, x_dim, y_dim, crs, variable_to_extract, dimension, time_offset):
    nc_full_path = os.path.join(in_nc_dir, nc)
    try:
        logging.info(Fore.BLUE + f"Processing {nc}...")
        nc_ds = xr.open_dataset(nc_full_path)

        for index, time in enumerate(nc_ds[dimension].values):
            original_time = np.datetime64(time)
            time_str = str(original_time)[:13].replace("T", "-")
            
            if time_offset is not None:
                time_offset_td = np.timedelta64(time_offset, 'h')
                new_time = original_time + time_offset_td
                time_str = str(new_time)[:13].replace("T", "-")

            outpath_tif = os.path.join(out_tif_dir, f"{variable_to_extract}_{time_str}.tif")
            
            if os.path.exists(outpath_tif):
                continue

            arr_slice = nc_ds[variable_to_extract].isel(time=index)
            arr_slice.rio.set_spatial_dims(x_dim=x_dim, y_dim=y_dim, inplace=True)
            arr_slice.rio.write_crs(crs, inplace=True)
            arr_slice.rio.to_raster(outpath_tif)

        logging.info(Fore.GREEN + f"Finished extracting {nc}")
    except Exception as e:
        logging.error(Fore.RED + f"Failed to process {nc}: {e}")

def extract_nc_subdatasets(in_nc_dir, out_tif_dir, x_dim, y_dim, crs, variable_to_extract, dimension, time_offset):
    os.makedirs(out_tif_dir, exist_ok=True)
    nc_file_list = [nc for nc in os.listdir(in_nc_dir) if nc.endswith(".nc")]
    if not nc_file_list:
        logging.warning(Fore.YELLOW + f"No NC files in {in_nc_dir}")
        return
    
    max_workers = min(len(nc_file_list), os.cpu_count() or 1)
    logging.info(f"Starting parallel extraction with {max_workers} workers...")
    
    process_func = functools.partial(
        process_single_nc_file,
        in_nc_dir=in_nc_dir, out_tif_dir=out_tif_dir, x_dim=x_dim, y_dim=y_dim, 
        crs=crs, variable_to_extract=variable_to_extract, dimension=dimension, time_offset=time_offset
    )

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        executor.map(process_func, nc_file_list)

# ==========================================
# Step 3: Clipping and Reprojection
# ==========================================
def process_single_tif(tif, in_tif_dir, out_tif_dir, clip_geom_wkt, clip_geom_crs, out_crs, ref_tif_path):
    out_tif_path = os.path.join(out_tif_dir, tif)
    if os.path.exists(out_tif_path):
        return

    try:
        ref_tif = rioxarray.open_rasterio(ref_tif_path)
        ds = rioxarray.open_rasterio(os.path.join(in_tif_dir, tif))
        
        clipped_tif = ds.rio.clip(clip_geom_wkt, clip_geom_crs)
        reprojected_tif = clipped_tif.rio.reproject_match(ref_tif)
        
        reprojected_tif.rio.to_raster(out_tif_path)
    except Exception as e:
        logging.error(Fore.RED + f"Failed to process {tif}: {e}")

def clip_reproject_tif(in_tif_dir, out_tif_dir, clip_geom, out_crs):
    os.makedirs(out_tif_dir, exist_ok=True)
    tif_file_list = [tif for tif in os.listdir(in_tif_dir) if tif.endswith(".tif")]
    
    if not tif_file_list:
        logging.warning(Fore.YELLOW + f"No tif files in {in_tif_dir}")
        return
    
    try:
        ref_tif_path = os.path.join(in_tif_dir, tif_file_list[0])
        ref_tif = rioxarray.open_rasterio(ref_tif_path)
        if clip_geom.crs != ref_tif.rio.crs:
            clip_geom = clip_geom.to_crs(crs=ref_tif.crs)

        ref_tif = ref_tif.rio.clip(clip_geom.geometry.values, clip_geom.crs)
        ref_tif = ref_tif.rio.reproject(out_crs)
        
        temp_ref_path = os.path.join(out_tif_dir, "temp_reference.tif")
        ref_tif.rio.to_raster(temp_ref_path)
    except Exception as e:
        logging.error(Fore.RED + f"Failed to create reference template: {e}")
        return
    
    max_workers = os.cpu_count() or 1
    logging.info(f"Starting parallel clipping/reprojecting with {max_workers} workers...")

    process_func = functools.partial(
        process_single_tif,
        in_tif_dir=in_tif_dir, out_tif_dir=out_tif_dir, clip_geom_wkt=clip_geom.geometry.values, 
        clip_geom_crs=clip_geom.crs, out_crs=out_crs, ref_tif_path=temp_ref_path
    )

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_func, tif) for tif in tif_file_list]
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Clipping/Reprojecting"):
            future.result()

    if os.path.exists(temp_ref_path):
        os.remove(temp_ref_path)

# ==========================================
# Step 4: Daily Tmax and Tmin Calculation
# ==========================================
def process_single_day(item, in_dir, out_dir):
    date_str, files = item
    date_str_formatted = date_str.replace("-", "")
    out_path_max = os.path.join(out_dir, f"HAR_tmax_{date_str_formatted}.tif")
    out_path_min = os.path.join(out_dir, f"HAR_tmin_{date_str_formatted}.tif")
    
    if os.path.exists(out_path_max) and os.path.exists(out_path_min):
        return
        
    if len(files) != 24:
        logging.warning(Fore.YELLOW + f"{date_str} only has {len(files)} hrs, skipping aggregation")
        return

    try:
        das = []
        for tif in files:
            tif_path = os.path.join(in_dir, tif)
            da = rioxarray.open_rasterio(tif_path)
            das.append(da)
        
        if das:
            combined = xr.concat(das, dim='time')
            
            if not os.path.exists(out_path_max):
                # Scientific conversion: Kelvin -> Celsius (- 273.15)
                daily_max = combined.max(dim='time') - 273.15
                daily_max.rio.write_crs(das[0].rio.crs, inplace=True)
                daily_max.rio.to_raster(out_path_max)
                logging.info(Fore.GREEN + f"Saved daily tmax for {date_str}")
                
            if not os.path.exists(out_path_min):
                # Scientific conversion: Kelvin -> Celsius (- 273.15)
                daily_min = combined.min(dim='time') - 273.15
                daily_min.rio.write_crs(das[0].rio.crs, inplace=True)
                daily_min.rio.to_raster(out_path_min)
                logging.info(Fore.GREEN + f"Saved daily tmin for {date_str}")
            
    except Exception as e:
        logging.error(Fore.RED + f"Failed to calculate daily tmax/tmin for {date_str}: {e}")

def calculate_daily_tmax_tmin_temperature(in_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    tif_files = [f for f in os.listdir(in_dir) if f.endswith('.tif')]
    
    if not tif_files:
        logging.warning(Fore.YELLOW + f"No TIF files found in {in_dir}")
        return

    files_by_date = {}
    for tif in tif_files:
        try:
            date_str = tif.split('_')[-1][:10] 
            year = int(date_str.split('-')[0])
            
            # Use year 2000 hours to fill early 2001, but don't generate year 2000 daily files
            if year < 2001 or year > 2005:
                continue
                
            if date_str not in files_by_date:
                files_by_date[date_str] = []
            files_by_date[date_str].append(tif)
        except Exception as e:
            logging.warning(Fore.YELLOW + f"Skipping file {tif}: {e}")

    max_workers = os.cpu_count() or 1
    logging.info(f"Starting parallel aggregation with {max_workers} workers...")

    process_func = functools.partial(process_single_day, in_dir=in_dir, out_dir=out_dir)

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        items = list(sorted(files_by_date.items()))
        executor.map(process_func, items)

# ==========================================
# Main Execution Block
# ==========================================
if __name__ == '__main__':
    logging.info('Starting HARv2 pipeline (Parallel)')

    logging.info('--- Step 1: Downloading NetCDF Files ---')
    files_to_download = get_file_links(base_url=base_url, keyword="t2_")
    total_files = len(files_to_download)
    for i, filename in enumerate(files_to_download, start=1):
        download_nc(filename, base_url, out_dir_nc, idx=i, total=total_files)
    
    logging.info('--- Step 2: Extracting Hourly TIFs (Parallel) ---')
    extract_nc_subdatasets(
        in_nc_dir=out_dir_nc, out_tif_dir=out_dir_hr,
        x_dim="west_east", y_dim="south_north", crs=har_crs,
        variable_to_extract="t2", dimension="time", time_offset=6
    )

    logging.info('--- Step 3: Clipping and Reprojecting (Parallel) ---')
    clip_reproject_tif(
        in_tif_dir=out_dir_hr, out_tif_dir=out_dir_hr_clip_reproject,
        clip_geom=roi, out_crs="EPSG:4326"
    )

    logging.info('--- Step 4: Calculating Daily Tmax and Tmin (Parallel) ---')
    calculate_daily_tmax_tmin_temperature(
        in_dir=out_dir_hr_clip_reproject, out_dir=out_dir_daily
    )

    logging.info(Fore.GREEN + "Pipeline Finished Successfully!")