# %%
# download mean daily temperature from CHIRTS-ERA5 for Nepal 
# v2 - 2026-08-12 - Updating date rangre for final run, using chirts-era5L for coverage

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
import rioxarray as rxr
import geopandas as gpd
from tqdm import tqdm
from colorama import Fore, Style, init

init(autoreset=True)

base_url_tmax = "https://data.chc.ucsb.edu/experimental/CHIRTS-ERA5/tmax/netcdf/daily/"
base_url_tmin = "https://data.chc.ucsb.edu/experimental/CHIRTS-ERA5/tmin/netcdf/daily/"

base_out_dir = r"..\Data_Temp\Datasets\CHIRTS"
temp_dir = os.path.join(base_out_dir, r"Temp")
os.makedirs(base_out_dir, exist_ok=True)
os.makedirs(temp_dir, exist_ok=True)

out_dir_nc = os.path.join(temp_dir, "1_nc")
out_dir_tif = os.path.join(temp_dir, "2_tif")
out_dir_clip_reproject = os.path.join(temp_dir, "3_clip_reproject")

os.makedirs(out_dir_nc, exist_ok=True)
os.makedirs(out_dir_tif, exist_ok=True)
os.makedirs(out_dir_clip_reproject, exist_ok=True)

roi = gpd.read_file(r"C:\Users\A S U S\Documents\Study\ClimateData\CD-Workspace\Data\Datasets\roi_nepal_rectangle\nepal_rectangle_shape.shp")

log_dir = os.path.join(base_out_dir, "Logs")
os.makedirs(log_dir, exist_ok=True)
log_filename = os.path.join(log_dir, f"chirts_pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler(log_filename, mode='w'), logging.StreamHandler()]
)

# %%
# Step 1: Downloading 

def get_file_links(base_url: str, keyword: str):
    logging.info(Fore.BLUE + f"Listing files from {base_url} (keyword={keyword})")
    try:
        r = requests.get(base_url)
        r.raise_for_status()
        nc_files = re.findall(r'href="(.*?\.nc)"', r.text)
        
        filtered = []
        for x in nc_files:
            if keyword in x:
                match = re.search(r'.(\d{4})\.nc$', x)
                if match:
                    year = int(match.group(1))
                    # Download 2000 to cleanly capture overflow hours for Jan 1st 2001
                    if 2001 <= year <= 2003:
                        filtered.append(x)
        
        logging.info(Fore.BLUE + f"Found {len(filtered)} matching files (2000-2025)")
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
        logging.info(Style.RESET_ALL + f"{progress_prefix} Skipping (Already downloaded): {filename}")
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

# %%
# Step 2: Extraction

def process_single_nc_file(
    nc: str, 
    in_nc_dir: str, 
    out_tif_dir: str, 
    file_prefix: str, 
    x_dim: str = "x", 
    y_dim: str = "y", 
    crs: str = "EPSG:4326",
    dimension: str = "time"
):
    nc_full_path = os.path.join(in_nc_dir, nc)
    try:
        logging.info(Fore.BLUE + f"Processing {nc}...")
        
        with xr.open_dataset(nc_full_path) as nc_ds:
            time_values = nc_ds[dimension].values
            
            # Identify the actual internal data variable name (e.g. 'maximum_temperature') to slice
            internal_var = list(nc_ds.data_vars)[0]

            for index, time in enumerate(time_values):
                # Extract YYYYMMDD string without hyphens
                date_str = str(np.datetime64(time))[:10].replace("-", "")
                
                # Combine user prefix with the formatted date
                outpath_tif = os.path.join(out_tif_dir, f"{file_prefix}_{date_str}.tif")
                
                if os.path.exists(outpath_tif):
                    continue

                # Dynamically slice and squeeze out the 'band' dimension
                arr_slice = nc_ds[internal_var].isel({dimension: index}).squeeze()
                
                arr_slice.rio.set_spatial_dims(x_dim=x_dim, y_dim=y_dim, inplace=True)
                arr_slice.rio.write_crs(crs, inplace=True)
                arr_slice.rio.to_raster(outpath_tif)

        logging.info(Fore.GREEN + f"Finished extracting {nc}")

    except Exception as e:
        logging.error(Fore.RED + f"Failed to process {nc}: {e}")

def extract_nc_subdatasets(
    in_nc_dir: str, 
    out_tif_dir: str, 
    file_prefix: str, 
    x_dim: str = "x", 
    y_dim: str = "y", 
    crs: str = "EPSG:4326", 
    dimension: str = "time"
):
    os.makedirs(out_tif_dir, exist_ok=True)
    nc_file_list = [nc for nc in os.listdir(in_nc_dir) if nc.endswith(".nc")]
    
    if not nc_file_list:
        logging.warning(Fore.YELLOW + f"No NC files found in {in_nc_dir}")
        return
    
    max_workers = min(len(nc_file_list), os.cpu_count() or 1)
    logging.info(f"Starting parallel extraction with {max_workers} workers...")
    
    process_func = functools.partial(
        process_single_nc_file,
        in_nc_dir=in_nc_dir,
        out_tif_dir=out_tif_dir,
        file_prefix=file_prefix,
        x_dim=x_dim,
        y_dim=y_dim,
        crs=crs,
        dimension=dimension
    )

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        executor.map(process_func, nc_file_list)

# %%
# Step 3: Clipping and Reprojection

def process_single_tif(tif, in_tif_dir, out_tif_dir, clip_geom_wkt, clip_geom_crs, out_crs, ref_tif_path):
    # tif filename is already formatted as "CHIRTS_tmax_YYYYMMDD.tif" from step 2
    out_tif_path = os.path.join(out_tif_dir, tif)
    if os.path.exists(out_tif_path):
        return

    try:
        ref_tif = rxr.open_rasterio(ref_tif_path)
        ds = rxr.open_rasterio(os.path.join(in_tif_dir, tif))
        
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
        ref_tif = rxr.open_rasterio(ref_tif_path)
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
        in_tif_dir=in_tif_dir, 
        out_tif_dir=out_tif_dir, 
        clip_geom_wkt=clip_geom.geometry.values, 
        clip_geom_crs=clip_geom.crs, 
        out_crs=out_crs, 
        ref_tif_path=temp_ref_path
    )

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_func, tif) for tif in tif_file_list]
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Clipping/Reprojecting"):
            future.result()

    if os.path.exists(temp_ref_path):
        os.remove(temp_ref_path)

# %%
# Main Execution Block

if __name__ == '__main__':
    logging.info('Starting CHIRTS pipeline (Parallel) -- Tmax')

    logging.info('--- Step 1: Downloading NetCDF Files for Tmax ---')
    files_to_download = get_file_links(base_url=base_url_tmax, keyword="Tmax")
    total_files = len(files_to_download)
    for i, filename in enumerate(files_to_download, start=1):
        download_nc(filename, base_url_tmax, out_dir_nc, idx=i, total=total_files)
    
    logging.info('--- Step 2: Extracting Hourly TIFs (Parallel) ---')
    extract_nc_subdatasets(
        in_nc_dir=out_dir_nc, 
        out_tif_dir=out_dir_tif,
        file_prefix="CHIRTS_tmax" 
    )

    logging.info('--- Step 3: Clipping and Reprojecting (Parallel) ---')
    clip_reproject_tif(
        in_tif_dir=out_dir_tif,                   
        out_tif_dir=out_dir_clip_reproject,       
        clip_geom=roi, 
        out_crs="EPSG:4326"
    )

    logging.info(Fore.GREEN + "Pipeline Finished Successfully!")