from osgeo import gdal
import xarray as xr
import numpy as np
import rioxarray
import os
from datetime import datetime
import requests
import logging
import re
from pathlib import Path
from tqdm import tqdm
import geopandas as gpd
from colorama import Fore, Style
import concurrent.futures
import functools

##########################################################################
"""
███████╗███████╗████████╗██╗   ██╗██████╗ 
██╔════╝██╔════╝╚══██╔══╝██║   ██║██╔══██╗
███████╗█████╗     ██║   ██║   ██║██████╔╝
╚════██║██╔══╝     ██║   ██║   ██║██╔═══╝ 
███████║███████╗   ██║   ╚██████╔╝██║     
╚══════╝╚══════╝   ╚═╝    ╚═════╝ ╚═╝     
"""
#########################################################################
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
out_dir_nc = os.path.join(base_dir, "Data", "Datasets", "Raw", "har_datasets", "1_nc_prcp") #original
out_dir_hr = os.path.join(base_dir, "Data", "Datasets", "Raw", "har_datasets", "2_hr_prcp") #original subdatasets
out_dir_hr_clip_reproject = os.path.join(base_dir, "Data", "Datasets", "Raw", "har_datasets", "3_hr_clp_reprj") #agg daily nepal time
out_dir_daily = os.path.join(base_dir, "Data", "Datasets", "Raw", "har_datasets", "4_daily_prcp") #final clipped wgs84
roi = gpd.read_file(r"E:\work\climate_downscaling\Data\Datasets\roi_nepal_rectangle.zip")
roi.to_crs(crs=har_crs, inplace=True)

log_dir = os.path.join(base_dir, "Data", "Datasets", "Logs")
os.makedirs(log_dir, exist_ok=True)
log_filename = os.path.join(log_dir, f"har_pipeline_parallel_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename, mode='w'), 
        logging.StreamHandler()                      
    ]
)

logging.info(f"📝 Logging started. Saving to: {os.path.abspath(log_filename)}")
#####################################################################
"""
███╗   ██╗███████╗████████╗ ██████╗██████╗ ███████╗    ██████╗ ██╗    ██╗███╗   ██╗██╗     ██████╗ 
████╗  ██║██╔════╝╚══██╔══╝██╔════╝██╔══██╗██╔════╝    ██╔══██╗██║    ██║████╗  ██║██║     ██╔══██╗
██╔██╗ ██║█████╗     ██║   ██║     ██║  ██║█████╗      ██║  ██║██║ █╗ ██║██╔██╗ ██║██║     ██║  ██║
██║╚██╗██║██╔══╝     ██║   ██║     ██║  ██║██╔══╝      ██║  ██║██║███╗██║██║╚██╗██║██║     ██║  ██║
██║ ╚████║███████╗   ██║   ╚██████╗██████╔╝██║         ██████╔╝╚███╔███╔╝██║ ╚████║███████╗██████╔╝
╚═╝  ╚═══╝╚══════╝   ╚═╝    ╚═════╝╚═════╝ ╚═╝         ╚═════╝  ╚══╝╚══╝ ╚═╝  ╚═══╝╚══════╝╚═════╝ 
"""                                                               
#####################################################################
def get_file_links(base_url: str, keyword: str = "prcp"):
    logging.info(Fore.BLUE + f"Listing files from {base_url} (keyword={keyword})")
    try:
        r = requests.get(base_url)
        r.raise_for_status()
        nc_files = re.findall(r'href="(.*?\.nc)"', r.text)
        
        filtered = []
        for x in nc_files:
            if keyword in x:
                # Extract year from filename like HARv2_d10km_h_2d_prcp_c_1989.nc
                match = re.search(r'_(\d{4})\.nc$', x)
                if match:
                    year = int(match.group(1))
                    if year >= 1989:
                        filtered.append(x)
        
        logging.info(Fore.BLUE+f"Found {len(filtered)} matching files (year >= 1989)")
        return filtered
    except Exception as e:
        logging.error(Fore.RED+f"Failed to fetch file list: {e}")
        return []

def download_nc(filename, base_url, dest, idx=None, total=None):
    dest = Path(dest)
    os.makedirs(dest, exist_ok=True)
    url = base_url + filename
    dest_path = dest / filename
    progress_prefix = f"({idx}/{total}) " if idx and total else ""
    if dest_path.exists():
        logging.info(Style.RESET_ALL+f"{progress_prefix}Skipping (Already downloaded): {filename}")
        return dest_path
    logging.info(Style.RESET_ALL+f"{progress_prefix}Downloading {url}...")

    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            
            total_size = int(r.headers.get('content-length', 0))
            bar_desc = Fore.BLUE+f"{progress_prefix}{filename}"
            with tqdm(total=total_size, unit='iB', unit_scale=True, desc=bar_desc) as bar:
                with open(dest_path, 'wb') as f:
                    for chunk in r.iter_content(chunk_size=10 * 1024 * 1024):
                        if chunk:
                            f.write(chunk)
                            bar.update(len(chunk))
        logging.info(Fore.GREEN+ f"Saved to {dest_path}")
        return dest_path
    except Exception as e:
        logging.error(Fore.RED+f"Download failed for {filename}: {e}")
        return None
####################################################################
"""
██╗  ██╗████████╗██████╗  █████╗  ██████╗████████╗    ███████╗██╗   ██╗██████╗ ██████╗  █████╗ ████████╗ █████╗ 
╚██╗██╔╝╚══██╔══╝██╔══██╗██╔══██╗██╔════╝╚══██╔══╝    ██╔════╝██║   ██║██╔══██╗██╔══██╗██╔══██╗╚══██╔══╝██╔══██╗
 ╚███╔╝    ██║   ██████╔╝███████║██║        ██║       ███████╗██║   ██║██████╔╝██║  ██║███████║   ██║   ███████║
 ██╔██╗    ██║   ██╔══██╗██╔══██║██║        ██║       ╚════██║██║   ██║██╔══██╗██║  ██║██╔══██║   ██║   ██╔══██║
██╔╝ ██╗   ██║   ██║  ██║██║  ██║╚██████╗   ██║       ███████║╚██████╔╝██████╔╝██████╔╝██║  ██║   ██║   ██║  ██║
╚═╝  ╚═╝   ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═╝ ╚═════╝   ╚═╝       ╚══════╝ ╚═════╝ ╚═════╝ ╚═════╝ ╚═╝  ╚═╝   ╚═╝   ╚═╝  ╚═╝
"""
###################################################################
def process_single_nc_file(nc, in_nc_dir, out_tif_dir, x_dim, y_dim, crs, variable_to_extract, dimension, time_offset):
    nc_full_path = os.path.join(in_nc_dir, nc)
    try:
        logging.info(Fore.BLUE+f"Processing {nc}...")
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
            arr_slice.rio.set_spatial_dims(x_dim=x_dim, y_dim = y_dim, inplace=True)
            arr_slice.rio.write_crs(crs, inplace=True)
            arr_slice.rio.to_raster(outpath_tif)
            # logging.info(Fore.BLUE+f"Done for {time_str}") # Reduce logging in parallel to avoid clutter
        logging.info(Fore.GREEN+f"Finished extracting {nc}")
    except Exception as e:
        logging.error(Fore.RED+f"Failed to process {nc}: {e}")

def extract_nc_subdatasets(
        in_nc_dir, 
        out_tif_dir,
        x_dim, 
        y_dim,
        crs,
        variable_to_extract,
        dimension,
        time_offset
        ):
    os.makedirs(out_tif_dir, exist_ok=True)
    nc_file_list = [nc for nc in os.listdir(in_nc_dir) if nc.endswith(".nc")]
    if not nc_file_list:
        logging.warning(Fore.YELLOW+f"No NC files in {in_nc_dir}")
        return
    
    # Use ProcessPoolExecutor for parallel processing
    max_workers = min(len(nc_file_list), os.cpu_count() or 1)
    logging.info(f"Starting parallel extraction with {max_workers} workers...")
    
    process_func = functools.partial(
        process_single_nc_file,
        in_nc_dir=in_nc_dir,
        out_tif_dir=out_tif_dir,
        x_dim=x_dim,
        y_dim=y_dim,
        crs=crs,
        variable_to_extract=variable_to_extract,
        dimension=dimension,
        time_offset=time_offset
    )

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        executor.map(process_func, nc_file_list)

########################################################################
"""
 ██████╗██╗     ██╗██████╗     ██████╗ ███████╗██████╗ ██████╗      ██╗
██╔════╝██║     ██║██╔══██╗    ██╔══██╗██╔════╝██╔══██╗██╔══██╗     ██║
██║     ██║     ██║██████╔╝    ██████╔╝█████╗  ██████╔╝██████╔╝     ██║
██║     ██║     ██║██╔═══╝     ██╔══██╗██╔══╝  ██╔═══╝ ██╔══██╗██   ██║
╚██████╗███████╗██║██║         ██║  ██║███████╗██║     ██║  ██║╚█████╔╝
 ╚═════╝╚══════╝╚═╝╚═╝         ╚═╝  ╚═╝╚══════╝╚═╝     ╚═╝  ╚═╝ ╚════╝ 
"""
#########################################################################
def process_single_tif(tif, in_tif_dir, out_tif_dir, clip_geom_wkt, clip_geom_crs, out_crs, ref_tif_path):
    # Reconstruct clip_geom from WKT to avoid pickling issues with complex objects if any
    # But passing geometry values directly is better. 
    # Here we pass the WKT string and CRS to reconstruct the GeoSeries/Geometry
    
    out_tif_path = os.path.join(out_tif_dir, tif)

    if os.path.exists(out_tif_path):
        return

    logging.info(Fore.BLUE+f"Processing {tif}...")
    tif_full_path = os.path.join(in_tif_dir, tif)

    try:
        # Re-open ref_tif inside worker
        ref_tif = rioxarray.open_rasterio(ref_tif_path)
        
        # Reconstruct geometry
        # Assuming clip_geom_wkt is a list of polygons or a single polygon WKT
        # For simplicity, if we passed the geometry object's values (shapely objects), they are picklable.
        # Let's assume we passed the geometry object itself if it's picklable (GeoDataFrame is).
        # To be safe, let's just use the passed geometry values.
        
        ds = rioxarray.open_rasterio(tif_full_path)
        clipped_tif = ds.rio.clip(clip_geom_wkt, clip_geom_crs)
        
        reprojected_tif = clipped_tif.rio.reproject_match(ref_tif)
        reprojected_tif.rio.to_raster(out_tif_path)
    except Exception as e:
        logging.error(Fore.RED + f"Failed to process {tif}: {e}")

def clip_reproject_tif(in_tif_dir, out_tif_dir, clip_geom, out_crs):
    os.makedirs(out_tif_dir, exist_ok=True)
    tif_file_list = [tif for tif in os.listdir(in_tif_dir) if tif.endswith(".tif")]
    
    if not tif_file_list:
        logging.warning(Fore.YELLOW+f"No tif files in {in_tif_dir}")
        return
    
    # Create reference template once
    try:
        ref_tif_path = os.path.join(in_tif_dir, tif_file_list[0])
        ref_tif = rioxarray.open_rasterio(ref_tif_path)
        if clip_geom.crs != ref_tif.rio.crs:
                clip_geom = clip_geom.to_crs(crs=ref_tif.crs)

        ref_tif = ref_tif.rio.clip(clip_geom.geometry.values, clip_geom.crs)
        ref_tif = ref_tif.rio.reproject(out_crs)
        logging.info(Fore.GREEN+f"Ref template created from {tif_file_list[0]} for matching")
        
        # Save ref tif temporarily to be opened by workers, or just use the first file as base and re-do steps?
        # Better: Save the processed reference tif to disk so workers can load it and match against it.
        temp_ref_path = os.path.join(out_tif_dir, "temp_reference.tif")
        ref_tif.rio.to_raster(temp_ref_path)
        
    except Exception as e:
        logging.error(Fore.RED+f"Failed to create reference template from {ref_tif_path}: {e}")
        return
    
    # Use ProcessPoolExecutor
    max_workers = os.cpu_count() or 1
    logging.info(f"Starting parallel clipping/reprojecting with {max_workers} workers...")

    # Extract geometry values to pass to workers (picklable)
    clip_geom_values = clip_geom.geometry.values
    clip_geom_crs = clip_geom.crs

    process_func = functools.partial(
        process_single_tif,
        in_tif_dir=in_tif_dir,
        out_tif_dir=out_tif_dir,
        clip_geom_wkt=clip_geom_values,
        clip_geom_crs=clip_geom_crs,
        out_crs=out_crs,
        ref_tif_path=temp_ref_path
    )

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        futures = [executor.submit(process_func, tif) for tif in tif_file_list]
        # Wait for completion and handle exceptions if needed
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures), desc="Clipping/Reprojecting"):
            try:
                future.result()
            except Exception as e:
                logging.error(f"Worker failed: {e}")

    # Clean up temp ref file
    if os.path.exists(temp_ref_path):
        os.remove(temp_ref_path)

#########################################################################
"""
██████╗  █████╗ ██╗██╗  ██╗   ██╗    ██████╗ ██████╗  ██████╗██████╗ 
██╔══██╗██╔══██╗██║██║  ╚██╗ ██╔╝    ██╔══██╗██╔══██╗██╔════╝██╔══██╗
██║  ██║███████║██║██║   ╚████╔╝     ██████╔╝██████╔╝██║     ██████╔╝
██║  ██║██╔══██║██║██║    ╚██╔╝      ██╔═══╝ ██╔══██╗██║     ██╔═══╝ 
██████╔╝██║  ██║██║███████╗██║       ██║     ██║  ██║╚██████╗██║     
╚═════╝ ╚═╝  ╚═╝╚═╝╚══════╝╚═╝       ╚═╝     ╚═╝  ╚═╝ ╚═════╝╚═╝     
"""
#########################################################################
def process_single_day(item, in_dir, out_dir):
    date_str, files = item
    out_path = os.path.join(out_dir, f"prcp_{date_str}.tif")
    
    if os.path.exists(out_path):
        logging.info(f"Skipping {date_str} (already exists)")
        return
    if len(files) != 24:
        logging.warning(Fore.YELLOW+f"The {date_str} does not have 24 hrs precipitation, skipping aggregation")
        return

    logging.info(Fore.BLUE + f"Calculating daily total for {date_str}...")
    
    try:
        daily_total = None
        
        for tif in files:
            tif_path = os.path.join(in_dir, tif)
            da = rioxarray.open_rasterio(tif_path)
            
            if daily_total is None:
                daily_total = da
            else:
                daily_total = daily_total + da
        
        if daily_total is not None:
            daily_total.rio.to_raster(out_path)
            logging.info(Fore.GREEN + f"Saved {out_path}")
            
    except Exception as e:
        logging.error(Fore.RED + f"Failed to calculate daily total for {date_str}: {e}")

def calculate_daily_total_precipitation(in_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    
    # List all TIF files
    tif_files = [f for f in os.listdir(in_dir) if f.endswith('.tif')]
    if not tif_files:
        logging.warning(Fore.YELLOW + f"No TIF files found in {in_dir}")
        return

    # Group files by date (YYYY-MM-DD)
    files_by_date = {}
    for tif in tif_files:
        try:
            date_str = tif.split('_')[-1][:10] 
            if date_str not in files_by_date:
                files_by_date[date_str] = []
            files_by_date[date_str].append(tif)
        except Exception as e:
            logging.warning(Fore.YELLOW + f"Skipping file with unexpected name format {tif}: {e}")

    # Use ProcessPoolExecutor
    max_workers = os.cpu_count() or 1
    logging.info(f"Starting parallel aggregation with {max_workers} workers...")

    process_func = functools.partial(
        process_single_day,
        in_dir=in_dir,
        out_dir=out_dir
    )

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Convert items to list for map
        items = list(sorted(files_by_date.items()))
        executor.map(process_func, items)

if __name__ == '__main__':
    logging.info('Starting HARv2 pipeline (Parallel)')

    #1. Download Netcdf (Keep sequential as it depends on network and server limits)
    logging.info('--- Step 1: Downloading NetCDF Files ---')
    files_to_download = get_file_links(
        base_url=base_url,
        keyword= "prcp",

    )
    total_files = len(files_to_download)
    for i, filename in enumerate(files_to_download, start=1):
        download_nc(
            filename, 
            base_url, 
            out_dir_nc, 
            idx=i, 
            total=total_files
        )
    
    #2. Extraxct Subdatasets
    logging.info('--- Step 2: Extracting Hourly TIFs (Parallel) ---')
    extract_nc_subdatasets(
        in_nc_dir = out_dir_nc, 
        out_tif_dir= out_dir_hr,
        x_dim = "west_east", 
        y_dim = "south_north",
        crs = har_crs,
        variable_to_extract = "prcp",
        dimension = "time",
        time_offset = 6
    )

    #3. Clip and ReProject
    logging.info('--- Step 3: Clipping and Reprojecting (Parallel) ---')
    clip_reproject_tif(
        in_tif_dir=out_dir_hr,
        out_tif_dir=out_dir_hr_clip_reproject,
        clip_geom=roi,
        out_crs="EPSG:4326"
    )

    #4. Calculate Daily Total Precipitation
    logging.info('--- Step 4: Calculating Daily Total Precipitation (Parallel) ---')
    calculate_daily_total_precipitation(
        in_dir=out_dir_hr_clip_reproject,
        out_dir=out_dir_daily
    )

    logging.info(Style.RESET_ALL+'''
        ███████ ██ ███    ██ 
        ██      ██ ████   ██ 
        █████   ██ ██ ██  ██ 
        ██      ██ ██  ██ ██ 
        ██      ██ ██   ████ 
    ''')
