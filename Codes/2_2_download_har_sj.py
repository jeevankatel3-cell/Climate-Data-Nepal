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
base_dir = os.path.dirname(os.getcwd()) 
out_dir_nc = os.path.join(base_dir, "Data", "Datasets", "Raw", "har_datasets", "1_nc_prcp") #original
out_dir_hr = os.path.join(base_dir, "Data", "Datasets", "Raw", "har_datasets", "2_hr_prcp") #original subdatasets
out_dir_hr_clip_reproject = os.path.join(base_dir, "Data", "Datasets", "Raw", "har_datasets", "3_hr_clp_reprj") #agg daily nepal time
# out_dir_daily_roi = os.path.join(base_dir, "Data", "Datasets", "Raw", "har_datasets", "4_daily_prcp_roi") #final clipped wgs84
roi = gpd.read_file(r"E:\work\climate_downscaling\Data\Datasets\roi_nepal_rectangle.zip")
roi.to_crs(crs=har_crs, inplace=True)

logging .basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
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
        filtered = [x for x in nc_files if keyword in x]
        logging.info(Fore.BLUE+f"Found {len(filtered)} matching files")
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
    
    for nc in nc_file_list:
        nc_full_path = os.path.join(in_nc_dir, nc)
        outpath_tif = "Unknown File"
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
                logging.info(Fore.BLUE+f"Done for {time_str}")
        except Exception as e:
            logging.error(Fore.RED+f"Failed to save {outpath_tif}: {e}")
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
def clip_reproject_tif(in_tif_dir, out_tif_dir, clip_geom, out_crs):
    os.makedirs(out_tif_dir, exist_ok=True)
    tif_file_list = [tif for tif in os.listdir(in_tif_dir) if tif.endswith(".tif")]
    
    if not tif_file_list:
        logging.warning(Fore.YELLOW+f"No tif files in {in_tif_dir}")
        return
    try:
        ref_tif_path = os.path.join(in_tif_dir, tif_file_list[0])
        ref_tif = rioxarray.open_rasterio(ref_tif_path)
        if clip_geom.crs != ref_tif.rio.crs:
                clip_geom = clip_geom.to_crs(crs=ref_tif.crs)

        ref_tif = ref_tif.rio.clip(clip_geom.geometry.values, clip_geom.crs)
        ref_tif = ref_tif.rio.reproject(out_crs)
        logging.info(Fore.GREEN+f"Ref template created from {tif_file_list[0]} for matching")
    except Exception as e:
        logging.error(Fore.RED+f"Failed to create reference template from {ref_tif_path}: {e}")
        return
    
    for tif in tif_file_list:
        logging.info(Fore.BLUE+f"Processing {tif}...")
        tif_full_path = os.path.join(in_tif_dir, tif)
        out_tif_path = os.path.join(out_tif_dir, tif)

        if os.path.exists(out_tif_path):
            continue

        try:
            ds = rioxarray.open_rasterio(tif_full_path)
            clipped_tif = ds.rio.clip(clip_geom.geometry.values, clip_geom.crs)
        except Exception as e:
            logging.error(Fore.RED + f"Cound not clip {tif}: {e}")
        try:
            reprojected_tif = clipped_tif.rio.reproject_match(ref_tif)
        except Exception as e:
            logging.error(Fore.RED + f"Cound not reproject {tif}: {e}")
        try:
            print(f"TIF TYPE- {type(reprojected_tif)}")
            reprojected_tif.rio.to_raster(out_tif_path)
        except Exception as e:
            logging.error(Fore.RED + f"Cound not save {tif}: {e}")
        break

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
        # Expected format: prcp_YYYY-MM-DD-HH.tif
        # Extract YYYY-MM-DD
        try:
            # Assuming format ends with YYYY-MM-DD-HH.tif
            # Example: prcp_2020-01-01-06.tif
            date_str = tif.split('_')[-1][:10] # Get '2020-01-01'
            if date_str not in files_by_date:
                files_by_date[date_str] = []
            files_by_date[date_str].append(tif)
        except Exception as e:
            logging.warning(Fore.YELLOW + f"Skipping file with unexpected name format {tif}: {e}")

    for date_str, files in sorted(files_by_date.items()):
        out_path = os.path.join(out_dir, f"prcp_{date_str}.tif")
        
        if os.path.exists(out_path):
            logging.info(f"Skipping {date_str} (already exists)")
            continue
        if len(files) != 24:
            logging.warning(Fore.YELLOW+f"The {date_str} does not have 24 hrs precipitation, skipping aggregation")
            continue

        logging.info(Fore.BLUE + f"Calculating daily total for {date_str} from {len(files)} files...")
        
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

if __name__ == '__main__':
    logging.info('Starting HARv2 pipeline')

    #1. Download Netcdf
    logging.info('--- Step 1: Downloading NetCDF Files ---')
    files_to_download = get_file_links(
        base_url=base_url,
        keyword= "prcp_2020",

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
    logging.info('--- Step 2: Extracting Hourly TIFs ---')
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
    logging.info('--- Step 3: Clipping and Reprojecting ---')
    clip_reproject_tif(
        in_tif_dir=out_dir_hr,
        out_tif_dir=out_dir_hr_clip_reproject,
        clip_geom=roi,
        out_crs="EPSG:4326"
    )

    #4. Calculate Daily Total Precipitation
    logging.info('--- Step 4: Calculating Daily Total Precipitation ---')
    out_dir_daily_roi = os.path.join(base_dir, "Data", "Datasets", "Raw", "har_datasets", "4_daily_prcp_roi")
    calculate_daily_total_precipitation(
        in_dir=out_dir_hr_clip_reproject,
        out_dir=out_dir_daily_roi
    )

    logging.info(Style.RESET_ALL+'''
        ███████ ██ ███    ██ 
        ██      ██ ████   ██ 
        █████   ██ ██ ██  ██ 
        ██      ██ ██  ██ ██ 
        ██      ██ ██   ████ 
    ''')