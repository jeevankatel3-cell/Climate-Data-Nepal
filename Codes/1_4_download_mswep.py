import os
import io
import time
import logging
import concurrent.futures
from datetime import date, timedelta, datetime
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
import rioxarray
import rasterio
from tqdm import tqdm
from google.oauth2.service_account import Credentials as ServiceAccountCredentials
from googleapiclient.discovery import build, Resource
from googleapiclient.http import MediaIoBaseDownload

# --- Configuration ---
CREDENTIAL_FILE = Path(r"C:\Users\A S U S\Documents\Keys\credentials.json")
FOLDER_URL = "https://drive.google.com/drive/folders/1DVR90Ud1C444bTOPeENX-3I7tgqPLnao"
FOLDER_ID = FOLDER_URL.split('/folders/')[-1].split('?')[0]
OUTPUT_TEMP_PATH = Path(r"C:\Users\A S U S\Documents\Study\ClimateData\CD-Workspace\Data\Datasets\MSWEP_TEMP")
OUTPUT_FINAL_PATH = Path(r"C:\Users\A S U S\Documents\Study\ClimateData\CD-Workspace\Data\Datasets\MSWEP_TIF")
ROI_PATH = Path(r"C:\Users\A S U S\Documents\Study\ClimateData\CD-Workspace\Data\Datasets\roi_nepal_rectangle\nepal_rectangle_shape.shp")
VARIABLE_NAME = "precipitation"
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']
START_YEAR = 2020
END_YEAR = 2021  # Exclusive
NODATA_VALUE = -9999
MAX_RETRIES = 5
BATCH_SIZE = 8
REQUIRED_VALID_FILES = 6  # Threshold for daily aggregation

# --- Setup Logging ---
OUTPUT_FINAL_PATH.mkdir(parents=True, exist_ok=True)
OUTPUT_TEMP_PATH.mkdir(parents=True, exist_ok=True)

log_file_path = OUTPUT_FINAL_PATH / "download_errors.log"
logging.basicConfig(
    filename=log_file_path,
    level=logging.ERROR,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def log_error(message: str):
    """Logs an error message to the file and prints it to console."""
    logging.error(message)
    print(f"❌ {message}")

# --- Helper Functions ---

def initialize_drive_service() -> Optional[Resource]:
    """Initializes and returns the Google Drive API service."""
    try:
        creds = ServiceAccountCredentials.from_service_account_file(str(CREDENTIAL_FILE), scopes=SCOPES)
        return build('drive', 'v3', credentials=creds)
    except Exception as e:
        log_error(f"initialize_drive_service: Error loading credentials: {e}")
        return None

def generate_file_list(start_year: int, end_year: int) -> List[str]:
    """Generates a list of filenames for the specified year range."""
    core_filename_list = []
    
    start_date = datetime(start_year, 1, 1)
    end_date = datetime(end_year, 1, 1)
    
    # We need 3-hourly data
    date_range = pd.date_range(start=start_date, end=end_date - timedelta(hours=3), freq='3h')

    for dt in date_range:
        year_str = str(dt.year)
        doy = str(dt.timetuple().tm_yday).zfill(3)
        hour_str = str(dt.hour).zfill(2)
        filename = f'{year_str}{doy}.{hour_str}.nc'
        core_filename_list.append(filename)

    # Buffer files from 1989 for time shift
    buffer_files = ['1989365.18.nc', '1989365.21.nc']
    final_list = buffer_files + core_filename_list
    
    # Remove the last two files to maintain the count
    final_list = final_list[:-2]
    
    print(f"Total files in list: {len(final_list)}")
    return final_list

def get_file_id(drive_service: Resource, filename: str) -> Optional[str]:
    """Finds the file ID for a given filename with retry logic."""
    for attempt in range(MAX_RETRIES):
        try:
            query = f"name='{filename}' and '{FOLDER_ID}' in parents and trashed=false"
            response = drive_service.files().list(q=query, spaces='drive', fields='files(id, name)', pageSize=1000).execute()
            files = response.get('files', [])
            if not files:
                return None
            return files[0]['id']
        except Exception as e:
            log_error(f"get_file_id: Attempt {attempt + 1} failed for {filename}: {e}")
            time.sleep(2 ** attempt)
    return None

def download_file_with_retry(drive_service: Resource, file_id: str, filename: str, output_path: Path) -> Optional[Path]:
    """Downloads a file with retry logic and immediate validation."""
    dest_path = output_path / filename

    # Check if existing file is valid
    if dest_path.exists():
        try:
            with xr.open_dataset(dest_path, engine='netcdf4') as ds:
                pass
            return dest_path
        except Exception:
            log_error(f"Existing file {filename} is corrupt. Re-downloading.")
            dest_path.unlink()

    for attempt in range(MAX_RETRIES):
        try:
            request = drive_service.files().get_media(fileId=file_id)
            with io.FileIO(str(dest_path), 'wb') as fh:
                # Increased chunk size to 10MB for speed
                downloader = MediaIoBaseDownload(fh, request, chunksize=10 * 1024 * 1024)
                done = False
                while not done:
                    status, done = downloader.next_chunk()
            
            # Validate immediately
            with xr.open_dataset(dest_path, engine='netcdf4') as ds:
                pass
            return dest_path

        except Exception as e:
            log_error(f"Attempt {attempt + 1} failed for {filename}: {e}")
            if dest_path.exists():
                try:
                    # Log first 100 bytes to debug HTML errors
                    with open(dest_path, 'rb') as f:
                        header = f.read(100)
                    log_error(f"File header: {header}")
                except Exception:
                    pass
                dest_path.unlink()
            
            if attempt < MAX_RETRIES - 1:
                time.sleep(2 ** attempt)
                
    return None

def net_cdf_to_tif(netcdf_path: Path, output_dir: Path, roi_gdf: gpd.GeoDataFrame) -> Optional[Path]:
    """Converts a NetCDF file to a TIF, clipped to ROI."""
    try:
        with xr.open_dataset(netcdf_path, engine='netcdf4') as ds:
            if 'lon' in ds.coords and 'lat' in ds.coords:
                ds = ds.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=True)
            ds.rio.write_crs("EPSG:4326", inplace=True)

            if roi_gdf.crs != "EPSG:4326":
                roi_4326 = roi_gdf.to_crs("EPSG:4326")
            else:
                roi_4326 = roi_gdf

            ds_clipped = ds.rio.clip(roi_4326.geometry, roi_4326.crs, drop=True)
            
            basename = netcdf_path.stem + '.tif'
            output_path = output_dir / basename
            
            ds_clipped[VARIABLE_NAME].isel(time=0).rio.to_raster(output_path)
            
        return output_path
    except Exception as e:
        log_error(f"net_cdf_to_tif: Error processing {netcdf_path}: {e}")
        return None

def process_daily_batch(drive_service: Resource, batch_files: List[str], roi_gdf: gpd.GeoDataFrame) -> bool:
    """Downloads files sequentially, converts, sums, and saves."""
    last_filename = batch_files[-1]
    try:
        y_str = last_filename[:4]
        doy_str = last_filename[4:7]
        date_obj = datetime.strptime(f"{y_str}{doy_str}", "%Y%j").date()
        date_str = date_obj.strftime("%Y%m%d")
    except ValueError:
        date_str = "unknown_date"

    output_filename = f"MSWEP_Prec_{date_str}.tif"
    output_full_path = OUTPUT_FINAL_PATH / output_filename

    if output_full_path.exists():
        return True

    temp_tifs = []
    temp_ncs = []
    
    try:
        # 1. Sequential Download and Convert
        for fname in batch_files:
            # Get ID
            fid = get_file_id(drive_service, fname)
            if not fid:
                log_error(f"File not found in Drive: {fname}")
                continue

            # Download
            nc_path = download_file_with_retry(drive_service, fid, fname, OUTPUT_TEMP_PATH)
            if not nc_path:
                continue
            temp_ncs.append(nc_path)

            # Convert
            tif_path = net_cdf_to_tif(nc_path, OUTPUT_TEMP_PATH, roi_gdf)
            if tif_path:
                temp_tifs.append(tif_path)
            
            # Small delay to respect rate limits
            #time.sleep(0.5)

        # 2. Check if we have enough data
        if not temp_tifs:
            log_error(f"No valid files downloaded for {date_str}")
            return False

        # 3. Merge (Sum)
        src_files_to_close = []
        
        try:
            daily_total = None
            for tif in temp_tifs:
                src = rioxarray.open_rasterio(tif)
                src_files_to_close.append(src)

            if daily_total is None:
                daily_total = src
            else:
                daily_total = daily_total + src

            # 4. Save Final TIF
            if daily_total is not None:
                daily_total.rio.to_raster(output_full_path)
                logging.info(f"Saved {output_full_path}")
        except Exception as e:
            log_error(f"Failed to calculate daily total for {date_str}: {e}")
        

        for src in src_files_to_close:
            src.close()

        return True

    except Exception as e:
        log_error(f"process_daily_batch: Failed batch ending {batch_files[-1]}: {e}")
        return False
    finally:
        # Cleanup
        for f in temp_ncs:
            if f.exists():
                try: f.unlink()
                except OSError: pass
        for f in temp_tifs:
            if f.exists():
                try: f.unlink()
                except OSError: pass

# --- Main Execution ---

def main():
    print("🚀 Starting MSWEP Download and Processing Script...")
    
    drive_service = initialize_drive_service()
    if not drive_service:
        print("❌ Failed to initialize Drive service. Exiting.")
        return

    try:
        roi = gpd.read_file(ROI_PATH)
    except Exception as e:
        log_error(f"Failed to load ROI shapefile: {e}")
        return

    file_list = generate_file_list(START_YEAR, END_YEAR)
    
    batches = [file_list[i:i + BATCH_SIZE] for i in range(0, len(file_list), BATCH_SIZE)]
    
    print(f"📅 Processing {len(batches)} daily batches...")

    with tqdm(total=len(batches), desc="Processing Days", unit="day") as pbar:
        for batch in batches:
            if len(batch) < BATCH_SIZE:
                log_error(f"Skipping incomplete batch: {batch}")
                pbar.update(1)
                continue
                
            success = process_daily_batch(drive_service, batch, roi)
            if success:
                pbar.set_postfix_str(f"Last: {batch[-1]}")
            else:
                pbar.set_postfix_str("Error in batch")
            
            pbar.update(1)

    print("\n✅ Processing complete. Check logs for any errors.")

if __name__ == "__main__":
    main()