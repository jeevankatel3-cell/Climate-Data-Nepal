#!/usr/bin/env python3
"""
Download MSWEP data from shared drive folder and process it.

Workflow:
1. Initialize Google Drive service.
2. Generate a list of all 3-hourly .nc filenames from 1990-2019.
3. Prepend two 1989 files to the list to handle the time-shift.
4. Loop through the file list in 8-file (daily) chunks.
   - Download the 8 3-hourly files (e.g., '1989365.18.nc' to '1990001.15.nc').
   - For each file:
     - Read data (which is in mm/3-hour accumulation).
     - Clip to ROI.
     - Standardize to a North-up/West-left coordinate system.
   - Sum the 8 accumulation files using a nan-safe sum.
   - Correct for all-NaN pixels (which sum to 0) by setting them to NoData.
   - Save the final daily sum as a single TIF
     (e.g., 'MSWEP_prec_19900101.tif').
5. Log all errors to a single 'error_log.txt' file.
"""

import os
import io
import sys
import time  # <-- Added for retry sleep
from datetime import date, timedelta, datetime

import geopandas as gpd
import netCDF4
import numpy as np
import rasterio
from google.oauth2.service_account import Credentials as ServiceAccountCredentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from rasterio.io import MemoryFile
from rasterio.mask import mask
from rasterio.transform import from_origin

# --- Configuration ---
CREDENTIAL_FILE = r"C:\Users\A S U S\Documents\Keys\credentials.json"
FOLDER_URL = "https://drive.google.com/drive/folders/1DVR90Ud1C444bTOPeENX-3I7tgqPLnao"
FOLDER_ID = FOLDER_URL.split('/folders/')[-1].split('?')[0]
OUTPUT_TEMP_DIR = r"C:\Users\A S U S\Documents\Study\ClimateData\CD-Workspace\Data\Datasets\MSWEP_TEMP"
OUTPUT_FIN_DIR = r"C:\Users\A S U S\Documents\Study\ClimateData\CD-Workspace\Data\Datasets\MSWEP_TIF"
ROI_PATH = r"C:\Users\A S U S\Documents\Study\ClimateData\CD-Workspace\Data\Datasets\roi_nepal_rectangle\nepal_rectangle_shape.shp"
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']

# --- Variables ---
START_YEAR = 1990
END_YEAR = 2020  # Exclusive (processes up to 2019-12-31)
NC_VARIABLE_NAME = 'precipitation'
FINAL_NODATA_VALUE = -9999.0  # NoData value for final daily TIFs

# --- Helper Functions ---

def log_error_to_file(log_file, message):
    """Appends a timestamped error message to the log file."""
    try:
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        with open(log_file, 'a') as f:
            f.write(f"[{timestamp}] {message}\n")
    except Exception as e:
        print(f"CRITICAL: Failed to write to log file {log_file}: {e}")
        print(f"Original log message was: {message}")


def initialize_drive_service(credential_file, scopes, log_file_path):
    """Initializes and returns the Google Drive API service."""
    try:
        creds = ServiceAccountCredentials.from_service_account_file(credential_file, scopes=scopes)
        print("✅ Loaded service account credentials.")
        drive_service = build('drive', 'v3', credentials=creds)
        print("✅ Google Drive API client initialized successfully.")
        return drive_service
    except Exception as e:
        error_msg = f"❌ Error loading credentials: {e}"
        print(error_msg)
        log_error_to_file(log_file_path, f"initialize_drive_service: {error_msg}")
        return None


def check_folder_exists(drive_service, folder_id, log_file_path):
    """Checks if the specified folder exists in Google Drive."""
    try:
        folder = drive_service.files().get(fileId=folder_id, fields='id, name').execute()
        print(f"✅ Folder found: {folder['name']} (ID: {folder['id']})")
        return True
    except Exception as e:
        error_msg = f"❌ Error accessing folder ID {folder_id}: {e}"
        print(error_msg)
        log_error_to_file(log_file_path, f"check_folder_exists: {error_msg}")
        return False


def generate_file_list(start_year, end_year):
    """Generates the list of filenames to process, including time-shift buffer."""
    print(f"Generating core filename list from {start_year} to {end_year - 1}...")
    core_filename_list = []
    for year in range(start_year, end_year):
        start_date = date(year, 1, 1)
        end_date = date(year, 12, 31)

        for day_offset in range((end_date - start_date).days + 1):
            date_itr = start_date + timedelta(days=day_offset)

            for hour in range(0, 24, 3):
                year_str = str(date_itr.year)
                doy = str(date_itr.timetuple().tm_yday).zfill(3)
                hour_str = str(hour).zfill(2)
                filename = f'{year_str}{doy}.{hour_str}.nc'
                core_filename_list.append(filename)

    print(f"Generated {len(core_filename_list)} core filenames.")

    # Prepend the two files from 1989 needed for the 6-hour time shift.
    final_list = ['1989365.18.nc', '1989365.21.nc'] + core_filename_list
    print(f"Total files in list (including 2 buffer files): {len(final_list)}")
    return final_list


def load_clipped_nc_array(nc_path, vector_gdf, nc_variable_name, log_file_path):
    """
    Reads a single NetCDF file, clips it, and returns the data as a 
    numpy array and profile.
    
    - Assumes data is 'mm/3-hour' and performs NO unit conversion.
    - Handles (flips) ascending latitude arrays.
    - Sets NoData values to np.nan for safe summation.
    
    Returns:
        (data_array, profile) or (None, None) on failure.
    """
    try:
        # --- 1. Read NetCDF Data ---
        with netCDF4.Dataset(nc_path, 'r') as nc_data:
            lat_var = 'latitude' if 'latitude' in nc_data.variables else 'lat'
            lon_var = 'longitude' if 'longitude' in nc_data.variables else 'lon'

            if nc_variable_name not in nc_data.variables:
                raise ValueError(f"Variable '{nc_variable_name}' not in {nc_path}")

            var_obj = nc_data.variables[nc_variable_name]
            data = var_obj[:]
            lat = nc_data.variables[lat_var][:]
            lon = nc_data.variables[lon_var][:]
            
            if data.ndim == 3:
                data = data[0, :, :]
            
            # --- FIX: Check for both _FillValue and missing_value ---
            if '_FillValue' in var_obj.ncattrs():
                nodata_val = var_obj.getncattr('_FillValue')
            elif 'missing_value' in var_obj.ncattrs():
                nodata_val = var_obj.getncattr('missing_value')
            else:
                nodata_val = None

            # --- 2. Georeference & Handle Latitude Order ---
            
            # --- FIX: Guard against 1-dimensional arrays ---
            if lat.size < 2 or lon.size < 2:
                raise ValueError(f"Latitude/Longitude arrays in {nc_path} must have at least 2 points to determine resolution.")

            res_lat = abs(lat[1] - lat[0])
            res_lon = abs(lon[1] - lon[0])
            
            # --- FIX: Calculate the true upper-left *corner* from the cell *centers* ---
            west = float(lon.min())/2 #- res_lon / 2.0)
            north = float(lat.max())/2 #+ res_lat / 2.0)
            
            # --- FIX: Pass positive res_lat. from_origin handles the negative y-scale. ---
            transform = from_origin(west, north, res_lon, res_lat)
            
            data_to_write = data.copy()
            # Check if latitude array is ascending (South-to-North)
            if lat[0] < lat[-1]:
                print("     (Note: Flipping latitude axis for N-up transform)")
                # Flip the data array vertically to match the N-up transform
                data_to_write = np.flipud(data_to_write)

            full_profile = {
                'driver': 'GTiff',
                'height': data.shape[0],
                'width': data.shape[1],
                'count': 1,
                'dtype': data.dtype,
                'crs': 'EPSG:4326',  # WGS84
                'transform': transform,
                'nodata': nodata_val  # --- FIX: Use the original nodata_val ---
            }
            
    except Exception as e:
        error_msg = f"Error during NetCDF read for {nc_path}: {e}"
        print(f"     ❌ {error_msg}")
        log_error_to_file(log_file_path, f"load_clipped_nc_array (NC Read): {error_msg}")
        return None, None

    # --- 3. Clip the Raster In-Memory ---
    try:
        with MemoryFile() as memfile:
            with memfile.open(**full_profile) as src:
                # Write 2D data (flipped or not) to band 1
                src.write(data_to_write.astype(full_profile['dtype']), 1)
                
                # --- 3a. Match CRS ---
                # --- FIX: Use .equals() for robust CRS comparison ---
                if not vector_gdf.crs.equals(src.crs):
                    vector_gdf_proj = vector_gdf.to_crs(src.crs)
                    geometries = vector_gdf_proj.geometry.values
                else:
                    geometries = vector_gdf.geometry.values

                # --- 3b. Perform Clip (Mask) ---
                out_image, out_transform = mask(
                    src,
                    geometries,
                    crop=True,
                    filled=True  # Fills outside areas with nodata_val
                )
                
                out_meta = src.meta.copy()
                out_meta.update({
                    "driver": "GTiff",
                    "height": out_image.shape[1],
                    "width": out_image.shape[2],
                    "transform": out_transform,
                    "dtype": 'float32' # Ensure float for sums
                })

        # --- 4. Handle NoData ---
        
        # Change dtype to float32 for calculations
        out_image_processed = out_image.astype('float32')

        # Set NoData to np.nan for safe math
        if nodata_val is not None:
            # Note: Using np.isclose() is safer for floats, but == is ok
            # if nodata_val is an exact integer or float sentinel.
            out_image_processed[out_image_processed == nodata_val] = np.nan
        
        # *** NO UNIT CONVERSION (REMOVED * 3.0) ***
        # Data is already mm/3-hour accumulation
        
        # --- FIX: REMOVED `out_meta['nodata'] = np.nan` ---
        # The profile 'out_meta' already correctly inherits 
        # 'nodata': nodata_val (e.g., -9999.0) from 'full_profile'

        return out_image_processed, out_meta

    except Exception as e:
        error_msg = f"Error during raster clipping for {nc_path}: {e}"
        print(f"     ❌ {error_msg}")
        log_error_to_file(log_file_path, f"load_clipped_nc_array (Clip): {error_msg}")
        return None, None


# --- Main Download and Process Function ---

def download_and_process_files(drive_service, folder_id, target_filenames, nc_variable_name, temp_dir, final_dir,
                               log_file_path, roi_shapefile_path):
    """
    Main workflow to download, apply time-shift, aggregate to daily, and convert files.
    """
    print("\n--- Starting Download and Processing Workflow ---")

    # --- 1. Load ROI Shapefile ONCE ---
    try:
        print(f"Loading ROI: {roi_shapefile_path}")
        vector_gdf = gpd.read_file(roi_shapefile_path)
    except Exception as e:
        error_msg = f"CRITICAL: Failed to read ROI shapefile: {e}. Exiting."
        print(f"     ❌ {error_msg}")
        log_error_to_file(log_file_path, f"WorkflowError: {error_msg}")
        return

    # Loop in 8-file (daily) chunks, skipping the last 2 buffer files
    total_to_process = len(target_filenames) - 2
    num_days = total_to_process // 8
    
    print(f"Applying 6-hour (2-file) time-shift. Processing {num_days} days...")

    for i in range(0, total_to_process, 8):
        
        # --- 2. Identify Day and Define Filename ---
        date_str_doy = target_filenames[i + 2][0:7]  # e.g., "1990001"
        
        try:
            date_obj = datetime.strptime(date_str_doy, '%Y%j')
            date_str_save = date_obj.strftime('%Y%m%d') # e.g., "19900101"
        except ValueError:
            error_msg = f"Invalid date format: {date_str_doy}. Skipping day."
            print(f"     ❌ {error_msg}")
            log_error_to_file(log_file_path, f"WorkflowError: {error_msg}")
            continue

        day_num = (i // 8) + 1
        print(f"\n[{day_num}/{num_days}] Processing Day: {date_obj.strftime('%Y-%m-%d')}")
        
        final_tif_path = os.path.join(final_dir, f"MSWEP_prec_{date_str_save}.tif")

        # --- 3. Skip Check ---
        if os.path.exists(final_tif_path):
            print(f"     SKIPPING: Final TIF already exists at {final_tif_path}")
            continue

        # --- 4. Process 8 3-Hourly Files for the Day ---
        daily_sum_array = None
        day_profile = None
        all_nan_mask = None # For tracking all-NaN pixels
        day_processed_successfully = True

        for j in range(8):
            filename_to_download = target_filenames[i + j]
            temp_nc_path = os.path.join(temp_dir, filename_to_download)
            print(f"     [{j+1}/8] Processing {filename_to_download}...")
            
            # ================= RETRY LOGIC START =================
            max_retries = 5
            file_success = False
            data_array = None
            profile = None

            for attempt in range(max_retries + 1):
                try:
                    # --- 4a. Download ---
                    query = f"name='{filename_to_download}' and '{folder_id}' in parents and trashed=false"
                    response = drive_service.files().list(
                        q=query, spaces='drive', fields='files(id, name)', pageSize=1
                    ).execute()
                    drive_files = response.get('files', [])

                    if not drive_files:
                        raise FileNotFoundError(f"File not found on Drive: {filename_to_download}")
                    
                    file_id = drive_files[0].get('id')
                    request = drive_service.files().get_media(fileId=file_id)
                    
                    with open(temp_nc_path, 'wb') as f:
                        downloader = MediaIoBaseDownload(f, request)
                        done = False
                        while not done:
                            status, done = downloader.next_chunk()

                    # --- 4b. Process (Clip, Standardize) ---
                    (data_array, profile) = load_clipped_nc_array(
                        temp_nc_path, vector_gdf, nc_variable_name, log_file_path
                    )

                    if data_array is None:
                        raise Exception(f"Processing (Load/Clip) returned None for {filename_to_download}")
                    
                    # If we got here, success!
                    file_success = True
                    break  # Exit retry loop

                except Exception as e:
                    # Clean up potentially bad file before retry
                    if os.path.exists(temp_nc_path):
                        try:
                            os.remove(temp_nc_path)
                        except:
                            pass # Ignore delete error during exception handling

                    if attempt < max_retries:
                        # Rebound: 5s, 10s, 15s, 20s, 25s
                        wait_seconds = (attempt + 1) * 5
                        print(f"        ⚠️ Error (Attempt {attempt+1}/{max_retries}): {e}")
                        print(f"           Retrying in {wait_seconds} seconds...")
                        time.sleep(wait_seconds)
                    else:
                        # Final failure
                        error_msg = f"Failed to process file {filename_to_download} after {max_retries} retries. Error: {e}"
                        print(f"     ❌ {error_msg}")
                        log_error_to_file(log_file_path, f"WorkflowError: {error_msg}")
                        file_success = False
            
            # --- Cleanup Temp File (Always) ---
            if os.path.exists(temp_nc_path):
                try:
                    os.remove(temp_nc_path)
                except Exception as e:
                    log_error_to_file(log_file_path, f"CleanupWarning: Failed to remove {temp_nc_path}: {e}")

            # Check if file succeeded; if not, break the day loop
            if not file_success:
                day_processed_successfully = False
                break
            # ================= RETRY LOGIC END =================

            # --- 4c. Accumulate Sum and All-NaN Mask ---
            try:
                if daily_sum_array is None:
                    daily_sum_array = data_array
                    day_profile = profile
                    all_nan_mask = np.isnan(data_array) # Initialize mask
                else:
                    if daily_sum_array.shape != data_array.shape:
                        raise Exception("Shape mismatch during daily sum.")
                    
                    # Sum the data
                    daily_sum_array = np.nansum(
                        [daily_sum_array, data_array], axis=0
                    )
                    # Update the mask (pixel is all-NaN only if it was all-NaN before *AND* is NaN now)
                    all_nan_mask = all_nan_mask & np.isnan(data_array)

            except Exception as e:
                error_msg = f"Failed to accumulate sum for file {filename_to_download}: {e}"
                print(f"     ❌ {error_msg}")
                log_error_to_file(log_file_path, f"WorkflowError: {error_msg}")
                day_processed_successfully = False
                break  # Stop processing this day

        # --- 5. Save Final Daily File ---
        if day_processed_successfully:
            try:
                print(f"     Saving daily sum to {final_tif_path}...")
                
                # *** FIX for np.nansum behavior ***
                # Where all 8 inputs were NaN, nansum returns 0.
                # Use the mask to set these pixels back to NaN.
                daily_sum_array[all_nan_mask] = np.nan
                
                # Convert all remaining NaNs (both original and restored) 
                # back to the standard NoData value
                daily_sum_array[np.isnan(daily_sum_array)] = FINAL_NODATA_VALUE
                
                # Update the profile
                day_profile['nodata'] = FINAL_NODATA_VALUE
                day_profile['dtype'] = 'float32' # Ensure dtype is float for sums

                with rasterio.open(final_tif_path, 'w', **day_profile) as dest:
                    # Write the 3D array (1, H, W) directly
                    dest.write(daily_sum_array.astype('float32'))
                
                print(f"     ✅ Successfully saved daily file.")
            
            except Exception as e:
                error_msg = f"Failed to save final daily TIF {final_tif_path}: {e}"
                print(f"     ❌ {error_msg}")
                log_error_to_file(log_file_path, f"WorkflowError: {error_msg}")
        else:
            print(f"     Skipping save for day {date_str_save} due to processing errors.")

    print("\n--- Workflow Finished ---")


# --- Main Execution ---
if __name__ == '__main__':
    os.makedirs(OUTPUT_TEMP_DIR, exist_ok=True)
    os.makedirs(OUTPUT_FIN_DIR, exist_ok=True)

    LOG_FILE_PATH = os.path.join(OUTPUT_FIN_DIR, 'error_log.txt')

    try:
        log_error_to_file(LOG_FILE_PATH, "--- SCRIPT EXECUTION STARTED ---")

        drive_service = initialize_drive_service(CREDENTIAL_FILE, SCOPES, LOG_FILE_PATH)

        if drive_service:
            if check_folder_exists(drive_service, FOLDER_ID, LOG_FILE_PATH):
                
                filename_list = generate_file_list(START_YEAR, END_YEAR)

                download_and_process_files(
                    drive_service=drive_service,
                    folder_id=FOLDER_ID,
                    target_filenames=filename_list,
                    nc_variable_name=NC_VARIABLE_NAME,
                    temp_dir=OUTPUT_TEMP_DIR,
                    final_dir=OUTPUT_FIN_DIR,
                    log_file_path=LOG_FILE_PATH,
                    roi_shapefile_path=ROI_PATH
                )
            else:
                print("🛑 Exiting: Drive folder not found or not accessible.")
                log_error_to_file(LOG_FILE_PATH, "Exiting: Drive folder not found.")
        else:
            print("🛑 Exiting: Drive service initialization failed. Check credentials.")
            log_error_to_file(LOG_FILE_PATH, "Exiting: Drive service initialization failed.")

    except Exception as e:
        print(f"❌ A critical, unhandled error occurred: {e}")
        log_error_to_file(LOG_FILE_PATH, f"CRITICAL UNHANDLED ERROR: {e}")

    finally:
        print(f"--- Script finished. All errors (if any) logged to {LOG_FILE_PATH} ---")
        log_error_to_file(LOG_FILE_PATH, "--- SCRIPT EXECUTION FINISHED ---")