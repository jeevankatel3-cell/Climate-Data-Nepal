# download mean daily temperature from CHIRTS for Nepal 

import os
import time
import logging
import traceback
from datetime import datetime, timedelta
from multiprocessing.dummy import Pool as ThreadPool
import ee
import geemap
from colorama import Fore, Back, Style
from retry import retry

# --- Earth Engine Initialization ---
cloud_project = 'ee-jeevankatel3'
try:
    ee.Initialize(project=cloud_project)
except Exception:
    ee.Authenticate()
    ee.Initialize(project=cloud_project)

# --- Configuration & Paths ---
start_time = time.time()
roi = ee.FeatureCollection("projects/ee-jeevankatel3/assets/nepRect")
start_year = 2001
end_year = 2005
output_path = r'C:/Users/A S U S/Documents/Study/ClimateData/CD-Workspace/Data_Temp/Datasets/CHIRTS'
os.makedirs(output_path, exist_ok=True)

print(Fore.BLUE + f'Pipeline started at: {datetime.fromtimestamp(start_time)}')

try:
    region_geojson = roi.geometry()
except Exception:
    region_geojson = roi

# --- Core Thread Worker ---
@retry(tries=5, delay=2, backoff=2)
def process_date_chunk(date_strs):
    for date_str in date_strs:
        try:
            date_str_formatted = date_str.replace('-', '')
            out_tmax = os.path.join(output_path, f"CHIRTS_tmax_{date_str_formatted}.tif")
            out_tmin = os.path.join(output_path, f"CHIRTS_tmin_{date_str_formatted}.tif")

            if os.path.exists(out_tmax) and os.path.exists(out_tmin):
                print(Fore.YELLOW + f"Skipping {date_str}: tmax and tmin files already exist.")
                continue
            
            print(Back.BLUE + f"Starting processing for {date_str}...")
            
            # Construct Earth Engine dates safely from the Python string
            ee_date = ee.Date(date_str)
            
            # Filter the collection to a single day slice
            chirts_dataset = ee.ImageCollection('UCSB-CHG/CHIRTS/DAILY')\
                .filterDate(ee_date, ee_date.advance(1, 'day'))
            
            temp_image = chirts_dataset.select(['minimum_temperature', 'maximum_temperature']).first()
            # Download tmax if it doesn't exist
            if not os.path.exists(out_tmax):
                tmax_image = temp_image.select('maximum_temperature').rename('tmax')
                try:
                    geemap.ee_export_image(
                        tmax_image, 
                        filename=out_tmax, 
                        scale=5566, 
                        region=region_geojson, 
                        crs="EPSG:4326",
                        file_per_band=False
                    )
                except Exception as e_export:
                    geemap.download_ee_image(
                        tmax_image, 
                        f"CHIRTS_tmax_{date_str_formatted}", 
                        output_path, 
                        "tif", 
                        scale=5566, 
                        region=region_geojson, 
                        crs="EPSG:4326"
                    )

            # Download tmin if it doesn't exist
            if not os.path.exists(out_tmin):
                tmin_image = temp_image.select('minimum_temperature').rename('tmin')
                try:
                    geemap.ee_export_image(
                        tmin_image, 
                        filename=out_tmin, 
                        scale=5566, 
                        region=region_geojson, 
                        crs="EPSG:4326",
                        file_per_band=False
                    )
                except Exception as e_export:
                    geemap.download_ee_image(
                        tmin_image, 
                        f"CHIRTS_tmin_{date_str_formatted}", 
                        output_path, 
                        "tif", 
                        scale=5566, 
                        region=region_geojson, 
                        crs="EPSG:4326"
                    )
                
            print(Back.GREEN + f"Successfully downloaded tmax and tmin for: {date_str}")
            time.sleep(0.2)

        except Exception as e:
            print(Back.RED + f"Processing failed for {date_str}: {e}")
            with open('error_log.txt', 'a') as error_log:
                error_log.write(f"Failed for {date_str}: {str(e)}\n{traceback.format_exc()}\n")

# --- Main Driver Block ---
if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)

    start_date = datetime(start_year, 1, 1)
    end_date = datetime(end_year, 12, 31)
    
    delta = end_date - start_date
    all_date_strs = [(start_date + timedelta(days=i)).strftime('%Y-%m-%d') for i in range(delta.days + 1)]

    # Chunk the dates across workers
    workers = 15
    chunk_size = max(1, len(all_date_strs) // workers)
    date_chunks = [all_date_strs[i:i + chunk_size] for i in range(0, len(all_date_strs), chunk_size)]

    print(Fore.CYAN + f"Distributing {len(all_date_strs)} tracking dates across {len(date_chunks)} parallel threads...")

    pool = ThreadPool(workers)
    try:
        pool.map(process_date_chunk, date_chunks)
    finally:
        pool.close()
        pool.join()

    end_time = time.time()
    print(Style.RESET_ALL + Back.GREEN + f'Execution complete at: {datetime.fromtimestamp(end_time)} | Duration: {end_time - start_time:.2f} seconds')