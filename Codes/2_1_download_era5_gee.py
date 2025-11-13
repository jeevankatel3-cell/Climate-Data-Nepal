##1.1 Importing and Initializing
import ee
import geemap

cloud_project = 'ee-jeevankatel3'

try:
    ee.Initialize(project=cloud_project)
except:
    ee.Authenticate()
    ee.Initialize(project=cloud_project)
from multiprocessing.dummy import Pool as ThreadPool
import threading
import logging
import time
from datetime import datetime
import os
import traceback
from colorama import Fore, Back
from retry import retry
import pandas as pd

start_time = time.time()
roi = ee.FeatureCollection("projects/ee-jeevankatel3/assets/nepRect")

#select output path
output_path = r'Data/Datasets/ERA5_land_prec'

os.makedirs(output_path, exist_ok=True)

print(Fore.BLUE + f'Started at..........{datetime.fromtimestamp(start_time)}')

era5_dataset = ee.ImageCollection('ECMWF/ERA5_LAND/HOURLY').filterDate('1990-01-01', '2020-01-01')

@retry(tries=10, delay=1, backoff=2)
def process_date(date_chunk):
    for myDate in date_chunk:
        try:
            nepal_date = str(myDate.format('YYYYMMdd', timeZone='Asia/Thimphu').getInfo())
            out_fc = os.path.join(output_path, f"ERA5L_prec_{nepal_date}"+'.tif')

            if os.path.exists(out_fc):
                print(Fore.YELLOW + f"..........Skipping {nepal_date}, file already exists..........")
                continue
            
            print(Back.BLUE + f"..........Started for {nepal_date}..........")
            filtered_date_dataset = era5_dataset.filterDate(myDate, myDate.advance(1, 'day'))
            precTot = filtered_date_dataset.select('total_precipitation_hourly').reduce(ee.Reducer.sum()).rename('precTot').multiply(1000)

            # disaggregate into single-band image and download
            prec_tot_image = precTot

            out_fname = f"ERA5L_prec_{nepal_date}.tif"
            out_path_full = os.path.join(output_path, out_fname)
            try:
                region_geojson = roi.geometry().getInfo()
            except Exception:
                region_geojson = roi

            # Try preferred export function, fall back to download_ee_image if not available
            try:
                # geemap.ee_export_image will synchronously export to a local file when possible
                geemap.ee_export_image(prec_tot_image, filename=out_path_full, scale=11132, region=region_geojson, file_per_band=False)
            except Exception as e_export:
                try:
                    # older geemap versions provide download_ee_image; use formatted name
                    geemap.download_ee_image(prec_tot_image, f"ERA5L_prec_{nepal_date}", output_path, "tif", scale=11132, region=region_geojson, crs="EPSG:4326")
                except Exception as e_download:
                    raise RuntimeError(f"Both export methods failed: ee_export_image error: {e_export}; download_ee_image error: {e_download}")
            time.sleep(0.5)
            print(Back.GREEN + f"..........Finished for {nepal_date}..........")
        
        except Exception as e:
            print(Back.RED + f"..........Failed for {nepal_date}..........: {e}")
            # write full traceback to log for debugging
            with open('error_log.txt', 'a') as error_log:
                error_log.write(f"Failed for {nepal_date}: {str(e)}\n")
                error_log.write(traceback.format_exc() + "\n")

if __name__ == "__main__":
    logging.basicConfig()
    results = []

    leap_years = [
        1904, 1908, 1912, 1916, 1920, 1924, 1928, 1932, 1936, 1940, 
        1944, 1948, 1952, 1956, 1960, 1964, 1968, 1972, 1976, 1980, 
        1984, 1988, 1992, 1996, 2000, 2004, 2008, 2012, 2016, 2020, 2024]

    for year in range(1990, 1991):
        dateList = []
        for month in range(1, 13):
            if month in [4, 6, 9, 11]:
                max_day = 30
            elif month in [1, 3, 5, 7, 8, 10, 12]:
                max_day = 31
            elif month == 2 and year in leap_years:
                max_day = 29
            else:
                max_day = 28
            
            for day in range(1, max_day + 1):
                current_date = ee.Date.fromYMD(year, month, day).advance(-6, 'hour')
                dateList.append(current_date)

        # after building dateList for the year, split into chunks and process
        workers = 15
        chunk_size = max(1, len(dateList) // workers)
        date_chunks = [dateList[i:i + chunk_size] for i in range(0, len(dateList), chunk_size)]

        pool = ThreadPool(workers)
        try:
            pool.map(process_date, date_chunks)
        finally:
            pool.close()
            pool.join()

        end_time = time.time()
        print(Back.GREEN + f'Completed.......{datetime.fromtimestamp(end_time)}|||||Completed in {end_time-start_time} sec')

