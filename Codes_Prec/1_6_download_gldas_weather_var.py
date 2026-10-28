# code to download weather variables from GLDAS 2.1
import ee
import geemap
import os
import time
import logging
import traceback
import calendar
from datetime import datetime
from multiprocessing.dummy import Pool as ThreadPool
from functools import partial
from colorama import Fore, Back, init
from retry import retry 

# Initialize colorama
init(autoreset=True)

cloud_project = 'ee-jeevankatel3'

try:
    ee.Initialize(project=cloud_project)
except Exception:
    ee.Authenticate(force=True)
    ee.Initialize(project=cloud_project)

start_time = time.time()
roi = ee.FeatureCollection("projects/ee-jeevankatel3/assets/nepRect")
region_geom = roi.geometry()


# select output path
outpath_air_temp = r"E:\ClimateData_Precipitation\Data\Datasets\GLDAS\air_temp"
outpath_specific_humidity = r"E:\ClimateData_Precipitation\Data\Datasets\GLDAS\specific_humidity"
outpath_wind_speed = r"E:\ClimateData_Precipitation\Data\Datasets\GLDAS\wind_speed"
outpath_surface_pressure = r"E:\ClimateData_Precipitation\Data\Datasets\GLDAS\surface_pressure"
outpath_swdown = r"E:\ClimateData_Precipitation\Data\Datasets\GLDAS\swdown"
outpath_lwdown = r"E:\ClimateData_Precipitation\Data\Datasets\GLDAS\lwdown"

os.makedirs(outpath_air_temp, exist_ok=True)
os.makedirs(outpath_specific_humidity, exist_ok=True)
os.makedirs(outpath_wind_speed, exist_ok=True)
os.makedirs(outpath_surface_pressure, exist_ok=True)
os.makedirs(outpath_swdown, exist_ok=True)
os.makedirs(outpath_lwdown, exist_ok=True)

print(Fore.BLUE + f'Started at..........{datetime.fromtimestamp(start_time)}')

gldas_dataset = ee.ImageCollection("NASA/GLDAS/V021/NOAH/G025/T3H").filterDate('2001-01-01', '2026-01-01')

def choose_secondary_variable():
    variable = input("Enter the secondary variable (air temperature, specific humidity, wind speed, surface pressure, sw down, lw down, soil moisture, all, skip): ").strip()
    if variable == 'air temperature':
        return ['air_temperature']
    elif variable == 'specific humidity':
        return ['specific_humidity']
    elif variable == 'wind speed':
        return ['wind_speed']
    elif variable == 'surface pressure':
        return ['surface_pressure']
    elif variable == 'sw down':
        return ['sw_down']
    elif variable == 'lw down':
        return ['lw_down']
    elif variable == 'all':
        return ['air_temperature', 'specific_humidity', 'wind_speed', 'surface_pressure', 'sw_down', 'lw_down']
    elif variable == 'skip':
        return None    
    else:
        raise ValueError(f"Unknown variable: {variable}")

@retry(tries=10, delay=1, backoff=2)
def process_date(date_chunk, variables):
    if not variables:
        return

    for myDate, nepal_date in date_chunk:
        try:
            filtered_date_dataset = gldas_dataset.filterDate(myDate, myDate.advance(1, 'day'))
            
            for variable in variables:
                # FIX: Checked against underscore strings to match what choose_secondary_variable() returns
                if variable == 'air_temperature':
                    out_dir = outpath_air_temp
                    file_name = f"GLDAS_air_temp_{nepal_date}.tif"
                elif variable == 'specific_humidity':
                    out_dir = outpath_specific_humidity
                    file_name = f"GLDAS_specific_humidity_{nepal_date}.tif"
                elif variable == 'wind_speed':
                    out_dir = outpath_wind_speed
                    file_name = f"GLDAS_wind_speed_{nepal_date}.tif"
                elif variable == 'surface_pressure':
                    out_dir = outpath_surface_pressure
                    file_name = f"GLDAS_surface_pressure_{nepal_date}.tif"
                elif variable == 'sw_down':
                    out_dir = outpath_swdown
                    file_name = f"GLDAS_sw_down_{nepal_date}.tif"
                elif variable == 'lw_down':
                    out_dir = outpath_lwdown
                    file_name = f"GLDAS_lw_down_{nepal_date}.tif" 
                else:
                    continue

                out_path_full = os.path.join(out_dir, file_name)

                if os.path.exists(out_path_full):
                    print(Fore.YELLOW + f"..........Skipping {file_name}, file already exists..........")
                    continue
                
                print(Back.BLUE + f"..........Started {variable} for {nepal_date}..........")

                image_to_export = None
                if variable == 'air_temperature':
                    image_to_export = filtered_date_dataset.select('Tair_f_inst').mean().rename('air_temperature')
                elif variable == 'specific_humidity':
                    image_to_export = filtered_date_dataset.select('Qair_f_inst').mean().rename('specific_humidity')
                elif variable == 'wind_speed':
                    image_to_export = filtered_date_dataset.select('Wind_f_inst').mean().rename('wind_speed')
                elif variable == 'surface_pressure':
                    image_to_export = filtered_date_dataset.select('Psurf_f_inst').mean().rename('surface_pressure')
                elif variable == 'sw_down':
                    image_to_export = filtered_date_dataset.select('SWdown_f_tavg').mean().rename('sw_down')
                elif variable == 'lw_down':
                    image_to_export = filtered_date_dataset.select('LWdown_f_tavg').mean().rename('lw_down')
                else:
                    print(Fore.RED + f"..........Unknown variable: {variable}.......... - skipping")
                    continue
                try:
                    geemap.ee_export_image(image_to_export, filename=out_path_full, scale=27830, region=region_geom, crs="EPSG:4326", file_per_band=False)
                except Exception as e_export:
                    try:
                        geemap.download_ee_image(image_to_export, filename=out_path_full, scale=27830, region=region_geom, crs="EPSG:4326")
                    except Exception as e_download:
                        raise RuntimeError(f"Both export methods failed for {variable} on {nepal_date}: ee_export_image error: {e_export}; download_ee_image error: {e_download}")
                
                print(Back.GREEN + f"..........Finished downloading {variable} for {nepal_date}..........")
                time.sleep(0.5)
            
        except Exception as e:
            print(Back.RED + f"..........Failed for {nepal_date}..........: {e}")
            with open('error_log.txt', 'a') as error_log:
                error_log.write(f"Failed for {nepal_date}: {str(e)}\n")
                error_log.write(traceback.format_exc() + "\n")

if __name__ == "__main__":
    logging.basicConfig()
    
    try:
        variables = choose_secondary_variable()
        if variables:
            for year in range(2001, 2026):
                dateList = []
                for month in range(1, 13):
                    max_day = calendar.monthrange(year, month)[1]
                    for day in range(1, max_day + 1):
                        current_date = ee.Date.fromYMD(year, month, day).advance(-6, 'hour')
                        date_str = f"{year}{month:02d}{day:02d}"
                        dateList.append((current_date, date_str))

                workers = 2
                chunk_size = max(1, len(dateList) // workers)
                date_chunks = [dateList[i:i + chunk_size] for i in range(0, len(dateList), chunk_size)]

                pool = ThreadPool(workers)
                try:
                    func = partial(process_date, variables=variables)
                    pool.map(func, date_chunks)
                finally:
                    pool.close()
                    pool.join()

            end_time = time.time()
            print(Back.GREEN + f'Completed.......{datetime.fromtimestamp(end_time)}|||||Completed in {end_time-start_time:.2f} sec')
        else:
            print("Skipping processing as requested.")
            
    except Exception as e:
        print(Back.RED + f"An error occurred: {e}")