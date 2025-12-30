#code to download required secondary variables from ERA5-Land
#variables include: 1. wind speed (u and v component) 2. dewpoint temperature (a measure of relative humidity)
#3. Surface pressure (atmospheric pressure at the surface)
#4. Soil moisture (volumetric soil water layer 1)
#5. Total evaporation (total evaporation)

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
    ee.Authenticate()
    ee.Initialize(project=cloud_project)

start_time = time.time()
roi = ee.FeatureCollection("projects/ee-jeevankatel3/assets/nepRect")

#select output path
output_path_wind_speed = r'Data/Datasets/ERA5L_2_wind_speed/'
output_path_dewpoint_temperature = r'Data/Datasets/ERA5L_2_dewpoint_temperature/'
output_path_surface_pressure = r'Data/Datasets/ERA5L_2_surface_pressure/'

#add soil moisture and total evaporation
output_path_soil_moisture = r"Data/Datasets/ERA5L_2_soil_moisture/"
output_path_total_evaporation = r"Data/Datasets/ERA5L_2_total_evaporation/"

os.makedirs(output_path_wind_speed, exist_ok=True)
os.makedirs(output_path_dewpoint_temperature, exist_ok=True)
os.makedirs(output_path_surface_pressure, exist_ok=True)
os.makedirs(output_path_soil_moisture, exist_ok=True)
os.makedirs(output_path_total_evaporation, exist_ok=True)

print(Fore.BLUE + f'Started at..........{datetime.fromtimestamp(start_time)}')

era5_dataset = ee.ImageCollection('ECMWF/ERA5_LAND/HOURLY').filterDate('2000-01-01', '2019-12-31')

def choose_secondary_variable():
    #take variable input from user and return the chosen variable
    variable = input("Enter the secondary variable (wind_speed, dewpoint_temperature, surface_pressure, soil_moisture, total_evaporation, all, skip): ").strip()
    if variable == 'wind_speed':
        return ['wind_speed']
    elif variable == 'dewpoint_temperature':
        return ['dewpoint_temperature']
    elif variable == 'surface_pressure':
        return ['surface_pressure']
    elif variable == 'soil_moisture':
        return ['soil_moisture']
    elif variable == 'total_evaporation':
        return ['total_evaporation']
    elif variable == 'all':
        return ['wind_speed', 'dewpoint_temperature', 'surface_pressure', 'soil_moisture', 'total_evaporation']
    elif variable == 'skip':
        return None    
    else:
        raise ValueError(f"Unknown variable: {variable}")

@retry(tries=10, delay=1, backoff=2)
def process_date(date_chunk, variables):
    if not variables:
        return

    for myDate in date_chunk:
        try:
            nepal_date = myDate.format('YYYYMMdd', timeZone='Asia/Thimphu').getInfo()
            
            # Filter dataset for the specific date
            filtered_date_dataset = era5_dataset.filterDate(myDate, myDate.advance(1, 'day'))
            
            for variable in variables:
                # Determine output path and filename
                if variable == 'wind_speed':
                    out_dir = output_path_wind_speed
                    file_name = f"ERA5L_wind_speed_{nepal_date}.tif"
                elif variable == 'dewpoint_temperature':
                    out_dir = output_path_dewpoint_temperature
                    file_name = f"ERA5L_dewpoint_temperature_{nepal_date}.tif"
                elif variable == 'surface_pressure':
                    out_dir = output_path_surface_pressure
                    file_name = f"ERA5L_surface_pressure_{nepal_date}.tif"
                elif variable == 'soil_moisture':
                    out_dir = output_path_soil_moisture
                    file_name = f"ERA5L_soil_moisture_{nepal_date}.tif"
                elif variable == 'total_evaporation':
                    out_dir = output_path_total_evaporation
                    file_name = f"ERA5L_total_evaporation_{nepal_date}.tif"
                else:
                    continue

                out_path_full = os.path.join(out_dir, file_name)

                if os.path.exists(out_path_full):
                    print(Fore.YELLOW + f"..........Skipping {file_name}, file already exists..........")
                    continue
                
                print(Back.BLUE + f"..........Started {variable} for {nepal_date}..........")

                # Prepare image
                image_to_export = None
                if variable == 'wind_speed':
                    u_component = filtered_date_dataset.select('u_component_of_wind_10m').reduce(ee.Reducer.mean())
                    v_component = filtered_date_dataset.select('v_component_of_wind_10m').reduce(ee.Reducer.mean())
                    image_to_export = (u_component.pow(2).add(v_component.pow(2))).sqrt().rename('wind_speed')
                elif variable == 'dewpoint_temperature':
                    image_to_export = filtered_date_dataset.select('dewpoint_temperature_2m').reduce(ee.Reducer.mean()).rename('dewpoint_temperature')
                elif variable == 'surface_pressure':
                    image_to_export = filtered_date_dataset.select('surface_pressure').reduce(ee.Reducer.mean()).rename('surface_pressure')
                elif variable == 'soil_moisture':
                    image_to_export = filtered_date_dataset.select('volumetric_soil_water_layer_1').reduce(ee.Reducer.mean()).rename('soil_moisture')
                elif variable == 'total_evaporation':
                    image_to_export = filtered_date_dataset.select('total_evaporation_hourly').reduce(ee.Reducer.mean()).rename('total_evaporation')
                
                # Get region geometry
                try:
                    region_geojson = roi.geometry().getInfo()
                except Exception:
                    region_geojson = roi

                # Export/Download
                try:
                    geemap.ee_export_image(image_to_export, filename=out_path_full, scale=11132, region=region_geojson, file_per_band=False)
                except Exception as e_export:
                    try:
                        # Fallback to download_ee_image
                        geemap.download_ee_image(image_to_export, filename=out_path_full, scale=11132, region=region_geojson, crs="EPSG:4326")
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
            for year in range(2000, 2020):
                dateList = []
                for month in range(1, 13):
                    max_day = calendar.monthrange(year, month)[1]
                    for day in range(1, max_day + 1):
                        current_date = ee.Date.fromYMD(year, month, day).advance(-6, 'hour')
                        dateList.append(current_date)

                workers = 15
                chunk_size = max(1, len(dateList) // workers)
                date_chunks = [dateList[i:i + chunk_size] for i in range(0, len(dateList), chunk_size)]

                pool = ThreadPool(workers)
                try:
                    # Use partial to pass variables to process_date
                    func = partial(process_date, variables=variables)
                    pool.map(func, date_chunks)
                finally:
                    pool.close()
                    pool.join()

            end_time = time.time()
            print(Back.GREEN + f'Completed.......{datetime.fromtimestamp(end_time)}|||||Completed in {end_time-start_time} sec')
        else:
            print("Skipping processing as requested.")
            
    except Exception as e:
        print(Back.RED + f"An error occurred: {e}")
