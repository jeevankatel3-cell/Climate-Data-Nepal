# download daily chelsa precipitaiton data from repository

import rioxarray as rxr
import os
from tqdm import tqdm
import pandas as pd

template_raster = rxr.open_rasterio(r"../Data/Datasets/DEM_1km/DEM_1km_elevation_max.tif", masked = True).rio.write_crs("EPSG:4326") 

# 2. Construct the URL for the specific CHELSA variable 
start_date = "2002-01-01"
end_date = "2005-12-31"
daily_period = pd.date_range(start=start_date, end=end_date, freq='D')

variables = ["pr"]  

# Base URL for CHELSA V2.1 daily data on the SWITCH cloud storage
base_url = "https://os.unil.cloud.switch.ch/chelsa02/chelsa/global/daily"

download_path = r"../Data_Temp/Datasets/CHELSA/"
os.makedirs(download_path, exist_ok = True)

for var in variables:
    for day_i in tqdm(daily_period, desc=f"Processing {var}"):
        year = day_i.year
        month = day_i.month
        day = day_i.day
        var_name = var
        out_fp = os.path.join(download_path, f"CHELSA_{var_name}_{year}{month:02d}{day:02d}.tif")
        if os.path.exists(out_fp):
            tqdm.write(f"{var_name} for {day_i.strftime('%Y-%m-%d')} already exists. Skipping...")
            continue
        url = f"{base_url}/{var}/{year}/CHELSA_{var}_{day:02d}_{month:02d}_{year}_V.2.1.tif"
        try:
            with rxr.open_rasterio(url) as remote_rds:

                matched_raster = remote_rds.rio.reproject_match(template_raster)
                matched_raster = (matched_raster * 0.1)
                matched_raster.rio.to_raster(out_fp, no_data = -9999)
        except Exception as e:
            tqdm.write(f"An error occurred while streaming or matching the raster: {e}")
