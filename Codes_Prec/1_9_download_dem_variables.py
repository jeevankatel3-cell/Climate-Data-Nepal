#download following dem variables at 1km*1km: elevation_mean, slope_mean, aspect_mean, elevation_std, elevation_min, elevation_max
import ee
import geemap
import os
import time
from colorama import Fore, Back, init
from retry import retry
import logging
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

init(autoreset=True)
cloud_project = 'ee-jeevankatel3'
try:
    ee.Initialize(project=cloud_project)
except:
    ee.Authenticate()
    ee.Initialize(project=cloud_project)

roi = ee.FeatureCollection("projects/ee-jeevankatel3/assets/nepRect")
dem = ee.Image("CGIAR/SRTM90_V4")
projection = dem.projection()

projection_1k = projection.atScale(1000)

#process variables
elevation_mean = dem.reduceResolution(
    reducer = ee.Reducer.mean().unweighted(),
    maxPixels = 65536
).reproject(
    crs = projection_1k
).rename('elevation_mean')

elevation_std = dem.reduceResolution(
    reducer = ee.Reducer.stdDev().unweighted(),
    maxPixels = 65536
).reproject(
    crs = projection_1k
).rename('elevation_std')

elevation_min = dem.reduceResolution(
    reducer = ee.Reducer.min().unweighted(),
    maxPixels = 65536
).reproject(
    crs = projection_1k
).rename('elevation_min')

elevation_max = dem.reduceResolution(
    reducer = ee.Reducer.max().unweighted(),
    maxPixels = 65536
).reproject(
    crs = projection_1k
).rename('elevation_max')

slope_mean = ee.Terrain.slope(dem).reduceResolution(
    reducer = ee.Reducer.mean().unweighted(),
    maxPixels = 65536
).reproject(
    crs = projection_1k
).rename('slope_mean')

aspect_mean = ee.Terrain.aspect(dem).reduceResolution(
    reducer = ee.Reducer.mean().unweighted(),
    maxPixels = 65536
).reproject(
    crs = projection_1k
).rename('aspect_mean')

dem_ic = ee.ImageCollection.fromImages([elevation_mean, elevation_std, elevation_min, elevation_max, slope_mean, aspect_mean])
img_list = dem_ic.toList(dem_ic.size())
img_count = dem_ic.size()

#export each image
out_dir = r"C:\Users\A S U S\Documents\Study\ClimateData\CD-Workspace\Data\Datasets\DEM_1km"

os.makedirs(out_dir, exist_ok=True)

img_count_py = img_count.getInfo()
roi_geojson = roi.geometry().getInfo()  # Convert to GeoJSON

for i in range(img_count_py):
    img = ee.Image(img_list.get(i)).clip(roi)
    band_name = img.bandNames().get(0).getInfo()
    filename = "DEM_1km_" + band_name + ".tif"
    outpath_full = os.path.join(out_dir, filename)
    projection = img.projection()
    scale = projection.nominalScale().getInfo()
    
    # Skip if file already exists
    if os.path.exists(outpath_full):
        print(Fore.YELLOW + f"Skipping {band_name}, file already exists")
        continue
    
    try:
        geemap.ee_export_image(img, filename=outpath_full, region=roi_geojson, crs="EPSG:4326", scale=scale, file_per_band=False)
        print(Fore.GREEN + f"Completed Download for {outpath_full}")
    except Exception as e:
        print(Fore.RED + f"Failed to download: {band_name}")
        logging.error(f"Error downloading image {band_name}: {str(e)}")

# Download original 90m DEM using geemap (handles large files by tiling)
nepal_bigger_roi = ee.FeatureCollection("projects/ee-jeevankatel3/assets/nepRectBigger")
nepal_bigger_roi_geom = nepal_bigger_roi.geometry()

dem_90m_outpath = os.path.join(out_dir, "DEM_90m_elevation_geemap.tif")

if os.path.exists(dem_90m_outpath):
    print(Fore.YELLOW + f"Skipping DEM_90m, file already exists")
else:
    try:
        geemap.download_ee_image(
            image=dem,
            filename=dem_90m_outpath,
            region=nepal_bigger_roi_geom,
            crs="EPSG:4326",
            scale=90
        )
        print(Fore.GREEN + f"Completed Download for {dem_90m_outpath}")
    except Exception as e:
        print(Fore.RED + f"Failed to download 90m DEM: {str(e)}")
        logging.error(f"Error downloading 90m DEM: {str(e)}")

# save elevation_mean at bigger grid size
dem_1km_outpath = os.path.join(out_dir, "DEM_1km_elevation_mean_bigger_grid.tif")
projection = elevation_mean.projection()
scale = projection.nominalScale().getInfo()
geemap.ee_export_image(elevation_mean, filename=dem_1km_outpath, region=nepal_bigger_roi_geom, crs="EPSG:4326", scale=scale, file_per_band=False)