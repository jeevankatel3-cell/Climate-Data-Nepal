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

start_date = '2022-01-01'
end_date = '2022-12-31'

output_path_evi = r"C:\Users\A S U S\Documents\Study\ClimateData\CD-Workspace\Data\Datasets\MODIS_EVI"
output_path_lst_d = r"C:\Users\A S U S\Documents\Study\ClimateData\CD-Workspace\Data\Datasets\MODIS_lst_d"
output_path_lst_n = r"C:\Users\A S U S\Documents\Study\ClimateData\CD-Workspace\Data\Datasets\MODIS_lst_n"
output_path_lst_diff = r"C:\Users\A S U S\Documents\Study\ClimateData\CD-Workspace\Data\Datasets\MODIS_lst_diff"

os.makedirs(output_path_evi, exist_ok=True)
os.makedirs(output_path_lst_d, exist_ok=True)
os.makedirs(output_path_lst_n, exist_ok=True)
os.makedirs(output_path_lst_diff, exist_ok=True)

roi = ee.FeatureCollection("projects/ee-jeevankatel3/assets/nepRect")

log_file = 'download_errors_modis_secondary_variables.log'
log_file_path = os.path.join(r"..\Data\Logs", log_file)
os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
logging.basicConfig(filename=log_file_path, level=logging.ERROR)


def process_evi():
    try:
        terra_vi = (ee.ImageCollection('MODIS/061/MOD13A2')
                .filterDate(start_date, end_date)
                .filterBounds(roi)
                .select('EVI'))
        evi_ic = terra_vi.map(scale_evi)
        return evi_ic
    except Exception as e:
        logging.error(f"Error processing EVI: {str(e)}")
        return None

def scale_evi(img):
    try:
        img = img.select('EVI').multiply(0.0001).clip(roi)
        minMax_dict = img.reduceRegion(
            reducer=ee.Reducer.minMax(), 
            geometry=roi, 
            scale=1000
        )
        min_val = ee.Number(minMax_dict.get('EVI_min'))
        max_val = ee.Number(minMax_dict.get('EVI_max'))
        min_image = ee.Image.constant(min_val)
        max_image = ee.Image.constant(max_val)
        evi_normalized = img.select('EVI').subtract(min_image) \
            .divide(max_image.subtract(min_image)) \
            .rename('EVI')
        return img.addBands(evi_normalized).select('EVI')
    except Exception as e:
        logging.error(f"Error scaling EVI: {str(e)}")
        return None

def process_lst():
    try:
        ls_ic = (ee.ImageCollection("MODIS/061/MOD11A1")
                .filterDate(start_date, end_date)
                .filterBounds(roi)
                .select('LST_Day_1km', 'LST_Night_1km'))
        ls_ic = ls_ic.map(scale_lst)
        return ls_ic
    except Exception as e:
        logging.error(f"Error processing LST: {str(e)}")
        return None

def scale_lst(img):
    try:
        lst_day = img.select('LST_Day_1km').multiply(0.02).clip(roi).rename('LST_Day_1km')
        lst_night = img.select('LST_Night_1km').multiply(0.02).clip(roi).rename('LST_Night_1km')
        lst_diff = lst_day.subtract(lst_night).rename('LST_Diff_1km')
        return ee.Image.cat([lst_day, lst_night, lst_diff])
    except Exception as e:
        logging.error(f"Error scaling LST: {str(e)}")
        return None

def download_image_collection(image_collection, out_dir, variable):
    try:
        img_list = image_collection.toList(image_collection.size())
        img_count = image_collection.size().getInfo()
        roi_geojson = roi.geometry().getInfo()
        for i in range(img_count):
            image = ee.Image(img_list.get(i))
            filename = str(image.get("system:index").getInfo()).replace("_", "")
            out_path_full = os.path.join(out_dir, f"MODIS_{variable}_{filename}.tif")
            if os.path.exists(out_path_full):
                print(Fore.YELLOW + f"Skipping {filename}, file already exists")
                continue
            try:
                geemap.ee_export_image(image, filename=out_path_full, scale=1000, region=roi_geojson, file_per_band=False)
                print(Fore.GREEN + f"Downloaded: {filename}")
            except Exception as e:
                print(Fore.RED + f"Failed to download: {filename}")
                logging.error(f"Error downloading image {filename}: {str(e)}")
    except Exception as e:
        logging.error(f"Error downloading image collection: {str(e)}")

def choose_secondary_variable():
    try:
        variable = input("Enter the secondary variable (evi, lst_d, lst_n, lst_diff, all, skip): ").strip()
        if variable == 'evi':
            return ['evi']
        elif variable == 'lst_d':
            return ['lst_d']
        elif variable == 'lst_n':
            return ['lst_n']
        elif variable == 'lst_diff':
            return ['lst_diff']
        elif variable == 'all':
            return ['evi', 'lst_d', 'lst_n', 'lst_diff']
        elif variable == 'skip':
            return None
        else:
            raise ValueError(f"Unknown variable: {variable}")
    except Exception as e:
        logging.error(f"Error choosing secondary variable: {str(e)}")
        return None

@retry(tries=10, delay=1, backoff=2)
def process_download(variable):
    try:
        if variable == 'evi':
            out_dir = output_path_evi
            print(Back.BLUE + f"Started downloading {variable}")
            evi_ic = process_evi()
            if evi_ic:
                download_image_collection(evi_ic, out_dir, variable)
        elif variable == 'lst_d':
            out_dir = output_path_lst_d
            print(Back.BLUE + f"Started downloading {variable}")
            ls_ic = process_lst()
            if ls_ic:
                download_image_collection(ls_ic.select('LST_Day_1km'), out_dir, variable)
        elif variable == 'lst_n':
            out_dir = output_path_lst_n
            print(Back.BLUE + f"Started downloading {variable}")
            ls_ic = process_lst()
            if ls_ic:
                download_image_collection(ls_ic.select('LST_Night_1km'), out_dir, variable)
        elif variable == 'lst_diff':
            out_dir = output_path_lst_diff
            print(Back.BLUE + f"Started downloading {variable}")
            ls_ic = process_lst()
            if ls_ic:
                download_image_collection(ls_ic.select('LST_Diff_1km'), out_dir, variable)
        else:
            print(Fore.YELLOW + f"Skipping {variable}, unknown variable")
    except Exception as e:
        print(Fore.RED + f"Failed to download {variable}")
        logging.error(f"Error downloading {variable}: {str(e)}")


if __name__ == "__main__":
    start_time = time.time()
    print(Fore.BLUE + f'Started at: {datetime.fromtimestamp(start_time)}')
    
    try:
        variables = choose_secondary_variable()
        if variables:
            with ThreadPoolExecutor() as executor:
                executor.map(process_download, variables)
    except Exception as e:
        print(Fore.RED + "Failed to download image collection")
        logging.error(f"Error in main: {str(e)}")
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(Fore.BLUE + f'Ended at: {datetime.fromtimestamp(end_time)}')
    print(Fore.BLUE + f'Total time taken: {elapsed_time:.2f} seconds')
