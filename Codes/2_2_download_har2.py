import os
import requests
import re
import logging
import rasterio
from rasterio.transform import from_origin
import netCDF4
import geopandas
from rasterio.mask import mask as rasterio_mask
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
#Steps
#1. Get the file precipitation file list
#2. Download the data in memory
#3. Covert the netcdf to tif
#4. Mask the tif with the roi
print("🧔🏻‍♀️ I am watching you!")
mask = r"Data\Datasets\roi_nepal_rectangle.zip"
nc_out_path = r"Data\Raw\har_datasets\nc"
masked_tif_out_path = r"Data\Raw\har_datasets\masked_tif"
base_url= "https://data.klima.tu-berlin.de/HAR/v2/d10km/d/2d/"
def get_file_links(base_url, keyword= "prcp"):
    """
    Gets the file name from the har server and parses it to find .nc files
    containing specific keywords like "precip"
    """
    logging.debug(f"🚶‍♀️ Entering get_file_links with url='{base_url}' and keyword='{keyword}'")
    logging.debug(f"🙏 Requesting URL: {base_url}")
    try:
        response = requests.get(base_url)
        response.raise_for_status() # Raise an exception for HTTP errors
        logging.debug(f"HTTP Status Code: {response.status_code}")
        # logging.debug(f"Response Text (first 500 chars): {response.text[:500]}...") # Uncomment for full response debugging
    except requests.exceptions.RequestException as e:
        logging.error(f"⚠️error⚠️ Error fetching URL {base_url}: {e}")
        return [] # Return empty list on error

    all_nc = re.findall(r'href="(.*?\.nc)"', response.text)
    # logging.info(f"🔍 Found {len(all_nc)} .nc files before filtering.")
    all_nc_with_keyword = [nc for nc in all_nc if keyword in nc]
    logging.info(f"🔍 Found {len(all_nc_with_keyword)} .nc files after filtering with keyword '{keyword}'.")
    return all_nc_with_keyword


def download_nc_files(nc_file_list, base_url, nc_out_path):
    logging.debug(f"🚶‍♀️ Entering download_nc_files with {len(nc_file_list)} files.")
    if not nc_file_list:
        logging.info("No NetCDF files to download.")
        return

    os.makedirs(nc_out_path, exist_ok=True)
    logging.info(f"Ensured output directory exists: {nc_out_path}")

    for nc_filename in nc_file_list:
        download_url = f"{base_url}{nc_filename}"
        nc_save_path = os.path.join(nc_out_path, nc_filename)
        
        logging.info(f"Attempting to download: {download_url} to {nc_save_path}")
        try:
            response = requests.get(download_url, stream=True)
            response.raise_for_status() # Raise an exception for HTTP errors

            file_size = int(response.headers.get("content-length", 0))
            with tqdm(total=file_size, unit='B', unit_scale=True, desc= os.path.basename(nc_out_path)) as bar:
                with open(nc_save_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                        bar.update(len(chunk))
                logging.info(f"✅ Successfully downloaded {nc_filename}")
        except requests.exceptions.RequestException as e:
            logging.error(f"⚠️error⚠️ Error downloading {nc_filename} from {download_url}: {e}")
        except IOError as e:
            logging.error(f"⚠️error⚠️ Error writing file {nc_save_path}: {e}")

def process_nc_to_masked_tif(nc_file_path, mask_path, output_tif_path):
    logging.debug(f"🚶‍♀️ Entering process_nc_to_masked_tif for {nc_file_path}")
    try:
        # Open the NetCDF file
        with netCDF4.Dataset(nc_file_path, 'r') as nc_file:
            # Assuming 'prcp' is the precipitation variable
            # You might need to adjust the variable name based on your NetCDF files
            if 'prcp' not in nc_file.variables:
                logging.error(f"⚠️error⚠️ 'prcp' variable not found in {nc_file_path}. Skipping.")
                return

            data = nc_file.variables['prcp'][:]
            lat = nc_file.variables['lat'][:]
            lon = nc_file.variables['lon'][:]

            # Determine the georeferencing for the TIFF
            # Assuming regular grid and data is (time, lat, lon) or (lat, lon)
            # For simplicity, taking the first time slice if it's a 3D array
            if data.ndim == 3:
                data = data[0, :, :] # Take the first time slice
            
            # Calculate resolution
            lon_res = abs(lon[1] - lon[0])
            lat_res = abs(lat[1] - lat[0])

            # Create a transform object
            transform = from_origin(lon.min(), lat.max(), lon_res, lat_res)

            # Open the mask shapefile using geopandas
            gdf = geopandas.read_file(mask_path)
            shapes = [feature for feature in gdf.geometry]

            # Mask the data
            masked_data, masked_transform = rasterio_mask(rasterio.MemoryFile().open(
                driver='GTiff',
                height=data.shape[0],
                width=data.shape[1],
                count=1,
                dtype=data.dtype,
                crs='EPSG:4326', # Assuming WGS84, adjust if necessary
                transform=transform
            ).write(data, 1), shapes, crop=True)

            # Define metadata for the output TIFF
            out_meta = {
                "driver": "GTiff",
                "height": masked_data.shape[1],
                "width": masked_data.shape[2],
                "count": 1,
                "dtype": masked_data.dtype,
                "crs": 'EPSG:4326', # Assuming WGS84, adjust if necessary
                "transform": masked_transform
            }

            # Ensure output directory exists
            os.makedirs(os.path.dirname(output_tif_path), exist_ok=True)

            # Write the masked data to a new TIFF file
            with rasterio.open(output_tif_path, "w", **out_meta) as dest:
                dest.write(masked_data[0], 1)
            
            logging.info(f"✅ Successfully processed and saved masked TIFF: {output_tif_path}")

    except FileNotFoundError:
        logging.error(f"⚠️error⚠️ File not found: {nc_file_path} or {mask_path}")
    except Exception as e:
        logging.error(f"⚠️error⚠️ Error processing {nc_file_path}: {e}")

if __name__ == "__main__":
    logging.info("🚀 Starting HAR data download and processing script.")
    
    # 1. Get file links
    nc_file_list = get_file_links(base_url, keyword="prcp")

    if nc_file_list:
        # 2. Download NetCDF files
        download_nc_files(nc_file_list, base_url, nc_out_path)

        # 3. Process downloaded NetCDF files
        os.makedirs(masked_tif_out_path, exist_ok=True)
        logging.info(f"Ensured output directory exists for masked TIFFs: {masked_tif_out_path}")

        for nc_filename in nc_file_list:
            nc_full_path = os.path.join(nc_out_path, nc_filename)
            if os.path.exists(nc_full_path):
                tif_filename = nc_filename.replace(".nc", ".tif")
                masked_tif_full_path = os.path.join(masked_tif_out_path, tif_filename)
                process_nc_to_masked_tif(nc_full_path, mask, masked_tif_full_path)
            else:
                logging.warning(f"Skipping processing for {nc_filename} as it was not found locally.")
    else:
        logging.info("💀⚠️error⚠️ No NetCDF files found to download or process.")
    
    logging.info("🎉 Script finished.")
