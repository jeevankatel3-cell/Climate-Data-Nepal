import requests
import re
from tqdm import tqdm
import os
import netCDF4
import rasterio
from rasterio.transform import from_origin

def get_file_links(url, keyword):
    """
    Fetches the main directory page and parses it to find .nc files
    containing the specified keyword.
    """
    print(f"⏳🤌🏻 Fetching filenames")


base_url = "https://data.klima.tu-berlin.de/HAR/v2/d10km/d/2d/"
out_path = r"Z:\Projects\2025\2025_ClimateDataCorrection\Datasets\harV2Daily\precipitation"

response = requests.get(base_url)
all_nc = re.findall(r'href="(.*?\.nc)"', response.text)

def convert_nc_to_tif(nc_path, tif_path, variable_name):
    """
    Reads a NetCDF file and saves one variable as a GeoTIFF.
    Handles 2D (lat, lon) and 3D (time, lat, lon) data.
    """
    try:
        with netCDF4.Dataset(nc_path, 'r') as nc_data:
            # --- 3. Read file from temp directory ---
            
            # Find the correct variable names for lat/lon (common variations)
            lat_var = 'latitude' if 'latitude' in nc_data.variables else 'lat'
            lon_var = 'longitude' if 'longitude' in nc_data.variables else 'lon'

            if variable_name not in nc_data.variables:
                print(f"    ❌ Variable '{variable_name}' not in {nc_path}")
                print(f"       Available: {list(nc_data.variables.keys())}")
                return False
            
            # Extract data and coordinates
            var_obj = nc_data.variables[variable_name]
            data = var_obj[:]
            lat = nc_data.variables[lat_var][:]
            lon = nc_data.variables[lon_var][:]

            # Handle 3D data (e.g., [time, lat, lon]) by taking the first slice
            if data.ndim == 3:
                print(f"    (Note: 3D data found, using first time-slice)")
                data = data[0, :, :]
            elif data.ndim != 2:
                print(f"    ❌ Error: Variable '{variable_name}' has unsupported {data.ndim} dimensions.")
                return False
            
            # Get NoData value if it exists
            nodata_val = var_obj.getncattr('_FillValue') if '_FillValue' in var_obj.ncattrs() else None

            # Determine georeferencing (assumes regular grid)
            res_lat = abs(lat[1] - lat[0])
            res_lon = abs(lon[1] - lon[0])
            # from_origin expects the *center* of the upper-left pixel
            # For MSWEP, latitude is descending, so lat.max() is correct
            transform = from_origin(lon.min(), lat.max(), res_lon, res_lat)

            # Define GeoTIFF profile
            profile = {
                'driver': 'GTiff',
                'height': data.shape[0],
                'width': data.shape[1],
                'count': 1,
                'dtype': data.dtype,
                'crs': 'EPSG:4326', # WGS84
                'transform': transform,
                'nodata': nodata_val
            }
            
            # --- 4. Save to final directory ---
            with rasterio.open(tif_path, 'w', **profile) as dst:
                dst.write(data.astype(profile['dtype']), 1) # Ensure dtype match
            
            print(f"    ✅ Converted and saved to: {tif_path}")
            return True

    except Exception as e:
        print(f"    ❌ Error during NC-to-TIF conversion: {e}")
        return False

for nc in all_nc:
    if "prcp" in nc:
        download_url = base_url+nc
        print(download_url)
        save_path = os.path.join(out_path, nc)
        with requests.get(download_url, stream=True) as r:
            file_size = int(r.headers.get("content-length", 0))
            
            with tqdm(total=file_size, unit='B', unit_scale=True, desc= os.path.basename(out_path)) as bar:
                with open(save_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size = 8192):
                        f.write(chunk)
                        bar.update(len(chunk))