import os
import gc
import warnings
import numpy as np
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor

import rasterio
from rasterio.warp import reproject, Resampling

warnings.filterwarnings("ignore")

# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------
TEMPLATE_RASTER = "../../Temperature/Data/Datasets/DEM_1km/DEM_1km_elevation_max.tif"
START_DATE = "2001-01-01"
END_DATE = "2025-12-31"
VARIABLES = ["pr"]
BASE_URL = "https://os.unil.cloud.switch.ch/chelsa02/chelsa/global/daily"
DOWNLOAD_PATH = "../Data/Datasets/CHELSA/"

N_WORKERS =  3     # parallel download/reproject processes
BATCH_SIZE = 20    # recycle the worker pool after this many files

os.makedirs(DOWNLOAD_PATH, exist_ok=True)

GDAL_ENV = dict(
    GDAL_CACHEMAX=256,                      # MB, hard cap on GDAL block cache
    VSI_CACHE="FALSE",                      # don't cache whole remote reads
    CPL_VSIL_CURL_CACHE_SIZE=16_000_000,    # ~16MB curl cache, bounded
    GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
    CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif",
    GDAL_HTTP_MULTIPLEX="YES",
)


def get_template_profile(path):
    """Read only the georeferencing metadata, never the pixel array."""
    with rasterio.open(path) as tmpl:
        return {
            "crs": tmpl.crs,
            "transform": tmpl.transform,
            "width": tmpl.width,
            "height": tmpl.height,
        }


def build_tasks(template_profile):
    daily_period = pd.date_range(start=START_DATE, end=END_DATE, freq="D")
    tasks = []
    for var in VARIABLES:
        var_name = "tmax" if var == "tasmax" else "tmin"
        for day_i in daily_period:
            year, month, day = day_i.year, day_i.month, day_i.day
            out_fp = os.path.join(
                DOWNLOAD_PATH, f"CHELSA_{var_name}_{year}{month:02d}{day:02d}.tif"
            )
            if os.path.exists(out_fp):
                continue
            tasks.append((var, var_name, year, month, day, out_fp, template_profile))
    return tasks


def process_one(task):
    """
    Download + reproject a single day/variable.
    Runs inside a worker process so memory is reclaimed by the OS when the
    worker is recycled, regardless of any native-code caching behavior.
    """
    var, var_name, year, month, day, out_fp, tp = task
    url = f"{BASE_URL}/{var}/{year}/CHELSA_{var}_{day:02d}_{month:02d}_{year}_V.2.1.tif"

    try:
        with rasterio.Env(**GDAL_ENV):
            with rasterio.open(url) as src:
                src_nodata = src.nodata
                dst_nodata = -9999.0
                dst_array = np.empty((tp["height"], tp["width"]), dtype="float32")

                reproject(
                    source=rasterio.band(src, 1),
                    destination=dst_array,
                    src_transform=src.transform,
                    src_crs=src.crs,
                    src_nodata=src_nodata,
                    dst_transform=tp["transform"],
                    dst_crs=tp["crs"],
                    dst_nodata=dst_nodata,
                    resampling=Resampling.nearest,  # matches reproject_match's default
                    num_threads=1,
                )

            valid = dst_array != dst_nodata
            dst_array[valid] = dst_array[valid] * 0.1 - 273.15

            profile = {
                "driver": "GTiff",
                "height": tp["height"],
                "width": tp["width"],
                "count": 1,
                "dtype": "float32",
                "crs": tp["crs"],
                "transform": tp["transform"],
                "nodata": dst_nodata,
                "compress": "deflate",
                "tiled": True,
            }
            with rasterio.open(out_fp, "w", **profile) as dst:
                dst.write(dst_array, 1)

        del dst_array
        gc.collect()
        return ("ok", out_fp)

    except Exception as e:
        return ("error", f"{var_name} {year}-{month:02d}-{day:02d}: {e}")


def run_in_batches(tasks, n_workers=N_WORKERS, batch_size=BATCH_SIZE):
    """
    Process tasks in small batches, spinning up a fresh process pool for each
    batch. This is what actually stops the OOM: killing the worker processes
    at the end of every batch forces the OS to reclaim all their memory,
    even memory GDAL/rasterio never explicitly freed.
    """
    with tqdm(total=len(tasks)) as pbar:
        for i in range(0, len(tasks), batch_size):
            batch = tasks[i:i + batch_size]
            with ProcessPoolExecutor(max_workers=n_workers) as ex:
                for status, msg in ex.map(process_one, batch):
                    if status == "error":
                        tqdm.write(f"An error occurred: {msg}")
                    pbar.update(1)


def main():
    template_profile = get_template_profile(TEMPLATE_RASTER)
    tasks = build_tasks(template_profile)
    print(f"{len(tasks)} files left to download.")
    run_in_batches(tasks)


if __name__ == "__main__":
    main()