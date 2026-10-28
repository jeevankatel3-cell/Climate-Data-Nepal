"""
MSWEP 3-hourly (UTC) -> daily precipitation totals for the LOCAL (Nepal) calendar day,
clipped to an ROI and written as GeoTIFF.

How the local day is built
--------------------------
MSWEP files are 3-hourly and stamped in UTC. Nepal is UTC+05:45; because the data come in
3-hour steps the closest whole-step shift is 6 h (a ~15 min approximation).
Local day D = [D 00:00, D+1 00:00) local = [D-1 18:00, D 18:00) UTC, so each local day needs
8 files, the first of which belongs to the PREVIOUS UTC day (for 1 Jan that means 31 Dec of
the previous year). Files are therefore computed per local day instead of being sliced
out of one long list.

!! Check STAMP_IS_PERIOD_START against the MSWEP documentation / a file's `time` attribute.
   True  -> the stamp marks the START of the 3-h accumulation (needs 18h and 21h of day D-1)
   False -> the stamp marks the END of the accumulation   (needs 21h of day D-1 and 18h of day D)
"""
import io
import logging
import os
import tempfile
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import geopandas as gpd
import xarray as xr
import rioxarray  # noqa: F401  (registers the .rio accessor)
from google.oauth2.service_account import Credentials as ServiceAccountCredentials
from googleapiclient.discovery import build, Resource
from googleapiclient.http import MediaIoBaseDownload
from tqdm import tqdm

# --- Configuration ---
CREDENTIAL_FILE = Path(r"C:\Users\A S U S\Documents\Keys\credentials.json")

# near real time url (2021 onwards)
#FOLDER_URL = "https://drive.google.com/drive/folders/1XBonFS0_t3aSM_C4CYobwsN-spmj29vo"

#past url (till 2020)
FOLDER_URL= "https://drive.google.com/drive/folders/1DVR90Ud1C444bTOPeENX-3I7tgqPLnao"
FOLDER_ID = FOLDER_URL.split('/folders/')[-1].split('?')[0]

# Scratch space on a LOCAL disk (the network share is slow for lots of small read/write cycles).
OUTPUT_TEMP_PATH = Path(tempfile.gettempdir()) / "MSWEP_TEMP"
OUTPUT_FINAL_PATH = Path(r"E:\ClimateData_Precipitation\Data\Datasets\MSWEP")
ROI_PATH = Path(r"E:\ClimateData_Precipitation\Data\Datasets\roi_nepal_rectangle\nepal_rectangle_shape.shp")

VARIABLE_NAME = "precipitation"
X_DIM, Y_DIM = "lon", "lat"
SCOPES = ['https://www.googleapis.com/auth/drive.readonly']

START_YEAR = 2001
END_YEAR = 2026            # exclusive

UTC_OFFSET_HOURS = 6       # Nepal = UTC+05:45 -> nearest multiple of the 3-h step
STAMP_IS_PERIOD_START = True   # VERIFY (see module docstring)
STEP_HOURS = 3
STEPS_PER_DAY = 24 // STEP_HOURS   # 8

NODATA_VALUE = -9999.0
MAX_RETRIES = 5


# --- Logging ---
OUTPUT_FINAL_PATH.mkdir(parents=True, exist_ok=True)
OUTPUT_TEMP_PATH.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    filename=OUTPUT_FINAL_PATH / "download_errors.log",
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)


def log_error(message: str):
    logging.error(message)
    tqdm.write(f"❌ {message}")      # tqdm.write keeps the progress bar intact


# --- Drive helpers ---
def initialize_drive_service() -> Optional[Resource]:
    try:
        creds = ServiceAccountCredentials.from_service_account_file(str(CREDENTIAL_FILE), scopes=SCOPES)
        return build('drive', 'v3', credentials=creds)
    except Exception as e:
        log_error(f"initialize_drive_service: Error loading credentials: {e}")
        return None


def build_drive_index(drive_service: Resource) -> Dict[str, dict]:
    """List the folder ONCE (paginated) and map filename -> {id, size}.
    Replaces ~73,000 individual files().list() calls."""
    index: Dict[str, dict] = {}
    page_token = None
    query = f"'{FOLDER_ID}' in parents and trashed=false"
    while True:
        for attempt in range(MAX_RETRIES):
            try:
                resp = drive_service.files().list(
                    q=query, spaces='drive', pageSize=1000, pageToken=page_token,
                    fields='nextPageToken, files(id, name, size)',
                    supportsAllDrives=True, includeItemsFromAllDrives=True,
                ).execute()
                break
            except Exception as e:
                log_error(f"build_drive_index: attempt {attempt + 1} failed: {e}")
                time.sleep(2 ** attempt)
        else:
            raise RuntimeError("Could not list the Drive folder after retries")

        for f in resp.get('files', []):
            index[f['name']] = {'id': f['id'], 'size': int(f['size']) if 'size' in f else None}
        page_token = resp.get('nextPageToken')
        if not page_token:
            return index


def is_valid_netcdf(path: Path, expected_size: Optional[int]) -> bool:
    try:
        if expected_size is not None and path.stat().st_size != expected_size:
            return False
        with xr.open_dataset(path, engine='netcdf4'):
            pass
        return True
    except Exception:
        return False


def download_file(drive_service: Resource, entry: dict, filename: str, out_dir: Path) -> Optional[Path]:
    dest = out_dir / filename
    if dest.exists():
        if is_valid_netcdf(dest, entry['size']):
            return dest
        log_error(f"Existing file {filename} is corrupt/incomplete. Re-downloading.")
        dest.unlink()

    for attempt in range(MAX_RETRIES):
        try:
            request = drive_service.files().get_media(fileId=entry['id'])
            with io.FileIO(str(dest), 'wb') as fh:
                downloader = MediaIoBaseDownload(fh, request, chunksize=10 * 1024 * 1024)
                done = False
                while not done:
                    _, done = downloader.next_chunk()
            if is_valid_netcdf(dest, entry['size']):
                return dest
            raise IOError("downloaded file failed size/NetCDF validation")
        except Exception as e:
            log_error(f"Attempt {attempt + 1} failed for {filename}: {e}")
            dest.unlink(missing_ok=True)
            if attempt < MAX_RETRIES - 1:
                time.sleep(2 ** attempt)
    return None


# --- Time handling ---
def utc_stamp_to_filename(dt: datetime) -> str:
    return f"{dt.year}{dt.timetuple().tm_yday:03d}.{dt.hour:02d}.nc"


def files_for_local_day(day: date) -> List[str]:
    """The 8 UTC 3-hourly files that make up one LOCAL calendar day (see module docstring)."""
    local_midnight_in_utc = datetime(day.year, day.month, day.day) - timedelta(hours=UTC_OFFSET_HOURS)
    first = local_midnight_in_utc if STAMP_IS_PERIOD_START else local_midnight_in_utc + timedelta(hours=STEP_HOURS)
    return [utc_stamp_to_filename(first + timedelta(hours=STEP_HOURS * i)) for i in range(STEPS_PER_DAY)]


# --- Raster processing ---
def load_clipped(nc_path: Path, roi_4326: gpd.GeoDataFrame) -> xr.DataArray:
    """Open one NetCDF, take the precipitation field, clip to ROI, load into memory."""
    with xr.open_dataset(nc_path, engine='netcdf4') as ds:
        da = ds[VARIABLE_NAME]
        if 'time' in da.dims:
            da = da.isel(time=0, drop=True)
        da = da.rio.set_spatial_dims(x_dim=X_DIM, y_dim=Y_DIM).rio.write_crs("EPSG:4326")
        da = da.rio.clip(roi_4326.geometry, roi_4326.crs, drop=True)
        return da.load()


def build_daily_total(nc_paths: List[Path], roi_4326: gpd.GeoDataFrame) -> xr.DataArray:
    """Sum the 3-hourly fields (mm / 3 h) into mm / day. NaN in any step -> NaN in the total
    (a silently partial sum would be biased low)."""
    total = None
    for p in nc_paths:
        da = load_clipped(p, roi_4326)
        total = da if total is None else total + da
    return total


def write_geotiff(total: xr.DataArray, out_path: Path) -> None:
    total = total.rio.set_spatial_dims(x_dim=X_DIM, y_dim=Y_DIM)
    if float(total[Y_DIM][0]) < float(total[Y_DIM][-1]):          # ascending lat -> south-up GeoTIFF
        total = total.sortby(Y_DIM, ascending=False)
    total = (total.astype('float32')
                  .fillna(NODATA_VALUE)
                  .rio.write_crs("EPSG:4326")
                  .rio.write_nodata(NODATA_VALUE, encoded=True))
    tmp = out_path.with_name(out_path.name + ".part")
    total.rio.to_raster(tmp, driver="GTiff", compress="deflate")
    os.replace(tmp, out_path)                                     # atomic: no half-written "final" files


def process_local_day(drive_service: Resource, index: Dict[str, dict], day: date,
                      roi_4326: gpd.GeoDataFrame) -> bool:
    out_path = OUTPUT_FINAL_PATH / f"MSWEP_Prec_{day:%Y%m%d}.tif"
    if out_path.exists():
        return True

    nc_paths: List[Path] = []
    try:
        for fname in files_for_local_day(day):
            entry = index.get(fname)
            if entry is None:
                log_error(f"{day}: {fname} not found in Drive folder -> day skipped (no partial sums)")
                return False
            p = download_file(drive_service, entry, fname, OUTPUT_TEMP_PATH)
            if p is None:
                log_error(f"{day}: could not download {fname} -> day skipped")
                return False
            nc_paths.append(p)

        total = build_daily_total(nc_paths, roi_4326)
        write_geotiff(total, out_path)
        logging.info(f"Saved {out_path.name}")
        return True
    except Exception as e:
        log_error(f"process_local_day {day}: {e}")
        return False
    finally:
        for p in nc_paths:
            p.unlink(missing_ok=True)


# --- Main ---
def main():
    assert UTC_OFFSET_HOURS % STEP_HOURS == 0, "UTC_OFFSET_HOURS must be a multiple of the 3-h step"
    print("🚀 Starting MSWEP download + local-day aggregation...")

    drive_service = initialize_drive_service()
    if not drive_service:
        print("❌ Failed to initialize Drive service. Exiting.")
        return

    try:
        roi = gpd.read_file(ROI_PATH)
        roi_4326 = roi.to_crs("EPSG:4326") if roi.crs is not None else roi.set_crs("EPSG:4326")
    except Exception as e:
        log_error(f"Failed to load ROI shapefile: {e}")
        return

    print("📂 Indexing Drive folder...")
    index = build_drive_index(drive_service)
    print(f"   {len(index)} files found")

    first_day, end_day = date(START_YEAR, 1, 1), date(END_YEAR, 1, 1)
    days = [first_day + timedelta(days=i) for i in range((end_day - first_day).days)]

    # Pre-flight: the first and last days reach into neighbouring UTC days / years
    for d in (days[0], days[-1]):
        missing = [f for f in files_for_local_day(d) if f not in index]
        if missing:
            print(f"⚠️  {d}: files missing from Drive: {missing}")

    failed: List[date] = []
    with tqdm(total=len(days), desc="Processing days", unit="day") as pbar:
        for d in days:
            if not process_local_day(drive_service, index, d, roi_4326):
                failed.append(d)
            pbar.update(1)

    if failed:
        fail_file = OUTPUT_FINAL_PATH / "failed_days.txt"
        fail_file.write_text("\n".join(d.isoformat() for d in failed))
        print(f"\n⚠️  {len(failed)} day(s) failed -> {fail_file}. Re-run the script to retry them.")
    print("\n✅ Processing complete.")


if __name__ == "__main__":
    main()