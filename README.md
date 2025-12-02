# Climate Data Analysis Workspace

Welcome to the Climate Data Analysis Workspace! This project is designed to download, process, and analyze various climate datasets (Precipitation, Temperature, etc.) for the Nepal region. This guide will help you understand the project structure and how to run the scripts.

## 📂 Project Structure

The workspace is organized as follows:

- **`Codes/`**: Contains all the Python scripts and Jupyter notebooks for data processing and analysis.
    - Scripts are numbered sequentially (e.g., `1_...`, `2_...`, `3_...`) to indicate the workflow order.
- **`Data/`**: Stores input and output data.
    - `Datasets/`: Final processed datasets (e.g., ERA5, CHIRPS, MSWEP).
    - `Raw/`: Raw input files (e.g., CSVs, NetCDF).
    - `Processed/`: Intermediate processed data.
    - `Figures/`: Generated plots and figures.
- **`Literature Review/`**: Contains reference papers and documents.
- **`Notes/`**: Project notes.

## 🛠 Prerequisites

Before running the codes, ensure you have the following:

1.  **Python Environment**: A Python environment with necessary geospatial libraries:
    - `geemap`, `ee` (Google Earth Engine)
    - `xarray`, `rioxarray`, `rasterio`, `gdal`
    - `pandas`, `numpy`, `geopandas`
    - `tqdm`, `colorama`, `retry`
2.  **Google Earth Engine (GEE)**: You need a GEE account. The scripts use `ee-jeevankatel3` project. You may need to authenticate using `ee.Authenticate()`.
3.  **Google Drive Credentials**: For MSWEP data, you need a `credentials.json` file (Service Account) to access the Google Drive folder.

## 🚀 Workflow & Scripts

The scripts are numbered to guide you through the process.

### Step 1: Data Quality Control
*   **`1_check_InSitu_data.ipynb`**: Checks in-situ station data (Rainfall, Temperature) for missing values, ranges, and consistency.

### Step 2: Data Downloading & Processing
These scripts download gridded climate data, process them (convert to TIF, clip to Nepal ROI), and save them.

*   **`2_1_download_era5_gee.py`**: Downloads **ERA5-Land** hourly precipitation from GEE, aggregates to daily sums, and exports as TIF.
*   **`2_2_download_har_ai.py`** / **`2_2_download_har_sj.py`**: Downloads **HARv2** hourly precipitation (NetCDF), converts to daily TIFs, reprojects, and clips to Nepal.
*   **`2_3_download_chirps.py`**: Downloads **CHIRPS** daily precipitation from GEE.
*   **`2_4_download_mswep.py`** (Recommended): Downloads **MSWEP** 3-hourly data from Google Drive, converts to TIF, and aggregates to daily sums.
*   **`2_5_download_persiann.py`**: Downloads **PERSIANN-CDR** daily precipitation from GEE.

### Step 3: Analysis
*   **`3_1_correlation_coefficient.ipynb`**: Calculates the correlation coefficient between ground station observations (in-situ) and the downloaded gridded datasets (e.g., ERA5).

## 📊 Data Sources

| Dataset | Type | Source | Script |
| :--- | :--- | :--- | :--- |
| **ERA5-Land** | Reanalysis | Google Earth Engine | `2_1_...` |
| **HARv2** | Reanalysis | TU Berlin (Web) | `2_2_...` |
| **CHIRPS** | Satellite | Google Earth Engine | `2_3_...` |
| **MSWEP** | Merged | Google Drive | `2_4_...` |
| **PERSIANN** | Satellite | Google Earth Engine | `2_5_...` |

## 📝 Usage Tips

- **Run Order**: Generally, follow the numbering: `1_` -> `2_` -> `3_`.
- **Paths**: Check the paths in the scripts before running. Most are set to relative paths or specific local paths (e.g., `Data/Datasets/...`).
- **Logs**: Check `error_log.txt` or `download_errors.log` in the output directories if a script fails.
- **Authentication**: If GEE fails, try running `ee.Authenticate()` in a separate notebook or terminal.

---
*Generated using Gemini 3 Pro for the Climate Data Workspace.*
