# Climate Data Analysis Workspace

Welcome to the **Climate Data Analysis Workspace**! This project is designed to download, process, analyze, and validate various gridded precipitation and secondary climate datasets against in-situ station observations for the Nepal region.

**Last updated:** 2025-12-15

---

## 📂 Project Structure

```
CD-Workspace/
├── Codes/                      # Python scripts and Jupyter notebooks
│   ├── 1_*.ipynb              # Data Quality Control
│   ├── 2_*.py/*.ipynb         # Data Downloading & Processing
│   ├── 3_*.ipynb              # Analysis (Variogram, Clustering)
│   └── 4_*.ipynb              # Validation & Accuracy Assessment
├── Data/
│   ├── Datasets/              # Final gridded datasets (TIF format)
│   │   ├── ERA5L_prec/        # ERA5-Land precipitation
│   │   ├── CHIRPS_prec/       # CHIRPS precipitation
│   │   ├── HAR_prec/          # HARv2 precipitation
│   │   ├── MSWEP_prec/        # MSWEP precipitation
│   │   ├── PERSIANN_prec/     # PERSIANN-CDR precipitation
│   │   └── ERA5L_2_*/         # Secondary variables (wind, dewpoint, etc.)
│   ├── Raw/                   # Raw input files (CSVs, NetCDF)
│   ├── Processed/             # Intermediate processed data
│   └── Figures/               # Generated plots and figures
├── Literature Review/         # Reference papers and summaries
└── Notes/                     # Project notes and documentation
```

---

## 🛠 Prerequisites

### Python Environment
Required libraries for geospatial analysis:
- **GEE**: `geemap`, `ee` (Google Earth Engine Python API)
- **Raster**: `xarray`, `rioxarray`, `rasterio`, `gdal`
- **Vector**: `geopandas`, `shapely`
- **Data**: `pandas`, `numpy`
- **Geostatistics**: `geostatspy`, `pykrige`, `scipy`
- **Metrics**: `permetrics`, `HydroErr`
- **Utilities**: `tqdm`, `colorama`, `retry`

### Authentication
1. **Google Earth Engine**: Requires a GEE account. Uses project `ee-jeevankatel3`. Run `ee.Authenticate()` if needed.
2. **Google Drive Credentials**: For MSWEP data, a service account `credentials.json` is required.

---

## 🚀 Workflow & Scripts

Scripts are numbered sequentially to guide you through the process. Follow the order: `1_` → `2_` → `3_` → `4_`.

### Step 1: Data Quality Control
| Script | Description |
|:---|:---|
| `1_1_qc_insitu_prec.ipynb` | Quality control of in-situ station data (Rainfall, Temperature). Checks for missing values, range validation, and data consistency. |

---

### Step 2: Data Downloading & Processing

#### Primary Precipitation Datasets
| Script | Dataset | Source | Resolution | Notes |
|:---|:---|:---|:---|:---|
| `2_1_download_era5_gee.py` | **ERA5-Land** | GEE | ~11 km | Hourly → Daily sum |
| `2_2_download_har_parallel_gemini.py` | **HARv2** | TU Berlin | 10 km | NetCDF → Daily TIF |
| `2_2_download_har_sj.py` | **HARv2** (alt) | TU Berlin | 10 km | Alternative download script |
| `2_3_download_chirps.py` | **CHIRPS** | GEE | 0.05° | Daily precipitation |
| `2_4_download_mswep.py` | **MSWEP** | Google Drive | 0.1° | 3-hourly → Daily sum |
| `2_5_download_persiann.py` | **PERSIANN-CDR** | GEE | 0.25° | Daily precipitation |

#### Secondary Variables (ERA5-Land)
| Script | Variables |
|:---|:---|
| `2_6_download_era5L_secondary_var.py` | Wind Speed (U/V), Dewpoint Temperature, Surface Pressure, Soil Moisture, Total Evaporation |

#### Secondary Variables (MODIS)
| Script | Variables |
|:---|:---|
| `2_8_download_modis_secondary_var.py` | EVI (Enhanced Vegetation Index), LST Day/Night/Difference |

#### Kriging Preparation
| Script | Description |
|:---|:---|
| `2_9_prepare_kriging_predictor.ipynb` | Prepares Ordinary Kriging of station observations as predictor variables for spatial interpolation. |

---

### Step 3: Geostatistical Analysis

| Script | Description |
|:---|:---|
| `3_1_geospatial_clustering.ipynb` | Geospatial clustering analysis of station data |
| `3_2_validation_split_prec.ipynb` | Split data for validation purposes |
| `3_3_variogram_analysis_OK.ipynb` | **Variogram Analysis** for Ordinary Kriging. Performs seasonal variogram analysis (Winter, Pre-monsoon, Monsoon, Post-monsoon) with Gaussian, Spherical, and Exponential models. Outputs: RMSE, WRSS, Nugget, Sill, Range. |

---

### Step 4: Validation & Accuracy Assessment

| Script | Description |
|:---|:---|
| `4_1_assess_accuracy.ipynb` | Calculates correlation coefficients and error metrics between in-situ observations and gridded datasets. Computes continuous (RMSE, MAE, R², NSE) and categorical (POD, FAR, CSI) metrics. |

---

## 📊 Data Sources Summary

| Dataset | Type | Temporal | Spatial | Source |
|:---|:---|:---|:---|:---|
| **ERA5-Land** | Reanalysis | 1990–2019 | ~11 km | ECMWF via GEE |
| **HARv2** | Reanalysis | 1990–2019 | 10 km | TU Berlin |
| **CHIRPS** | Satellite | 1990–2019 | 0.05° | UCSB via GEE |
| **MSWEP** | Merged | 1990–2019 | 0.1° | Princeton GDrive |
| **PERSIANN-CDR** | Satellite | 1990–2019 | 0.25° | UC Irvine via GEE |
| **MODIS** | Satellite | 2000+ | 1 km | NASA via GEE |

---

## � Research Context

This project supports research on **precipitation estimation using machine learning regression approaches** with a focus on integrating:

1. **Multi-source gridded precipitation products**: Satellite-based (CHIRPS, PERSIANN), reanalysis (ERA5-Land, HARv2), and merged (MSWEP) datasets.
2. **In-situ station observations**: Ground-truth validation from 106+ meteorological stations across Nepal.
3. **Geostatistical methods**: Ordinary Kriging with variogram analysis for spatial interpolation.
4. **Secondary predictor variables**: Topography (DEM), vegetation (EVI), land surface temperature (LST), atmospheric conditions (dewpoint, pressure, wind).
5. **Accuracy assessment**: Comprehensive validation using continuous error metrics (RMSE, MAE, R², NSE, KGE) and categorical scores (POD, FAR, CSI).

### Study Area
- **Region**: Nepal (26.3°N–30.5°N, 80.0°E–88.2°E)
- **Study Period**: 1990–2019 (30 years)
- **Focus**: Seasonal precipitation patterns (Monsoon-dominated climate)

---

## 📝 Usage Tips

1. **Run Order**: Follow the script numbering `1_` → `2_` → `3_` → `4_`
2. **Paths**: Most scripts use relative paths. Update paths if directory structure differs.
3. **Logs**: Check `error_log.txt` or `download_errors.log` for troubleshooting.
4. **Threading**: Download scripts use multi-threading (15 workers) for efficiency.
5. **Resume**: Scripts skip existing files automatically to support resumable downloads.
6. **GEE Auth**: If GEE fails, run `ee.Authenticate()` in a separate session.

---

## 📚 Key References

- **ERA5-Land**: Muñoz Sabater, J. (2019). ERA5-Land hourly data. Copernicus Climate Data Store.
- **CHIRPS**: Funk, C., et al. (2015). The climate hazards infrared precipitation with stations—a new environmental record for monitoring extremes. Scientific Data.
- **MSWEP**: Beck, H.E., et al. (2019). MSWEP V2 Global 3-Hourly 0.1° Precipitation. Bull. Amer. Meteor. Soc.
- **HARv2**: Pradhananga, D., et al. (2021). Technical Note: The High Asia Refined Analysis (HAR).

---

*Last generated: 2025-12-15 | Climate Data Workspace for Nepal Precipitation Analysis*
