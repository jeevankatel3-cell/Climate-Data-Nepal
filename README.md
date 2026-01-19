# Climate Data Analysis Workspace: Nepal Precipitation Research

Welcome to the **Climate Data Analysis Workspace**! This project is a comprehensive framework for downloading, processing, and analyzing multi-source climate data. It integrates in-situ station observations with satellite-based and reanalysis gridded products to perform spatial clustering, geostatistical interpolation (Ordinary Kriging), and machine learning-ready predictor preparation for the Nepal region.

**Last updated:** 2026-01-19
**Study Area:** Nepal (26.3°N–30.5°N, 80.0°E–88.2°E)
**Study Period:** 1990–2019 (30 years)

---

## 📂 Project Structure

```bash
CD-Workspace/
├── Codes/                      # Analysis modules (Python & Jupyter)
│   ├── 1_*.ipynb              # In-situ Data Quality Control
│   ├── 2_*.py/*.ipynb         # Multi-source Data Acquisition & Processing
│   ├── 3_*.ipynb              # Geostatistics & Spatial Clustering
│   └── 4_*.ipynb              # Validation & Accuracy Metrics
├── Data/
│   ├── Datasets/              # Final Gridded Products (GeoTIFF)
│   │   ├── ERA5L_prec/        # ERA5-Land (Primary)
│   │   ├── CHIRPS_prec/       # CHIRPS v2.0
│   │   ├── HAR_prec/          # HARv2 (NetCDF converted)
│   │   ├── MSWEP_prec/        # MSWEP v2.8
│   │   ├── PERSIANN_prec/     # PERSIANN-CDR
│   │   ├── DEM_1km/           # SRTM-derived Topography (Mean, Slope, etc.)
│   │   └── ERA5L_2_*/         # Atmospheric Secondary Variables
│   ├── Processed/             # Fetch variables, relief, and clustering outputs
│   └── Figures/               # Visualizations (Clustering results, Workflows)
├── Literature Review/         # 37+ Reference papers and synthetic summaries
└── Notes/                     # Methodological guides (Kriging, Clustering, Metrics)
```

---

## 🛠 Prerequisites & Environment

The workspace relies on a robust geospatial stack. Key libraries include:
- **Cloud Geospatial**: `geemap`, `ee` (Google Earth Engine)
- **Raster Processing**: `xarray`, `rioxarray`, `rasterio`, `whitebox` (WBT), `gdal`
- **Vector & Data**: `geopandas`, `pandas`, `numpy`, `shapely`
- **Analysis**: `scikit-learn`, `geostatspy`, `pykrige`, `scipy`
- **Validation**: `permetrics`, `HydroErr`

### Authentication Requirements
1. **GEE**: Project `ee-jeevankatel3`. Run `ee.Authenticate()` for initial setup.
2. **Google Drive**: `credentials.json` required for high-resolution MSWEP access.

---

## 🚀 Research Workflow

### Step 1: In-Situ Quality Control
| Script | Description |
|:---|:---|
| `1_1_qc_insitu_prec.ipynb` | Validates station data across Nepal, checking for missing values, statistical outliers, and physical consistency. |

---

### Step 2: Multi-Source Data Acquisition & Topographic Analysis
Downloads are parallelized and support automatic resumption.

#### Primary Precipitation & Atmosphere
| Script | Dataset | Source | Spat/Temp Res |
|:---|:---|:---|:---|
| `2_1_download_era5_gee.py` | **ERA5-Land** | ECMWF (GEE) | 11 km / Daily |
| `2_2_download_har_parallel.py`| **HARv2** | TU Berlin | 10 km / Daily |
| `2_3_download_chirps.py` | **CHIRPS** | UCSB (GEE) | 0.05° / Daily |
| `2_4_download_mswep.py` | **MSWEP** | GloH2O (GDrive) | 0.1° / Daily |
| `2_5_download_persiann.py` | **PERSIANN-CDR** | UC Irvine (GEE) | 0.25° / Daily |
| `2_6_download_era5L_sec.py` | Atmospheric Vars | ECMWF (GEE) | Wind, Dewpt, Pressure |
| `2_7_download_modis_sec.py` | EVI/LST | NASA (GEE) | 1 km / 8-day/16-day |

#### Topographic & Morphometric Variables (SRTM)
| Script | Description |
|:---|:---|
| `2_9_download_dem_variables.py` | Downloads SRTM 90m and processes into 1km mean, std, min, max elevation, slope, and aspect. |
| `2_10_prepare_fetch.ipynb` | Uses **WhiteboxTools** to calculate local relief and fetch (distance/direction to ridge) for wind-terrain interaction analysis. |

---

### Step 3: Geostatistics & Spatial Machine Learning
This step focuses on characterizing the spatial structure of precipitation.

| Script | Phase | Description |
|:---|:---|:---|
| `3_1_1_var_prep.ipynb` | **Feature Prep** | Generates climatological rasters for 21+ variables (DEM, MODIS, ERA5L). |
| `3_1_2_rf_weights.ipynb` | **Weighting** | Fits Random Forest models to determine the importance of predictors for spatial clusters. |
| `3_1_3_spatial_cluster.ipynb` | **Clustering** | Implements **Weighted K-Means** to group stations based on climatic and topographic similarity. |
| `3_2_train_test_split.ipynb` | **Validation Split** | Performs spatially-aware splitting of stations for rigorous cross-validation. |
| `3_3_variogram_analysis.ipynb`| **Variography** | Fits Gaussian, Spherical, and Exponential models to seasonal precipitation residuals. |
| `3_4_prepare_kriging.ipynb` | **Interpolation** | Prepares Ordinary Kriging surfaces as core predictor variables. |

---

### Step 4: Validation & Accuracy Assessment
| Script | Description |
|:---|:---|
| `4_1_assess_accuracy.ipynb` | Computes continuous metrics (**RMSE, R², NSE, KGE, MAE**) and categorical scores (**POD, FAR, CSI**) to assess gridded product performance against ground truth. |

---

## 📊 Data Sources & Citations

| Dataset Category | Product | Source/Platform | Citation Focus |
|:---|:---|:---|:---|
| **Reanalysis** | ERA5-Land | [ECMWF via GEE](https://cds.climate.copernicus.eu/) | Muñoz-Sabater et al. (2021) |
| **Reanalysis** | HARv2 | [TU Berlin](https://www.tu.berlin/) | Pradhananga et al. (2021) |
| **Satellite** | CHIRPS v2.0 | [UCSB via GEE](https://chc.ucsb.edu/data/chirps) | Funk et al. (2015) |
| **Merged** | MSWEP v2.8 | [GloH2O](https://www.gloh2o.org/mswep/) | Beck et al. (2019) |
| **Satellite** | PERSIANN-CDR| [UC Irvine via GEE](https://chrs.web.uci.edu/) | Ashouri et al. (2015) |
| **Topography** | SRTM v4.1 | [CGIAR-CSI via GEE](https://srtm.csi.cgiar.org/) | Jarvis et al. (2008) |
| **Vegetation/LST**| MODIS (MCD11) | [NASA via GEE](https://lpdaac.usgs.gov/) | Wan et al. (2015) |

---

## 📝 Usage & Developer Notes

- **Execution Order**: Scripts are designed as a sequential pipeline (`1_x` → `2_x` → `3_x` → `4_x`).
- **Resumability**: Download scripts automatically check for existing files in `Data/Datasets` to prevent redundant API calls.
- **Internal Docs**: Refer to the `Notes/` directory for detailed mathematical derivations of the **Categorical Metrics** and **Ordinary Kriging** parameters used in this study.
- **Literature Reference**: The `Literature Review/` folder contains a synthetic summary (generated by NotebookLM) of the 37+ key papers underpinning the methodology.

---
*Climate Data Workspace for Nepal | Managed by Antigravity AI Assistant*
