# Climate Data Analysis Workspace: Nepal Precipitation Research

Welcome to the **Climate Data Analysis Workspace**! This project is a comprehensive framework for downloading, processing, and analyzing multi-source climate data. It integrates in-situ station observations with satellite-based and reanalysis gridded products to perform spatial clustering, geostatistical interpolation (Ordinary Kriging), and machine learning-ready predictor preparation for the Nepal region.

**Last updated:** 2026-06-19
**Study Area:** Nepal (26.3°N–30.5°N, 80.0°E–88.2°E)
**Study Period:** 1990–2019 (30 years)

---

## 📂 Project Structure

```bash
CD-Workspace/
├── Codes/                      # Analysis modules (Python & Jupyter)
│   ├── 1_*.py/*.ipynb         # Multi-source Data Acquisition & Topographic Processing
│   ├── 2_*.ipynb              # In-situ Data Quality Control & Initial Accuracy Assessment
│   ├── 3_*.ipynb              # Spatial Clustering & Validation Splits
│   ├── 4A_*.ipynb             # Monthly Precipitation Modeling & Kriging Calibration
│   ├── 4B_*.ipynb             # Daily Precipitation Modeling & Classification
│   ├── 5_*.ipynb              # Model Validation, Interannual Variation & Resolution/Scale Analysis
│   └── archived_codes/        # Archived/deprecated code versions
├── Data/
│   ├── Datasets/              # Final Gridded Products (GeoTIFF)
│   │   ├── ERA5L_prec/        # ERA5-Land (Primary)
│   │   ├── CHIRPS_prec/       # CHIRPS v2.0
│   │   ├── HAR_prec/          # HARv2 (NetCDF converted)
│   │   ├── MSWEP_prec/        # MSWEP v2.8
│   │   ├── PERSIANN_prec/     # PERSIANN-CDR
│   │   ├── GLDAS/             # GLDAS v2.1 weather variables (air_temp, specific_humidity, wind_speed, etc.)
│   │   ├── DEM_1km/           # SRTM-derived Topography (Mean, Slope, etc.)
│   │   ├── MODIS_2_*/         # Vegetation/LST Variables (EVI, LST Day/Night)
│   │   └── TemplateRasters/   # GeoTIFF masks and templates
│   ├── Processed/             # Fetch variables, relief, and clustering outputs
│   └── Figures/               # Visualizations (Clustering results, Workflows)
├── Literature Review/         # Reference papers and synthetic summaries
└── Notes/                     # Methodological guides & supplementary documentation
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

### Step 1: Multi-Source Data Acquisition & Topographic Analysis
Downloads are parallelized and support automatic resumption.

| Script | Dataset / Task | Source | Spat/Temp Res |
|:---|:---|:---|:---|
| [1_1_download_era5_gee.py](file:///c:/Users/A%20S%20U%20S/Documents/Study/ClimateData/CD-Workspace/Codes/1_1_download_era5_gee.py) | **ERA5-Land** | ECMWF (GEE) | 11 km / Daily |
| [1_2_download_har_parallel_gemini.py](file:///c:/Users/A%20S%20U%20S/Documents/Study/ClimateData/CD-Workspace/Codes/1_2_download_har_parallel_gemini.py) | **HARv2** (Parallelized) | TU Berlin | 10 km / Daily |
| [1_2_download_har_sj.py](file:///c:/Users/A%20S%20U%20S/Documents/Study/ClimateData/CD-Workspace/Codes/1_2_download_har_sj.py) | **HARv2** (Single Thread) | TU Berlin | 10 km / Daily |
| [1_3_download_chirps.py](file:///c:/Users/A%20S%20U%20S/Documents/Study/ClimateData/CD-Workspace/Codes/1_3_download_chirps.py) | **CHIRPS** | UCSB (GEE) | 0.05° / Daily |
| [1_4_download_mswep.py](file:///c:/Users/A%20S%20U%20S/Documents/Study/ClimateData/CD-Workspace/Codes/1_4_download_mswep.py) | **MSWEP** | GloH2O (GDrive) | 0.1° / Daily |
| [1_5_download_persiann.py](file:///c:/Users/A%20S%20U%20S/Documents/Study/ClimateData/CD-Workspace/Codes/1_5_download_persiann.py) | **PERSIANN-CDR** | UC Irvine (GEE) | 0.25° / Daily |
| [1_6_download_gldas_weather_var.py](file:///c:/Users/A%20S%20U%20S/Documents/Study/ClimateData/CD-Workspace/Codes/1_6_download_gldas_weather_var.py) | **GLDAS v2.1** Atmospheric Vars | NASA (GEE) | Air Temp, Spec. Humidity, Wind Speed, Surf Pressure, Shortwave/Longwave Radiation, Soil Moisture |
| [1_7_download_modis_seconday_var_parallel.py](file:///c:/Users/A%20S%20U%20S/Documents/Study/ClimateData/CD-Workspace/Codes/1_7_download_modis_seconday_var_parallel.py) | **MODIS (MCD11 & MYD13)** | NASA (GEE) | EVI (16-day 1km), LST (8-day 1km) |
| [1_9_download_dem_variables.py](file:///c:/Users/A%20S%20U%20S/Documents/Study/ClimateData/CD-Workspace/Codes/1_9_download_dem_variables.py) | **SRTM DEM Topography** | CGIAR-CSI (GEE) | 90m SRTM resampled to 1km (Mean, SD, Min, Max, Slope, Aspect) |
| [1_10_prepare_fetch_variables.ipynb](file:///c:/Users/A%20S%20U%20S/Documents/Study/ClimateData/CD-Workspace/Codes/1_10_prepare_fetch_variables.ipynb) | Wind-terrain relief/fetch | WhiteboxTools (WBT) | Local relief, fetch distance, and direction for wind-terrain interaction |

---

### Step 2: In-Situ Quality Control & Initial Accuracy Assessment

| Script | Description |
|:---|:---|
| [2_1_qc_insitu_prec.ipynb](file:///c:/Users/A%20S%20U%20S/Documents/Study/ClimateData/CD-Workspace/Codes/2_1_qc_insitu_prec.ipynb) | Quality control of in-situ station records across Nepal (statistical outliers, missing value handling, physical consistency). |
| [2_2_assess_initial_accuracy.ipynb](file:///c:/Users/A%20S%20U%20S/Documents/Study/ClimateData/CD-Workspace/Codes/2_2_assess_initial_accuracy.ipynb) | Computes initial accuracy metrics comparing raw gridded precipitation datasets against in-situ station observations. |

---

### Step 3: Spatial Clustering & Validation Splits

| Script | Description |
|:---|:---|
| [3_1_1_var_preparation_for_spatial_clustering.ipynb](file:///c:/Users/A%20S%20U%20S/Documents/Study/ClimateData/CD-Workspace/Codes/3_1_1_var_preparation_for_spatial_clustering.ipynb) | Prepares climatological features (elevation, aspect, LST, vegetation indices, atmospheric variables) for spatial zoning. |
| [3_1_2_rf_fit_to_derive_weights.ipynb](file:///c:/Users/A%20S%20U%20S/Documents/Study/ClimateData/CD-Workspace/Codes/3_1_2_rf_fit_to_derive_weights.ipynb) | Uses Random Forest feature importance to determine predictor weights for spatial clustering. |
| [3_1_3_spatial_cluster_weighted_k_means.ipynb](file:///c:/Users/A%20S%20U%20S/Documents/Study/ClimateData/CD-Workspace/Codes/3_1_3_spatial_cluster_weighted_k_means.ipynb) | Groups meteorological stations using Weighted K-Means based on climato-topographic similarity. |
| [3_2_train_test_split_stations.ipynb](file:///c:/Users/A%20S%20U%20S/Documents/Study/ClimateData/CD-Workspace/Codes/3_2_train_test_split_stations.ipynb) | Establishes a spatially-aware training/testing split of stations for cross-validation to prevent spatial autocorrelation bias. |

---

### Step 4A: Monthly Precipitation Modeling Workflow
This sequence implements machine learning and geostatistical calibration specifically for monthly climatological analysis.

| Script | Description |
|:---|:---|
| [4A_1_1_var_preparation_for_monthly_model.ipynb](file:///c:/Users/A%20S%20U%20S/Documents/Study/ClimateData/CD-Workspace/Codes/4A_1_1_var_preparation_for_monthly_model.ipynb) | Standardizes and builds monthly predictor rasters (fetch, relief, secondary variables). |
| [4A_1_2_df_preparation_for_final_model.ipynb](file:///c:/Users/A%20S%20U%20S/Documents/Study/ClimateData/CD-Workspace/Codes/4A_1_2_df_preparation_for_final_model.ipynb) | Aligns in-situ observations with gridded predictors to create the final monthly training DataFrame. |
| [4A_2_regression_model_monthly.ipynb](file:///c:/Users/A%20S%20U%20S/Documents/Study/ClimateData/CD-Workspace/Codes/4A_2_regression_model_monthly.ipynb) | Trains machine learning regression models (e.g., Random Forest) on monthly precipitation datasets. |
| [4A_3_1_prepare_df_for_predict_rasters.ipynb](file:///c:/Users/A%20S%20U%20S/Documents/Study/ClimateData/CD-Workspace/Codes/4A_3_1_prepare_df_for_predict_rasters.ipynb) | Reorganizes monthly grid data into tabular format for fast spatial prediction. |
| [4A_3_2_predict_rasters.ipynb](file:///c:/Users/A%20S%20U%20S/Documents/Study/ClimateData/CD-Workspace/Codes/4A_3_2_predict_rasters.ipynb) | Predicts high-resolution spatial precipitation rasters across Nepal using the monthly models. |
| [4A_4_calibration_kriging.ipynb](file:///c:/Users/A%20S%20U%20S/Documents/Study/ClimateData/CD-Workspace/Codes/4A_4_calibration_kriging.ipynb) | Performs residual Kriging calibration, blending machine learning outputs with station residuals. |
| [4A_5_1_variogram_analysis_best_gridded_product.ipynb](file:///c:/Users/A%20S%20U%20S/Documents/Study/ClimateData/CD-Workspace/Codes/4A_5_1_variogram_analysis_best_gridded_product.ipynb) | Fits experimental semivariograms on the residuals of the best raw gridded product. |
| [4A_5_2_krige_gridded_product.ipynb](file:///c:/Users/A%20S%20U%20S/Documents/Study/ClimateData/CD-Workspace/Codes/4A_5_2_krige_gridded_product.ipynb) | Generates Ordinary Kriging estimates using gridded product residuals. |
| [4A_6_1_krige_insitu_data.ipynb](file:///c:/Users/A%20S%20U%20S/Documents/Study/ClimateData/CD-Workspace/Codes/4A_6_1_krige_insitu_data.ipynb) | Generates monthly rainfall surfaces purely from in-situ observations using geostatistical interpolation (Ordinary Kriging). |

---

### Step 4B: Daily Precipitation Bias Correction Workflow
This sequence handles daily-scale workflows, which include two-stage models to resolve zero-inflation (classification + regression).

| Script | Description |
|:---|:---|
| [4B_1_krige_insitu_daily.ipynb](file:///c:/Users/A%20S%20U%20S/Documents/Study/ClimateData/CD-Workspace/Codes/4B_1_krige_insitu_daily.ipynb) | Direct daily Ordinary Kriging interpolation of station observations across Nepal. |
| [4B_2_var_preparation_for_daily_model.ipynb](file:///c:/Users/A%20S%20U%20S/Documents/Study/ClimateData/CD-Workspace/Codes/4B_2_var_preparation_for_daily_model.ipynb) | Compiles and prepares predictors for the daily model. |
| [4B_3_classification_model_daily.ipynb](file:///c:/Users/A%20S%20U%20S/Documents/Study/ClimateData/CD-Workspace/Codes/4B_3_classification_model_daily.ipynb) | Trains a daily classification model to predict precipitation occurrence (Rain vs. No-Rain) to address zero-inflation. |
| [4B_4_regression_model_daily.ipynb](file:///c:/Users/A%20S%20U%20S/Documents/Study/ClimateData/CD-Workspace/Codes/4B_4_regression_model_daily.ipynb) | Trains a daily machine learning regression model to predict rainfall amount on wet days. |

---

### Step 5: Model Validation, Interannual Variation & Scale Optimization

| Script | Description |
|:---|:---|
| [5_1_validation_accuracy.ipynb](file:///c:/Users/A%20S%20U%20S/Documents/Study/ClimateData/CD-Workspace/Codes/5_1_validation_accuracy.ipynb) | Assesses model accuracy on validation datasets using continuous (RMSE, R², NSE, KGE, MAE) and categorical (POD, FAR, CSI) metrics. |
| [5_2_interannual_variation_captured_by_skill.ipynb](file:///c:/Users/A%20S%20U%20S/Documents/Study/ClimateData/CD-Workspace/Codes/5_2_interannual_variation_captured_by_skill.ipynb) | Evaluates how effectively the developed algorithms capture interannual climate variation trends. |
| [5_x_Variogram_analysis_resolutions_sonnet4.6.ipynb](file:///c:/Users/A%20S%20U%20S/Documents/Study/ClimateData/CD-Workspace/Codes/5_x_Variogram_analysis_resolutions_sonnet4.6.ipynb) | Analyzes spatial autocorrelation structures using experimental semivariograms across multiple grid resolutions. |
| [5_x_block_aggregation_test_sonnet4.6.ipynb](file:///c:/Users/A%20S%20U%20S/Documents/Study/ClimateData/CD-Workspace/Codes/5_x_block_aggregation_test_sonnet4.6.ipynb) | Conducts block aggregation checks to investigate scale effects on model performance. |
| [5_x_determine_resolution_from_skill_sonnet4.6.ipynb](file:///c:/Users/A%20S%20U%20S/Documents/Study/ClimateData/CD-Workspace/Codes/5_x_determine_resolution_from_skill_sonnet4.6.ipynb) | Identifies the optimal spatial resolution (grid cell size) based on spatial predictor skill and validation metrics. |

---

## 📊 Data Sources & Citations

| Dataset Category | Product | Source/Platform | Citation Focus |
|:---|:---|:---|:---|
| **Reanalysis** | ERA5-Land | [ECMWF via GEE](https://cds.climate.copernicus.eu/) | Muñoz-Sabater et al. (2021) |
| **Reanalysis** | HARv2 | [TU Berlin](https://www.tu.berlin/) | Pradhananga et al. (2021) |
| **Satellite** | CHIRPS v2.0 | [UCSB via GEE](https://chc.ucsb.edu/data/chirps) | Funk et al. (2015) |
| **Merged** | MSWEP v2.8 | [GloH2O](https://www.gloh2o.org/mswep/) | Beck et al. (2019) |
| **Satellite** | PERSIANN-CDR| [UC Irvine via GEE](https://chrs.web.uci.edu/) | Ashouri et al. (2015) |
| **Reanalysis/Assimilation**| GLDAS v2.1 | [NASA via GEE](https://ldas.gsfc.nasa.gov/gldas) | Rodell et al. (2004) |
| **Topography** | SRTM v4.1 | [CGIAR-CSI via GEE](https://srtm.csi.cgiar.org/) | Jarvis et al. (2008) |
| **Vegetation/LST**| MODIS (MCD11 & MYD13) | [NASA via GEE](https://lpdaac.usgs.gov/) | Wan et al. (2015) |

---

## 📝 Usage & Developer Notes

- **Execution Order**: The pipeline generally follows the sequence: Acquisition (`1_x`) ➔ Quality Control (`2_x`) ➔ Clustering (`3_x`) ➔ Modeling (Monthly `4A_x` and/or Daily `4B_x`) ➔ Validation & Resolution Optimization (`5_x`).
- **Resumability**: Download scripts automatically verify existing datasets in `Data/Datasets` to bypass redundant API calls.
- **Internal Docs**: Refer to the `Notes/` directory for mathematical formulations of geostatistical interpolation (Ordinary Kriging), categorical metrics, and details on resolution-related validation.
- **Literature Reference**: The `Literature Review/` folder contains comprehensive summaries of key references underpinning the methodology.

---
*Climate Data Workspace for Nepal | Managed by Antigravity AI Assistant*
