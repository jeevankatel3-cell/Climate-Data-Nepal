# Libraries
import os
import sys
import logging
import numpy as np
import pandas as pd
import joblib
import json
import gc
import math
import matplotlib.pyplot as plt

from quantile_forest import RandomForestQuantileRegressor
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, make_scorer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.inspection import permutation_importance
from sklearn.feature_selection import RFECV
from sklearn.model_selection import GroupKFold

# Path configuration
log_dir = "Daily_bias_correction/logs"
tmax_train_p = r"Daily_bias_correction/final_tmax_test_df.parquet"
tmax_test_p = r"Daily_bias_correction/final_tmax_test_df.parquet"

tmin_train_p = r"Daily_bias_correction/final_tmin_train_df.parquet"
tmin_test_p = r"Daily_bias_correction/final_tmin_test_df.parquet"

regressor_model_path = r"Daily_bias_correction/Saved_regressors"
rscv_results_path = r"Daily_bias_correction/RSCV_results"
validation_results_path = r"Daily_bias_correction/Validation_results"
hpt_results_path = r"Daily_bias_correction/HPT_results"

os.makedirs(log_dir, exist_ok=True)
os.makedirs(regressor_model_path, exist_ok=True)
os.makedirs(rscv_results_path, exist_ok=True)
os.makedirs(validation_results_path, exist_ok=True)
os.makedirs(hpt_results_path, exist_ok=True)

# Logging Setup (Outputs to both log file and terminal console)
log_file_path = os.path.join(log_dir, "pipeline.log")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_file_path, mode="a"),
        logging.StreamHandler(sys.stdout)
    ]
)

logging.info("================ Running Bias Correction Pipeline ================")

# Configurations
random_state_num = 64
n_cores = 4
target_variable_tmax = 'tmax'
target_variable_tmin = 'tmin'

predictor_features_tmax = [
    'y', 'x', 'season_num',
    'chirts_tmax', 'era5l_tmax', 'har_tmax', 'har_tmax_lag1',
    'gldas_air_temp', 'gldas_lwdown', 'gldas_specific_humidity',
    'gldas_surface_pressure', 'gldas_swdown',
    'elev_mean', 'elev_std', 'elev_max', 'elev_min', 'slope_mean', 'aspect_mean',
    'elev_smoothed', 'fetch_1000m_dist', 'fetch_1000m_dir',
    'fetch_500m_dist', 'fetch_500m_dir', 'local_relief',
    'doy_sin', 'doy_cos'
]

predictor_features_tmin = [
    'y', 'x', 'season_num',
    'chirts_tmin', 'era5l_tmin', 'har_tmin', 'har_tmin_lag1',
    'gldas_air_temp', 'gldas_lwdown', 'gldas_specific_humidity',
    'gldas_surface_pressure', 'gldas_swdown',
    'elev_mean', 'elev_std', 'elev_max', 'elev_min', 'slope_mean', 'aspect_mean',
    'elev_smoothed', 'fetch_1000m_dist', 'fetch_1000m_dir',
    'fetch_500m_dist', 'fetch_500m_dir', 'local_relief',
    'doy_sin', 'doy_cos'
]

# Helpers

def get_param_grids():
    """Single QRF hyperparameter space."""
    return {
        'n_estimators': [500, 1000, 3000, 5000],
        'max_features': ['sqrt', 'log2'],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
    }

def compute_metrics(y_true, y_pred, label):
    """Compute RMSE, MAE, R² and log with label."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    logging.info(f"[{label}] RMSE={rmse:.4f}  MAE={mae:.4f}  R²={r2:.4f}  n={len(y_true)}")
    return {'rmse': round(rmse, 4), 'mae': round(mae, 4), 'r2': round(r2, 4), 'n': len(y_true)}

SEASONS = {
    'winter': [12, 1, 2],
    'pre_monsoon': [3, 4, 5],
    'monsoon': [6, 7, 8, 9],
    'post_monsoon': [10, 11],
}

def get_season(month):
    for s, months in SEASONS.items():
        if month in months:
            return s
    return None

SEASONS_NUM = {
    1: 'winter',
    2: 'pre_monsoon',
    3: 'monsoon',
    4: 'post_monsoon'
}

def get_season_num(season_str):
    for s_n, s_name in SEASONS_NUM.items():
        if s_name == season_str:
            return s_n
    return None

class QRFPredictor:
    def __init__(self, pipeline, y_train, input_features, selected_features):
        self.pipeline = pipeline
        self.y_train = np.asarray(y_train)
        self.input_features = input_features
        self.selected_features = selected_features

    @property
    def qrf_(self):
        return self.pipeline.named_steps['rfecv_reg'].estimator_

    def transform_to_selected(self, X):
        X_imp = self.pipeline.named_steps['imputer'].transform(X[self.input_features])
        mask = self.pipeline.named_steps['rfecv_reg'].support_
        return X_imp[:, mask]

    def predict_quantile(self, X, quantiles=0.5):
        return self.qrf_.predict(self.transform_to_selected(X), quantiles=quantiles)

    def predict_median(self, X):
        return self.predict_quantile(X, quantiles=0.5)

    def sample(self, X, n_realizations=100, random_state=None, max_proximities=None):
        rng = np.random.default_rng(random_state)
        X_sel = self.transform_to_selected(X)
        proximities = self.qrf_.proximity_counts(X_sel, max_proximities=max_proximities)

        n_rows = len(proximities)
        draws = np.empty((n_rows, n_realizations), dtype=float)
        for i, prox in enumerate(proximities):
            train_idx = np.fromiter((p[0] for p in prox), dtype=int, count=len(prox))
            weights = np.fromiter((p[1] for p in prox), dtype=float, count=len(prox))
            probs = weights / weights.sum()
            chosen = rng.choice(train_idx, size=n_realizations, p=probs)
            draws[i, :] = self.y_train[chosen]
        return draws

# 1. Load dfs and format for fitting
logging.info("Loading dataset Parquet files...")
tmax_train_df = pd.read_parquet(tmax_train_p)
tmax_test_df = pd.read_parquet(tmax_test_p)
tmin_train_df = pd.read_parquet(tmin_train_p)
tmin_test_df = pd.read_parquet(tmin_test_p)

logging.info("Cleaning missing values and generating seasonal features...")
tmax_train_df = tmax_train_df.dropna(subset=[target_variable_tmax])
tmax_test_df = tmax_test_df.dropna(subset=[target_variable_tmax])
tmin_train_df = tmin_train_df.dropna(subset=[target_variable_tmin])
tmin_test_df = tmin_test_df.dropna(subset=[target_variable_tmin])

for df in [tmax_train_df, tmax_test_df, tmin_train_df, tmin_test_df]:
    df['season'] = df['month'].apply(get_season)
    df['season_num'] = df['season'].apply(get_season_num)

# Tmax train/test arrays
X_train_tmax = tmax_train_df[predictor_features_tmax].copy()
y_train_tmax = tmax_train_df[target_variable_tmax].copy()
X_test_tmax = tmax_test_df[predictor_features_tmax].copy()
y_test_tmax = tmax_test_df[target_variable_tmax].copy()

# Tmin train/test arrays
X_train_tmin = tmin_train_df[predictor_features_tmin].copy()
y_train_tmin = tmin_train_df[target_variable_tmin].copy()
X_test_tmin = tmin_test_df[predictor_features_tmin].copy()
y_test_tmin = tmin_test_df[target_variable_tmin].copy()

train_data = {
    'tmax': (X_train_tmax, y_train_tmax),
    'tmin': (X_train_tmin, y_train_tmin)
}

test_data = {
    'tmax': (X_test_tmax, y_test_tmax),
    'tmin': (X_test_tmin, y_test_tmin)
}

groups_data = {
    'tmax': tmax_train_df['index_no'].values,
    'tmin': tmin_train_df['index_no'].values,
}

# 2. Fit and tune the QRF regressor using Randomized Search CV
param_dist_qrf = get_param_grids()
hpt_results_df = pd.DataFrame()
group_cv = GroupKFold(n_splits=5)

for t_variable in ['tmax', 'tmin']:
    logging.info(f"--- Starting tuning and fitting for target: {t_variable} ---")
    
    X_train, y_train = train_data[t_variable]
    X_test, y_test = test_data[t_variable]
    groups = groups_data[t_variable]

    base_model = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('reg', RandomForestQuantileRegressor(
            default_quantiles=0.5,
            random_state=random_state_num,
            n_jobs=n_cores,
        ))
    ])
    param_dist = {f'reg__{k}': v for k, v in param_dist_qrf.items()}

    search_cv = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_dist,
        n_iter=50,
        cv=group_cv,
        verbose=1,
        random_state=random_state_num,
        n_jobs=n_cores,
        scoring='neg_root_mean_squared_error',
    )

    logging.info(f"Fitting RandomizedSearchCV (50 iterations) on {t_variable} dataset...")
    search_cv.fit(X_train, y_train, groups=groups)
    logging.info(f"Completed RandomizedSearchCV for {t_variable}.")

    best_model = search_cv.best_estimator_

    rscv_results = pd.DataFrame(search_cv.cv_results_)
    rscv_file = os.path.join(rscv_results_path, f"{t_variable}_qrf.csv")
    rscv_results.to_csv(rscv_file, index=False)
    logging.info(f"Saved RSCV results to: {rscv_file}")

    best_params = search_cv.best_params_
    best_params_clean = {k.replace('reg__', ''): v for k, v in best_params.items()}
    logging.info(f"Best parameters for {t_variable}: {best_params_clean}")

    test_metrics = compute_metrics(y_test, best_model.predict(X_test), f'Test Score: {t_variable} - qrf')

    summary_row = {
        't_variable': t_variable,
        'model_type': 'qrf',
        'test_rmse': test_metrics['rmse'],
        **best_params_clean
    }

    hpt_results_df = pd.concat([hpt_results_df, pd.DataFrame([summary_row])], ignore_index=True)

hpt_out_file = os.path.join(hpt_results_path, "hpt_results.csv")
hpt_results_df.to_csv(hpt_out_file, index=False)
logging.info(f"Pipeline execution finished successfully. Summary saved to {hpt_out_file}.")