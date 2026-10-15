# %%
# =============================================================================
# QRF Daily Bias-Correction Regression Model
# =============================================================================
# Version history:
#   V3  - Replaced RF/XGB with a single shared QRF core (deterministic +
#         stochastic read modes).
#   V4A - Branched for Linux/AWS; uses 1% of total df for trial runs.
#   V5  - HPT runs on a cluster-stratified station subsample (HPT_SAMPLE_FRAC).
#         GroupKFold groups by `cluster` (leave-one-zone-out) everywhere.
#   V6  - HPT subsample switched to plain ROW-level stratified sampling by
#         cluster; station-level sampling no longer needed.
#   V7  - RFECV removed. Final model fits tuned QRF directly on the full
#         training set; Step 5 permutation importance serves as the feature-
#         relevance evidence for the paper.
#   V8  - Standalone pipeline .pkl dropped from Step 3; only the
#         QRFPredictor bundle is saved.
#
# Workflow:
#   1. Load train/test frames, format predictors/targets
#   2. HPT: RandomizedSearchCV over QRF space, cluster-stratified row
#      subsample, GroupKFold(groups=cluster), scored on median RMSE
#   3. Final model: fit tuned QRF on full training set, save QRFPredictor
#      bundle (.predict_median + .sample from ONE shared fitted forest)
#   4. Validate on held-out test set (deterministic median read)
#   5. Permutation feature importance at multiple quantile levels
#   6. Plot importances
#   7. Sanity-check dual read modes (reproducibility + sample shape)

# %%
# =============================================================================
# Imports
# =============================================================================
import os
import ast
import gc
import json
import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import joblib

from joblib import parallel_backend
from quantile_forest import RandomForestQuantileRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    mean_absolute_error,
    mean_pinball_loss,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import GroupKFold, RandomizedSearchCV
from sklearn.pipeline import Pipeline

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


# %%
# =============================================================================
# QRFPredictor class
# =============================================================================
class QRFPredictor:
    """
    Bundles a fitted imputer+QRF pipeline with its training targets, exposing
    two read modes from ONE shared fitted forest:

        .predict_median(X)           -> deterministic median
        .sample(X, n_realizations)   -> stochastic inverse-CDF draws

    No RFECV (removed in fitting script V7): every column in `input_features`
    is used. `selected_features` mirrors `input_features` to keep the public
    interface compatible with the older RFECV version.
    """

    def __init__(self, pipeline, y_train, input_features, selected_features):
        self.pipeline = pipeline
        self.y_train = np.asarray(y_train)
        self.input_features = input_features
        self.selected_features = selected_features

    @property
    def qrf_(self):
        """The fitted RandomForestQuantileRegressor."""
        return self.pipeline.named_steps['reg']

    def transform_to_input(self, X):
        """Impute X; all columns in `input_features` pass straight to QRF."""
        return self.pipeline.named_steps['imputer'].transform(X[self.input_features])

    def predict_quantile(self, X, quantiles=0.5):
        return self.qrf_.predict(self.transform_to_input(X), quantiles=quantiles)

    def predict_median(self, X):
        return self.predict_quantile(X, quantiles=0.5)

    def sample(self, X, n_realizations=100, random_state=None):
        rng = np.random.default_rng(random_state)
        random_quantiles = rng.uniform(0.0, 1.0, size=n_realizations)
        return self.qrf_.predict(self.transform_to_input(X), quantiles=random_quantiles.tolist())


# %%
# =============================================================================
# Paths and configuration
# =============================================================================

# Input data paths
tmax_train_p = fr"../Data/Processed/Daily_bias_correction/final_tmax_train_df.parquet"
tmax_test_p  = fr"../Data/Processed/Daily_bias_correction/final_tmax_test_df.parquet"
tmin_train_p = fr"../Data/Processed/Daily_bias_correction/final_tmin_train_df.parquet"
tmin_test_p  = fr"../Data/Processed/Daily_bias_correction/final_tmin_test_df.parquet"

# Output paths (separate from the RFECV version for side-by-side comparison)
regressor_model_path    = fr"../Data/Processed/Daily_bias_correction/Saved_regressors"
rscv_results_path       = fr"../Data/Processed/Daily_bias_correction/RSCV_results"
validation_results_path = fr"../Data/Processed/Daily_bias_correction/Validation_results"
hpt_results_path        = fr"../Data/Processed/Daily_bias_correction/HPT_results"
plots_path              = fr"../Data/Processed/Daily_bias_correction/plots"

for _p in [regressor_model_path, rscv_results_path,
           validation_results_path, hpt_results_path, plots_path]:
    os.makedirs(_p, exist_ok=True)

# General settings
random_state_num     = 64
n_cores              = 2
target_variable_tmax = 'tmax'
target_variable_tmin = 'tmin'

# Cluster / GroupKFold settings
# `cluster` (0-3, 4 spatial zones) drives two things:
#   1. HPT subsample: frac of each cluster's rows drawn independently at random.
#   2. GroupKFold cross-validation: leave-one-zone-out.
# N_GROUPKFOLD_SPLITS must equal the number of distinct clusters.
CLUSTER_COL         = 'cluster'
STATION_COL         = 'index_no'   # not a predictor; used only for diagnostics
HPT_SAMPLE_FRAC     = 0.20
N_GROUPKFOLD_SPLITS = 4            # == number of distinct clusters (0-3)

# Predictor feature lists
predictor_features_tmax = [
    'y', 'x', 'season_num',                                              # spatial coords & season
    'chirts_tmax', 'era5l_tmax', 'har_tmax',                            # gridded tmax products
    'gldas_air_temp', 'gldas_lwdown', 'gldas_specific_humidity',        # GLDAS weather
    'gldas_surface_pressure', 'gldas_swdown',
    'elev_mean', 'elev_std', 'elev_max', 'elev_min',                   # DEM / fetch variables
    'slope_mean', 'aspect_mean', 'elev_smoothed',
    'fetch_1000m_dist', 'fetch_1000m_dir',
    'fetch_500m_dist', 'fetch_500m_dir', 'local_relief', 'cluster',
    'doy_sin', 'doy_cos',                                               # cyclic day-of-year
]

predictor_features_tmin = [
    'y', 'x', 'season_num',                                              # spatial coords & season
    'chirts_tmin', 'era5l_tmin', 'har_tmin',                            # gridded tmin products
    'gldas_air_temp', 'gldas_lwdown', 'gldas_specific_humidity',        # GLDAS weather
    'gldas_surface_pressure', 'gldas_swdown',
    'elev_mean', 'elev_std', 'elev_max', 'elev_min',                   # DEM / fetch variables
    'slope_mean', 'aspect_mean', 'elev_smoothed',
    'fetch_1000m_dist', 'fetch_1000m_dir',
    'fetch_500m_dist', 'fetch_500m_dir', 'local_relief', 'cluster',
    'doy_sin', 'doy_cos',                                               # cyclic day-of-year
]


# %%
# =============================================================================
# Helper functions
# =============================================================================

def get_param_grids():
    """QRF hyperparameter search space."""
    return {
        'n_estimators':      [100, 250, 500, 1000, 2000, 4000],
        'max_features':      ['sqrt', 'log2'],
        'max_depth':         [10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf':  [1, 2, 4],
    }


def compute_metrics(y_true, y_pred, label):
    """Compute and print RMSE, MAE, R2; return dict."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    print(f"    [{label}]  RMSE={rmse:.4f}  MAE={mae:.4f}  R2={r2:.4f}  n={len(y_true)}")
    return {'rmse': round(rmse, 4), 'mae': round(mae, 4), 'r2': round(r2, 4), 'n': len(y_true)}


SEASONS = {
    'winter':       [12, 1, 2],
    'pre_monsoon':  [3, 4, 5],
    'monsoon':      [6, 7, 8, 9],
    'post_monsoon': [10, 11],
}

def get_season(month):
    for s, months in SEASONS.items():
        if month in months:
            return s
    return None


SEASONS_NUM = {1: 'winter', 2: 'pre_monsoon', 3: 'monsoon', 4: 'post_monsoon'}

def get_season_num(season_str):
    for s_n, s_name in SEASONS_NUM.items():
        if s_name == season_str:
            return s_n
    return None


def cluster_proportional_subsample(df, cluster_col, frac, random_state):
    """
    Proportionate cluster-stratified row sample: draws `frac` of each
    cluster's rows independently at random (no station-awareness needed since
    GroupKFold groups by cluster, not station).

    Station coverage is still near-complete in practice -- the chance a
    station contributes zero rows is (1-frac)^n_rows, negligible for any
    station with more than a few dozen records. Actual coverage is logged
    in Step 2.

    Returns pd.DataFrame with the original index preserved (usable with .loc[]).
    """
    return df.groupby(cluster_col, group_keys=False).sample(frac=frac, random_state=random_state)


def pick_model_parameters(t_var: str):
    """Read the best hyperparameters from the saved RSCV results CSV."""
    rscv_df = pd.read_csv(os.path.join(rscv_results_path, f"{t_var}_qrf.csv"))
    row = rscv_df[rscv_df['rank_test_score'] == 1].iloc[0]
    params = ast.literal_eval(row['params'])
    return {k.replace('reg__', ''): v for k, v in params.items()}


def make_pinball_scorer(bundle, alpha):
    """
    Return a scorer for permutation_importance that evaluates negative pinball
    loss at quantile `alpha` using bundle.predict_quantile (not the pipeline's
    default median predict).
    """
    def scorer(estimator, X, y):
        y_pred = bundle.predict_quantile(X, quantiles=alpha)
        return -mean_pinball_loss(y, y_pred, alpha=alpha)
    return scorer


# %%
# =============================================================================
# Step 1 -- Load and format train/test data
# =============================================================================
logging.info("Step 1: Loading and formatting train/test dataframes...")

tmax_train_df = pd.read_parquet(tmax_train_p)
tmax_test_df  = pd.read_parquet(tmax_test_p)
tmin_train_df = pd.read_parquet(tmin_train_p)
tmin_test_df  = pd.read_parquet(tmin_test_p)

tmax_train_df = tmax_train_df.dropna(subset=[target_variable_tmax])
tmax_test_df  = tmax_test_df.dropna(subset=[target_variable_tmax])
tmin_train_df = tmin_train_df.dropna(subset=[target_variable_tmin])
tmin_test_df  = tmin_test_df.dropna(subset=[target_variable_tmin])

for df in [tmax_train_df, tmax_test_df, tmin_train_df, tmin_test_df]:
    df['season']     = df['month'].apply(get_season)
    df['season_num'] = df['season'].apply(get_season_num)

# Sanity check: confirm the expected cluster count before a long HPT run.
for _name, _df in [('tmax_train', tmax_train_df), ('tmin_train', tmin_train_df)]:
    _n_clusters_found = _df[CLUSTER_COL].nunique()
    if _n_clusters_found != N_GROUPKFOLD_SPLITS:
        logging.warning(
            f"{_name}: expected {N_GROUPKFOLD_SPLITS} clusters, found "
            f"{_n_clusters_found} ({sorted(_df[CLUSTER_COL].unique())}). "
            f"GroupKFold(n_splits={N_GROUPKFOLD_SPLITS}) will fail if a "
            f"subsample ends up with fewer unique clusters."
        )

# Build float32 X/y arrays
X_train_tmax = tmax_train_df[predictor_features_tmax].copy().astype('float32')
y_train_tmax = tmax_train_df[target_variable_tmax].copy().astype('float32')
X_test_tmax  = tmax_test_df[predictor_features_tmax].copy().astype('float32')
y_test_tmax  = tmax_test_df[target_variable_tmax].copy().astype('float32')

X_train_tmin = tmin_train_df[predictor_features_tmin].copy().astype('float32')
y_train_tmin = tmin_train_df[target_variable_tmin].copy().astype('float32')
X_test_tmin  = tmin_test_df[predictor_features_tmin].copy().astype('float32')
y_test_tmin  = tmin_test_df[target_variable_tmin].copy().astype('float32')

train_data = {
    'tmax': (X_train_tmax, y_train_tmax),
    'tmin': (X_train_tmin, y_train_tmin),
}
test_data = {
    'tmax': (X_test_tmax, y_test_tmax),
    'tmin': (X_test_tmin, y_test_tmin),
}
# Full train DataFrames kept separately -- STATION_COL is not a predictor
# column so it's absent from X_train_*, but needed for HPT diagnostics.
train_df_data = {
    'tmax': tmax_train_df,
    'tmin': tmin_train_df,
}


# %%
# =============================================================================
# Step 2 -- Hyperparameter tuning (RandomizedSearchCV on cluster subsample)
# =============================================================================
# HPT trains on HPT_SAMPLE_FRAC of each cluster's rows (compute/RAM lever).
# Scoring: RMSE of the median read (Part C accuracy criterion).
# GroupKFold(groups=cluster) = leave-one-zone-out.
# The final fit in Step 3 uses the full training set.

param_dist_qrf = get_param_grids()
hpt_results_df = pd.DataFrame()

logging.info("Step 2: Hyperparameter tuning with RandomizedSearchCV...")
for t_variable in ['tmax', 'tmin']:
    full_train_df              = train_df_data[t_variable]
    X_train_full, y_train_full = train_data[t_variable]
    X_test, y_test             = test_data[t_variable]

    # Build the HPT subsample and align X/y arrays to it
    hpt_sample_df = cluster_proportional_subsample(
        df=full_train_df, cluster_col=CLUSTER_COL,
        frac=HPT_SAMPLE_FRAC, random_state=random_state_num,
    )
    hpt_idx = hpt_sample_df.index
    X_train = X_train_full.loc[hpt_idx]
    y_train = y_train_full.loc[hpt_idx]
    groups  = hpt_sample_df[CLUSTER_COL].values

    assert len(np.unique(groups)) >= N_GROUPKFOLD_SPLITS, (
        f"{t_variable}: HPT subsample has {len(np.unique(groups))} clusters, "
        f"need {N_GROUPKFOLD_SPLITS} -- raise HPT_SAMPLE_FRAC."
    )

    # Diagnostics: verify cluster row-shares and station coverage
    full_cluster_share   = full_train_df[CLUSTER_COL].value_counts(normalize=True).sort_index()
    sample_cluster_share = hpt_sample_df[CLUSTER_COL].value_counts(normalize=True).sort_index()
    station_coverage = (
        hpt_sample_df.groupby(CLUSTER_COL)[STATION_COL].nunique()
        / full_train_df.groupby(CLUSTER_COL)[STATION_COL].nunique()
    )
    logging.info(
        f"{t_variable}: HPT subsample = {len(X_train)}/{len(X_train_full)} rows "
        f"({len(X_train) / len(X_train_full):.1%})"
    )
    logging.info(
        f"{t_variable}: cluster row-share  full={full_cluster_share.round(3).to_dict()}  "
        f"sample={sample_cluster_share.round(3).to_dict()}"
    )
    logging.info(
        f"{t_variable}: station coverage per cluster = {station_coverage.round(3).to_dict()}"
    )

    # Build pipeline and run search
    base_model = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('reg', RandomForestQuantileRegressor(
            default_quantiles=0.5,  # predict() with no quantiles arg -> median
            random_state=random_state_num,
            n_jobs=n_cores,
        )),
    ])
    param_dist = {f'reg__{k}': v for k, v in param_dist_qrf.items()}

    search_cv = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_dist,
        n_iter=50,
        cv=GroupKFold(n_splits=N_GROUPKFOLD_SPLITS),
        verbose=3,
        random_state=random_state_num,
        n_jobs=n_cores,
        scoring='neg_root_mean_squared_error',
    )

    with parallel_backend('loky', inner_max_num_threads=1):
        search_cv.fit(X_train, y_train, groups=groups)

    best_model = search_cv.best_estimator_

    # Save full CV results
    pd.DataFrame(search_cv.cv_results_).to_csv(
        os.path.join(rscv_results_path, f"{t_variable}_qrf.csv"), index=False
    )

    best_params_clean = {k.replace('reg__', ''): v for k, v in search_cv.best_params_.items()}

    # Evaluate best HPT model on the full test set (this model is not saved;
    # Step 3 refits fresh on the full training set using these params).
    test_metrics = compute_metrics(
        y_test, best_model.predict(X_test), f'Test Score: {t_variable} - qrf'
    )

    summary_row = {
        't_variable':          t_variable,
        'model_type':          'qrf',
        'hpt_sample_frac':     HPT_SAMPLE_FRAC,
        'hpt_sample_rows':     len(X_train),
        'hpt_sample_stations': hpt_sample_df[STATION_COL].nunique(),
        'hpt_stations_total':  full_train_df[STATION_COL].nunique(),
        'test_rmse':           test_metrics['rmse'],
        **best_params_clean,
    }
    hpt_results_df = pd.concat([hpt_results_df, pd.DataFrame([summary_row])], ignore_index=True)

    del hpt_sample_df, hpt_idx, X_train, y_train, groups, X_train_full, y_train_full
    gc.collect()

print(hpt_results_df)

# %% [markdown]
# # Checkpoint 1
# HPT results saved above. To resume: rerun the imports/config cell and the
# Step 1 data cell, then continue from Step 3 below.

# %%
# =============================================================================
# Step 3 -- Fit final model on full training set (no RFECV)
# =============================================================================
# Loads the best hyperparameters from Step 2's saved CSV, fits QRF on the
# full training set, and saves a single QRFPredictor bundle (pipeline + y_train).
# No standalone pipeline file is saved -- bundle.pipeline is the only artifact.
# `selected_features` == `input_features` (all features used; no elimination).

for t_var in ['tmax', 'tmin']:
    logging.info(f"Step 3: Building final QRF model for {t_var}")
    print(f"\nBuilding final QRF model for {t_var}")

    parameter_set = pick_model_parameters(t_var)
    print(parameter_set)

    X_train, y_train = train_data[t_var]
    feature_order    = X_train.columns.tolist()

    regressor = RandomForestQuantileRegressor(
        default_quantiles=0.5,
        random_state=random_state_num,
        n_jobs=n_cores,
        verbose=2,
        **parameter_set,
    )
    reg_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('reg', regressor),
    ])

    with parallel_backend('loky', inner_max_num_threads=1):
        reg_pipeline.fit(X_train, y_train)

    # All features used (no RFECV elimination); `selected_features` kept as a
    # separate variable to match the RFECV version's interface.
    regressor_selected_features = feature_order
    print(f"  [QRF] Features used: {len(regressor_selected_features)} / {X_train.shape[1]} (no elimination)")

    # Bundle pipeline + training targets for dual-read-mode access downstream
    predictor_bundle = QRFPredictor(
        pipeline=reg_pipeline,
        y_train=y_train.to_numpy(),
        input_features=feature_order,
        selected_features=regressor_selected_features,
    )
    joblib.dump(predictor_bundle, os.path.join(regressor_model_path, f"{t_var}_qrf_predictor.pkl"))

    metadata = {
        'model_type':       'qrf',
        'input_features':   feature_order,
        'selected_feature': regressor_selected_features,
    }
    with open(os.path.join(regressor_model_path, f"{t_var}_qrf_feature_metadata.json"), 'w') as f:
        json.dump(metadata, f, indent=2)

    del X_train, y_train, regressor, reg_pipeline, regressor_selected_features, predictor_bundle, metadata
    gc.collect()
    print(f"QRF predictor bundle + metadata saved for: {t_var}")


# %%
# =============================================================================
# Step 4 -- Validate on held-out test set (deterministic median read)
# =============================================================================
# predict_median() evaluates the quantile function at p=0.5 (Part C).

for t_var in ['tmax', 'tmin']:
    logging.info(f"Step 4: Validating metrics for {t_var}")

    X_test, y_test = test_data[t_var]

    bundle: QRFPredictor = joblib.load(os.path.join(regressor_model_path, f"{t_var}_qrf_predictor.pkl"))
    with open(os.path.join(regressor_model_path, f"{t_var}_qrf_feature_metadata.json")) as f:
        metadata = json.load(f)

    compute_metrics(y_test, bundle.predict_median(X_test), f"{t_var}-{metadata['model_type']}")
    print(f"With features: {metadata['selected_feature']}\n")

    del bundle, metadata
    gc.collect()


# %%
# =============================================================================
# Step 5 -- Permutation feature importance at multiple quantile levels
# =============================================================================
# Scored with negative pinball loss (correct scoring rule for quantile
# predictions). Runs one full permutation_importance pass per quantile level.
# With RFECV gone, all predictors are present -- correlated features
# (e.g. 3 gridded products, 5 elevation variants) may understate individual
# importances; pair with a correlation/VIF check when writing up results.

QUANTILE_LEVELS = [0.05, 0.20, 0.50, 0.80, 0.95]
n_repeatsP      = 10
max_samplesP    = 0.25
perm_jobsP      = 1

for t_var in ['tmax', 'tmin']:
    logging.info(f"Step 5: Permutation importance for {t_var}")
    X_test, y_test = test_data[t_var]

    bundle: QRFPredictor = joblib.load(os.path.join(regressor_model_path, f"{t_var}_qrf_predictor.pkl"))

    all_rows = []
    for alpha in QUANTILE_LEVELS:
        print(f"\nPermutation importance -- {t_var}, quantile={alpha}")
        scorer = make_pinball_scorer(bundle, alpha)
        perm = permutation_importance(
            bundle.pipeline, X_test, y_test,
            scoring=scorer,
            n_repeats=n_repeatsP,
            max_samples=max_samplesP,
            random_state=random_state_num,
            n_jobs=perm_jobsP,
        )
        for i, feat in enumerate(X_test.columns):
            all_rows.append({
                'variable':        t_var,
                'model':           'qrf',
                'quantile':        alpha,
                'feature':         feat,
                'importance_mean': round(perm.importances_mean[i], 6),
                'importance_std':  round(perm.importances_std[i], 6),
            })

    individual_df = pd.DataFrame(all_rows)
    # Rank within each quantile separately (magnitudes not comparable across quantiles)
    individual_df['rank'] = (
        individual_df.groupby('quantile')['importance_mean']
        .rank(ascending=False, method='min').astype(int)
    )
    individual_df = individual_df.sort_values(
        ['quantile', 'importance_mean'], ascending=[True, False]
    ).reset_index(drop=True)
    individual_df.to_csv(
        os.path.join(validation_results_path, f"perm_imp_{t_var}_qrf.csv"), index=False
    )

    del bundle
    gc.collect()


# %%
# =============================================================================
# Step 6 -- Plot feature importances
# =============================================================================

sns.set_theme(style="whitegrid", font_scale=1.0)
fig, axes = plt.subplots(2, len(QUANTILE_LEVELS), figsize=(28, 12), sharex=False)

targets = ['tmax', 'tmin']
colors  = ['#d95f02', '#1b9e77']

for row, target_var in enumerate(targets):
    df = pd.read_csv(os.path.join(validation_results_path, f"perm_imp_{target_var}_qrf.csv"))
    for col, alpha in enumerate(QUANTILE_LEVELS):
        ax   = axes[row, col]
        df_q = df[df['quantile'] == alpha].sort_values('importance_mean', ascending=True)
        ax.barh(
            y=df_q['feature'],
            width=df_q['importance_mean'],
            xerr=df_q['importance_std'],
            color=colors[row],
            alpha=0.85,
            capsize=3,
            edgecolor='black',
            linewidth=0.6,
        )
        ax.set_title(f"{target_var}  q={alpha}", fontsize=11, fontweight='bold')
        ax.set_xlabel("Increase in pinball loss\nwhen permuted", fontsize=9)
        ax.set_ylabel(target_var if col == 0 else "", fontsize=11, fontweight='bold')

plt.tight_layout()
plot_out_path = os.path.join(plots_path, "fig_perm_imp_combined.png")
plt.savefig(plot_out_path, dpi=300, bbox_inches='tight', transparent=True)
plt.close(fig)
logging.info(f"Plot saved to {plot_out_path}")


# %%
# =============================================================================
# Step 7 -- Sanity-check dual read modes
# =============================================================================
# Verifies:
#   (a) Part C reproducibility: predict_median() == predict_quantile(q=0.5)
#       bit-for-bit (not computed by averaging samples).
#   (b) Part D sampling: .sample() returns the expected (n_rows, n_realizations)
#       shape; its mean differs from the median as expected for an asymmetric
#       empirical distribution.

n_check_rows        = 500   # small slice -- design-time check, not a production run
n_realizations_check = 100

for t_var in ['tmax', 'tmin']:
    logging.info(f"Step 7: Sanity-checking dual read modes for {t_var}")
    X_test, y_test = test_data[t_var]
    X_check = X_test.iloc[:n_check_rows]

    bundle: QRFPredictor = joblib.load(os.path.join(regressor_model_path, f"{t_var}_qrf_predictor.pkl"))

    median_pred   = bundle.predict_median(X_check)
    direct_median = bundle.predict_quantile(X_check, quantiles=0.5)

    # (a) Both calls must follow the exact same code path
    assert np.allclose(median_pred, direct_median), (
        "median must come from quantiles=0.5 directly, not from averaging samples"
    )

    samples     = bundle.sample(X_check, n_realizations=n_realizations_check, random_state=random_state_num)
    sample_mean = samples.mean(axis=1)

    print(
        f"[{t_var}] median vs. mean-of-{n_realizations_check}-samples, "
        f"mean abs diff = {np.mean(np.abs(median_pred - sample_mean)):.4f} "
        f"(expected: close but not identical for an asymmetric distribution)"
    )
    print(f"[{t_var}] sample ensemble shape: {samples.shape}  (rows, realizations)")

    del bundle
    gc.collect()
