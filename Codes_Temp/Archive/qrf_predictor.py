# qrf_predictor.py
import numpy as np


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