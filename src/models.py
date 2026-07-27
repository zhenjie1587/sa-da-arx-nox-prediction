"""Model factories and the proposed sensor-availability-aware predictor."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import (
    ExtraTreesRegressor,
    HistGradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

from config import RANDOM_SEED, RIDGE_ALPHA


def scaled_pipeline(estimator, *, add_indicator: bool = False) -> Pipeline:
    return Pipeline(
        [
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                    add_indicator=add_indicator,
                    keep_empty_features=True,
                ),
            ),
            ("scaler", StandardScaler()),
            ("regressor", estimator),
        ]
    )


def tree_pipeline(estimator, *, add_indicator: bool = False) -> Pipeline:
    return Pipeline(
        [
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                    add_indicator=add_indicator,
                    keep_empty_features=True,
                ),
            ),
            ("regressor", estimator),
        ]
    )


def make_baseline_models() -> dict[str, Pipeline]:
    """Return executable comparison models using the same delayed ARX inputs."""
    return {
        "SVR-ARX": scaled_pipeline(
            MultiOutputRegressor(
                SVR(C=100.0, epsilon=0.1, gamma="scale"),
                n_jobs=-1,
            )
        ),
        "RF-ARX": tree_pipeline(
            RandomForestRegressor(
                n_estimators=250,
                max_features=0.8,
                min_samples_leaf=1,
                random_state=RANDOM_SEED,
                n_jobs=-1,
            )
        ),
        "ExtraTrees-ARX": tree_pipeline(
            ExtraTreesRegressor(
                n_estimators=300,
                max_features=0.8,
                min_samples_leaf=1,
                random_state=RANDOM_SEED,
                n_jobs=-1,
            )
        ),
        "HistGBR-ARX": tree_pipeline(
            MultiOutputRegressor(
                HistGradientBoostingRegressor(
                    max_iter=300,
                    learning_rate=0.05,
                    max_leaf_nodes=31,
                    l2_regularization=1.0,
                    random_state=RANDOM_SEED,
                ),
                n_jobs=-1,
            )
        ),
        "MLP-ARX": scaled_pipeline(
            MLPRegressor(
                hidden_layer_sizes=(96, 48),
                activation="relu",
                solver="adam",
                alpha=1e-3,
                learning_rate_init=1e-3,
                max_iter=500,
                early_stopping=True,
                validation_fraction=0.15,
                n_iter_no_change=25,
                random_state=RANDOM_SEED,
            )
        ),
    }


def make_ridge(*, robust: bool) -> Pipeline:
    return scaled_pipeline(
        Ridge(alpha=RIDGE_ALPHA),
        add_indicator=robust,
    )


@dataclass
class SensorAvailabilityAwareModel:
    """Hard-gated clean and dropout-trained delayed ARX experts."""

    clean_expert: Pipeline
    robust_expert: Pipeline

    def predict(self, features):
        missing = np.asarray(features.isna().any(axis=1), dtype=bool)
        clean_prediction = self.clean_expert.predict(features)
        if not missing.any():
            return clean_prediction
        robust_prediction = self.robust_expert.predict(features)
        prediction = clean_prediction.copy()
        prediction[missing] = robust_prediction[missing]
        return prediction

