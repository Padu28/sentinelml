"""Feature engineering: a single, leakage-safe preprocessing pipeline.

All transformations are fitted on training data only, inside a scikit-learn
``Pipeline``, so the exact same steps are applied at inference time.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

LOG_FEATURES = ("load_kw",)


def _log1p_safe(x: np.ndarray) -> np.ndarray:
    return np.log1p(np.clip(x, 0, None))


def build_preprocessor(numeric: list[str], categorical: list[str]) -> ColumnTransformer:
    log_cols = [c for c in numeric if c in LOG_FEATURES]
    lin_cols = [c for c in numeric if c not in LOG_FEATURES]
    transformers = []
    if lin_cols:
        transformers.append(
            ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), lin_cols)
        )
    if log_cols:
        transformers.append(
            (
                "log",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median")),
                        ("log", FunctionTransformer(_log1p_safe, feature_names_out="one-to-one")),
                        ("scale", StandardScaler()),
                    ]
                ),
                log_cols,
            )
        )
    if categorical:
        transformers.append(
            (
                "cat",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                categorical,
            )
        )
    return ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=False)


def select_features(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    missing = [f for f in features if f not in df.columns]
    if missing:
        raise KeyError(f"Missing feature columns: {missing}")
    return df[features]
