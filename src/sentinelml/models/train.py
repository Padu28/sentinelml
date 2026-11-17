"""Model training, prediction and evaluation."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from sentinelml.common.config import get_settings
from sentinelml.common.utils import timer
from sentinelml.features.engineering import build_preprocessor

ALGORITHMS = {
    "logistic_regression": LogisticRegression,
    "random_forest": RandomForestClassifier,
    "hist_gradient_boosting": HistGradientBoostingClassifier,
}


def make_estimator(algorithm: str | None = None, random_state: int | None = None) -> Any:
    cfg = get_settings().model
    algorithm = algorithm or cfg["algorithm"]
    if algorithm == "xgboost":  # optional, compared fairly when installed
        from xgboost import XGBClassifier  # type: ignore

        return XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.08, eval_metric="logloss")
    params = dict(cfg["params"].get(algorithm, {}))
    rs = cfg["random_state"] if random_state is None else random_state
    if "random_state" in ALGORITHMS[algorithm]().get_params():
        params["random_state"] = rs
    return ALGORITHMS[algorithm](**params)


def train_model(
    df: pd.DataFrame,
    algorithm: str | None = None,
    random_state: int | None = None,
) -> Pipeline:
    s = get_settings()
    pipe = Pipeline(
        [
            ("prep", build_preprocessor(s.numeric_features, s.categorical_features)),
            ("clf", make_estimator(algorithm, random_state)),
        ]
    )
    pipe.fit(df[s.features], df[s.target].astype(int))
    return pipe


def predict_proba(model: Pipeline, df: pd.DataFrame) -> np.ndarray:
    s = get_settings()
    return model.predict_proba(df[s.features])[:, 1]


def classification_metrics(y_true: np.ndarray, proba: np.ndarray, threshold: float = 0.5) -> dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    pred = (proba >= threshold).astype(int)
    two_classes = len(np.unique(y_true)) > 1
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, proba)) if two_classes else float("nan"),
        "pr_auc": float(average_precision_score(y_true, proba)) if two_classes else float("nan"),
    }


def evaluate(model: Pipeline, df: pd.DataFrame) -> dict[str, float]:
    s = get_settings()
    with timer() as t:
        proba = predict_proba(model, df)
    metrics = classification_metrics(df[s.target].to_numpy(), proba)
    metrics["latency_ms_per_1k"] = t["ms"] / max(len(df), 1) * 1000
    return metrics


def latency_profile(model: Pipeline, df: pd.DataFrame, n_trials: int = 30) -> dict[str, float]:
    """Single-record latency percentiles (ms)."""
    s = get_settings()
    rows = df[s.features].sample(min(n_trials, len(df)), random_state=0)
    times = []
    for i in range(len(rows)):
        with timer() as t:
            model.predict_proba(rows.iloc[[i]])
        times.append(t["ms"])
    return {"p50_ms": float(np.percentile(times, 50)), "p95_ms": float(np.percentile(times, 95))}
