"""Global and local explanations.

* Global: permutation importance (model-agnostic, measured on held-out data).
* Local : SHAP if installed; otherwise an exact additive decomposition for linear
          models, or a reference-substitution sensitivity for any other model.

Explanations describe *model behaviour* (associations), not causal effects.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from sentinelml.common.config import get_settings


def global_importance(model: Pipeline, df: pd.DataFrame, n_repeats: int = 5, seed: int = 0) -> pd.DataFrame:
    s = get_settings()
    res = permutation_importance(model, df[s.features], df[s.target], scoring="roc_auc",
                                 n_repeats=n_repeats, random_state=seed)
    return (
        pd.DataFrame({"feature": s.features, "importance": res.importances_mean, "std": res.importances_std})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


def _shap_local(model: Pipeline, row: pd.DataFrame, background: pd.DataFrame) -> dict[str, float] | None:
    try:
        import shap  # type: ignore
    except ImportError:
        return None
    s = get_settings()
    f = lambda X: model.predict_proba(pd.DataFrame(X, columns=s.features))[:, 1]  # noqa: E731
    explainer = shap.Explainer(f, background[s.features].sample(min(100, len(background)), random_state=0))
    sv = explainer(row[s.features])
    return dict(zip(s.features, map(float, sv.values[0])))


def local_explanation(model: Pipeline, row: pd.DataFrame, background: pd.DataFrame) -> dict[str, Any]:
    """Return per-feature contributions for a single record (row is a 1-row frame)."""
    s = get_settings()
    proba = float(model.predict_proba(row[s.features])[:, 1][0])
    shap_vals = _shap_local(model, row, background)
    if shap_vals is not None:
        return {"method": "shap", "probability": proba, "contributions": shap_vals}

    clf = model.named_steps["clf"]
    prep = model.named_steps["prep"]
    if isinstance(clf, LogisticRegression):
        x = prep.transform(row[s.features])[0]
        base = prep.transform(background[s.features]).mean(axis=0)
        names = prep.get_feature_names_out()
        contrib_t = clf.coef_[0] * (x - base)        # log-odds contribution vs average record
        agg: dict[str, float] = {f: 0.0 for f in s.features}
        for name, c in zip(names, contrib_t):
            src = next((f for f in s.features if name == f or name.startswith(f + "_")), name)
            agg[src] = agg.get(src, 0.0) + float(c)
        return {"method": "linear_log_odds", "probability": proba, "contributions": agg}

    # Generic: substitute each feature with its background median / mode
    contrib: dict[str, float] = {}
    for f in s.features:
        alt = row.copy()
        if f in s.categorical_features:
            alt[f] = background[f].mode().iloc[0]
        else:
            alt[f] = background[f].median()
        contrib[f] = proba - float(model.predict_proba(alt[s.features])[:, 1][0])
    return {"method": "reference_substitution", "probability": proba, "contributions": contrib}


def explanation_table(expl: dict[str, Any]) -> pd.DataFrame:
    c = expl["contributions"]
    df = pd.DataFrame({"feature": list(c), "contribution": list(c.values())})
    df["abs"] = df["contribution"].abs()
    return df.sort_values("abs", ascending=False).drop(columns="abs").reset_index(drop=True)


def top_factors(expl: dict[str, Any], k: int = 3) -> list[str]:
    t = explanation_table(expl).head(k)
    return [f"{r.feature} ({'+' if r.contribution >= 0 else '−'}{abs(r.contribution):.3f})" for r in t.itertuples()]


def safe_float(v: Any) -> float | None:
    try:
        f = float(v)
        return None if np.isnan(f) else f
    except (TypeError, ValueError):
        return None
