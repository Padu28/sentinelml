"""Synthetic controlled stream with *known* drift ground truth.

The generator models an industrial asset-failure task: given sensor readings,
load, temperature and region, predict whether a failure flag is raised.
Four drift types can be injected after ``drift_start``:

* ``sudden``      – concept and covariate change at a single point.
* ``gradual``     – probability of sampling from the new concept rises linearly.
* ``incremental`` – coefficients interpolate smoothly from old to new.
* ``recurring``   – the old and new concept alternate every ``period`` records.

Data-quality faults (nulls, out-of-range values, unseen categories, duplicates)
can be injected into specific batches so that *quality faults* can be evaluated
separately from *genuine distribution drift*.

All output is SYNTHETIC and labelled as such (``is_synthetic`` column).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

DRIFT_TYPES = ("none", "sudden", "gradual", "incremental", "recurring")
REGIONS = np.array(["north", "south", "east", "west"])

# Concept parameters: logit = X @ w + region_effect + bias
_W_OLD = np.array([1.6, -1.1, 0.7, 0.012, 0.03])
_W_NEW = np.array([-0.4, -1.3, 1.8, 0.020, -0.02])
_REGION_OLD = {"north": 0.3, "south": -0.2, "east": 0.0, "west": -0.1}
_REGION_NEW = {"north": -0.5, "south": 0.4, "east": 0.2, "west": -0.1}


@dataclass
class StreamSpec:
    n_records: int = 12_000
    drift_type: str = "sudden"
    drift_start: int = 6_000
    drift_width: int = 1_500
    recurring_period: int = 1_500
    batch_size: int = 250
    quality_fault_batches: list[int] = field(default_factory=list)
    seed: int = 42

    def __post_init__(self) -> None:
        if self.drift_type not in DRIFT_TYPES:
            raise ValueError(f"drift_type must be one of {DRIFT_TYPES}")


def _new_concept_weight(idx: np.ndarray, spec: StreamSpec, rng: np.random.Generator) -> np.ndarray:
    """Return alpha ∈ [0, 1] per record: 0 = old concept, 1 = new concept."""
    t = idx - spec.drift_start
    if spec.drift_type == "none":
        return np.zeros_like(idx, dtype=float)
    if spec.drift_type == "sudden":
        return (t >= 0).astype(float)
    ramp = np.clip(t / max(spec.drift_width, 1), 0.0, 1.0)
    if spec.drift_type == "incremental":
        return ramp
    if spec.drift_type == "gradual":
        return (rng.random(len(idx)) < ramp).astype(float)
    # recurring: alternate concept blocks after drift_start
    block = np.floor(np.maximum(t, 0) / spec.recurring_period).astype(int)
    return ((t >= 0) & (block % 2 == 0)).astype(float)


def generate_stream(spec: StreamSpec) -> pd.DataFrame:
    rng = np.random.default_rng(spec.seed)
    n = spec.n_records
    idx = np.arange(n)
    alpha = _new_concept_weight(idx, spec, rng)

    # Covariates (with covariate shift proportional to alpha)
    cov = np.array([[1.0, 0.3, 0.0], [0.3, 1.0, 0.2], [0.0, 0.2, 1.0]])
    sensors = rng.multivariate_normal(np.zeros(3), cov, size=n)
    sensors[:, 0] += 1.2 * alpha            # sensor_a mean shift
    sensors[:, 2] *= 1.0 + 0.6 * alpha       # sensor_c variance inflation
    load_kw = rng.lognormal(mean=4.2 + 0.25 * alpha, sigma=0.35, size=n)
    temp_c = 22 + 6 * np.sin(2 * np.pi * idx / 2400) + rng.normal(0, 3, n) + 4 * alpha
    region_p_old = np.array([0.4, 0.3, 0.2, 0.1])
    region_p_new = np.array([0.15, 0.25, 0.25, 0.35])
    u = rng.random(n)
    region = np.where(
        alpha > 0.5,
        REGIONS[np.searchsorted(np.cumsum(region_p_new), u)],
        REGIONS[np.searchsorted(np.cumsum(region_p_old), u)],
    )

    X = np.column_stack([sensors, load_kw, temp_c])
    centred = X - np.array([0, 0, 0, 70, 22])
    w = (1 - alpha)[:, None] * _W_OLD + alpha[:, None] * _W_NEW
    reg_eff = np.array(
        [(1 - a) * _REGION_OLD[r] + a * _REGION_NEW[r] for a, r in zip(alpha, region)]
    )
    logit = 1.8 * (np.sum(centred * w, axis=1) + reg_eff) - 0.2
    label = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)

    df = pd.DataFrame(
        {
            "timestamp": pd.Timestamp("2026-01-01") + pd.to_timedelta(idx * 60, unit="s"),
            "sensor_a": sensors[:, 0],
            "sensor_b": sensors[:, 1],
            "sensor_c": sensors[:, 2],
            "load_kw": load_kw,
            "temp_c": temp_c,
            "region": region,
            "label": label,
            "concept_alpha": alpha,          # ground truth, never used as a feature
            "is_synthetic": True,
        }
    )
    df["batch_id"] = idx // spec.batch_size
    df["quality_fault"] = False
    for b in spec.quality_fault_batches:
        df = inject_quality_faults(df, b, rng)
    return df.reset_index(drop=True)


def inject_quality_faults(df: pd.DataFrame, batch_id: int, rng: np.random.Generator) -> pd.DataFrame:
    """Inject nulls, impossible values, an unseen category and duplicates into one batch."""
    df = df.copy()
    mask = df["batch_id"] == batch_id
    rows = df.index[mask].to_numpy()
    if len(rows) == 0:
        return df
    pick = lambda frac: rng.choice(rows, size=max(1, int(len(rows) * frac)), replace=False)  # noqa: E731
    df.loc[pick(0.15), "sensor_b"] = np.nan
    df.loc[pick(0.05), "temp_c"] = 999.0
    df.loc[pick(0.04), "load_kw"] = -50.0
    df["region"] = df["region"].astype(object)
    df.loc[pick(0.06), "region"] = "unknown_site"
    df.loc[rows, "quality_fault"] = True
    dups = df.loc[pick(0.05)]
    return pd.concat([df, dups]).sort_values("timestamp", kind="stable")
