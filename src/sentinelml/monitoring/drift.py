"""Drift detection.

Batch (reference vs current window):
    * Population Stability Index (PSI) for numeric and categorical features.
    * Two-sample Kolmogorov–Smirnov test with Bonferroni correction.

Streaming (one value at a time, e.g. per-batch error rate):
    * Page–Hinkley test for upward mean shifts.
    * ADWIN (ADaptive WINdowing, Bifet & Gavaldà 2007) — compact implementation.

No external drift library is required; Evidently/River can be swapped in.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

EPS = 1e-6


# ---------------------------------------------------------------- batch tests
def psi_numeric(reference: np.ndarray, current: np.ndarray, bins: int = 10) -> float:
    ref = np.asarray(reference, dtype=float)
    cur = np.asarray(current, dtype=float)
    ref, cur = ref[~np.isnan(ref)], cur[~np.isnan(cur)]
    if len(ref) < 2 or len(cur) < 2:
        return float("nan")
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    r = np.histogram(ref, edges)[0] / len(ref)
    c = np.histogram(cur, edges)[0] / len(cur)
    r, c = np.clip(r, EPS, None), np.clip(c, EPS, None)
    return float(np.sum((c - r) * np.log(c / r)))


def psi_categorical(reference: pd.Series, current: pd.Series) -> float:
    cats = sorted(set(reference.dropna().unique()) | set(current.dropna().unique()))
    r = reference.value_counts(normalize=True).reindex(cats, fill_value=0).to_numpy()
    c = current.value_counts(normalize=True).reindex(cats, fill_value=0).to_numpy()
    r, c = np.clip(r, EPS, None), np.clip(c, EPS, None)
    return float(np.sum((c - r) * np.log(c / r)))


@dataclass
class FeatureDrift:
    feature: str
    psi: float
    ks_stat: float | None
    p_value: float | None
    drifted: bool


@dataclass
class DriftReport:
    drift_detected: bool
    n_drifted: int
    share_drifted: float
    features: list[FeatureDrift] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["features"] = [asdict(f) for f in self.features]
        return d

    @property
    def max_psi(self) -> float:
        vals = [f.psi for f in self.features if not math.isnan(f.psi)]
        return max(vals) if vals else 0.0


def detect_window_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    numeric: list[str],
    categorical: list[str],
    psi_threshold: float = 0.2,
    ks_alpha: float = 0.01,
    min_drifted_features: int = 2,
) -> DriftReport:
    results: list[FeatureDrift] = []
    n_tests = max(len(numeric), 1)
    for col in numeric:
        r = reference[col].to_numpy(dtype=float)
        c = current[col].to_numpy(dtype=float)
        p = psi_numeric(r, c)
        rr, cc = r[~np.isnan(r)], c[~np.isnan(c)]
        ks = stats.ks_2samp(rr, cc) if len(rr) > 1 and len(cc) > 1 else None
        ks_stat = float(ks.statistic) if ks else None
        p_val = float(ks.pvalue) if ks else None
        # Require both: statistically significant AND practically large (PSI)
        drifted = bool(p >= psi_threshold and p_val is not None and p_val < ks_alpha / n_tests)
        results.append(FeatureDrift(col, round(p, 4), ks_stat, p_val, drifted))
    for col in categorical:
        p = psi_categorical(reference[col].astype(str), current[col].astype(str))
        results.append(FeatureDrift(col, round(p, 4), None, None, bool(p >= psi_threshold)))
    n_drift = sum(f.drifted for f in results)
    return DriftReport(
        drift_detected=n_drift >= min_drifted_features,
        n_drifted=n_drift,
        share_drifted=n_drift / max(len(results), 1),
        features=results,
    )


# ------------------------------------------------------------ streaming tests
class PageHinkley:
    """Detect an *increase* in the mean of a monitored signal (e.g. error rate)."""

    def __init__(self, delta: float = 0.005, threshold: float = 2.5, min_samples: int = 4) -> None:
        self.delta, self.threshold, self.min_samples = delta, threshold, min_samples
        self.reset()

    def reset(self) -> None:
        self.n = 0
        self.mean = 0.0
        self.cum = 0.0
        self.min_cum = 0.0

    def update(self, x: float) -> bool:
        self.n += 1
        self.mean += (x - self.mean) / self.n
        self.cum += x - self.mean - self.delta
        self.min_cum = min(self.min_cum, self.cum)
        if self.n >= self.min_samples and (self.cum - self.min_cum) > self.threshold * max(self.mean, 1e-3):
            self.reset()
            return True
        return False


class ADWIN:
    """Compact ADWIN: drop the oldest part of the window when two sub-windows differ."""

    def __init__(self, delta: float = 0.002, max_window: int = 2000) -> None:
        self.delta = delta
        self.window: deque[float] = deque(maxlen=max_window)

    def update(self, x: float) -> bool:
        self.window.append(float(x))
        n = len(self.window)
        if n < 10:
            return False
        arr = np.fromiter(self.window, dtype=float)
        csum = np.cumsum(arr)
        total = csum[-1]
        for split in range(5, n - 5):
            n0, n1 = split, n - split
            mu0, mu1 = csum[split - 1] / n0, (total - csum[split - 1]) / n1
            m = 1.0 / (1.0 / n0 + 1.0 / n1)
            dd = math.log(4.0 * n / self.delta)
            eps = math.sqrt(arr.var() * 2 * dd / m) + 2 * dd / (3 * m)
            if abs(mu0 - mu1) > eps:
                for _ in range(split):
                    self.window.popleft()
                return True
        return False
