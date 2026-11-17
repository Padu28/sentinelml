"""Schema and data-quality engine.

Checks: missing columns, dtype coercion failures, nulls, duplicates,
out-of-range values, unseen categories and robust (MAD) outliers.
Returns an issue-level report plus a 0–100 quality score.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd

SEVERITY_WEIGHT = {"critical": 40.0, "high": 15.0, "medium": 6.0, "low": 2.0}


@dataclass
class QualityIssue:
    check: str
    column: str | None
    severity: str
    count: int
    rate: float
    message: str


@dataclass
class QualityReport:
    n_rows: int
    score: float
    passed: bool
    issues: list[QualityIssue] = field(default_factory=list)

    @property
    def null_rate(self) -> float:
        return sum(i.rate for i in self.issues if i.check == "nulls")

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["issues"] = [asdict(i) for i in self.issues]
        return d


def _severity(rate: float) -> str:
    if rate >= 0.10:
        return "high"
    if rate >= 0.02:
        return "medium"
    return "low"


def run_quality_checks(
    df: pd.DataFrame,
    schema: dict[str, Any],
    outlier_z: float = 6.0,
    pass_score: float = 80.0,
) -> QualityReport:
    cols: dict[str, dict[str, Any]] = schema["columns"]
    n = len(df)
    issues: list[QualityIssue] = []

    missing = [c for c in cols if c not in df.columns]
    for c in missing:
        issues.append(QualityIssue("schema", c, "critical", n, 1.0, f"Required column '{c}' is missing"))

    if n == 0:
        return QualityReport(0, 0.0, False, issues + [QualityIssue("empty", None, "critical", 0, 1.0, "Empty batch")])

    feature_cols = [c for c in cols if c in df.columns]
    dup = int(df.duplicated(subset=feature_cols).sum())
    if dup:
        r = dup / n
        issues.append(QualityIssue("duplicates", None, _severity(r), dup, r, f"{dup} duplicate rows"))

    for col, spec in cols.items():
        if col not in df.columns:
            continue
        s = df[col]
        nulls = int(s.isna().sum())
        if nulls:
            r = nulls / n
            issues.append(QualityIssue("nulls", col, _severity(r), nulls, r, f"{nulls} null values"))

        dtype = spec.get("dtype")
        if dtype in ("float", "int"):
            num = pd.to_numeric(s, errors="coerce")
            bad_type = int((num.isna() & s.notna()).sum())
            if bad_type:
                r = bad_type / n
                issues.append(QualityIssue("dtype", col, "high", bad_type, r, f"{bad_type} non-numeric values"))
            lo, hi = spec.get("min"), spec.get("max")
            if lo is not None or hi is not None:
                oor = ((num < (lo if lo is not None else -np.inf)) | (num > (hi if hi is not None else np.inf))).sum()
                if oor:
                    r = int(oor) / n
                    issues.append(
                        QualityIssue("range", col, "high" if r > 0.01 else "medium", int(oor), r,
                                     f"{int(oor)} values outside [{lo}, {hi}]")
                    )
            if "allowed" in spec:
                bad = int((~num.dropna().isin(spec["allowed"])).sum())
                if bad:
                    issues.append(QualityIssue("allowed", col, "high", bad, bad / n, f"{bad} invalid values"))
            clean = num.dropna()
            if len(clean) > 20:
                med = clean.median()
                mad = (clean - med).abs().median() * 1.4826 or 1e-9
                out = int(((clean - med).abs() / mad > outlier_z).sum())
                if out:
                    r = out / n
                    issues.append(QualityIssue("outliers", col, "low", out, r, f"{out} robust-z outliers"))
        elif dtype == "category" and "allowed" in spec:
            unseen = s.dropna()[~s.dropna().isin(spec["allowed"])]
            if len(unseen):
                r = len(unseen) / n
                vals = ", ".join(map(str, sorted(unseen.unique())[:5]))
                issues.append(QualityIssue("category", col, "high", len(unseen), r, f"Unseen categories: {vals}"))
        elif dtype == "datetime":
            ts = pd.to_datetime(s, errors="coerce")
            bad = int(ts.isna().sum() - nulls)
            if bad:
                issues.append(QualityIssue("dtype", col, "high", bad, bad / n, f"{bad} unparseable timestamps"))
            if not ts.dropna().is_monotonic_increasing:
                issues.append(QualityIssue("ordering", col, "low", 1, 0.0, "Timestamps not monotonic"))

    penalty = sum(SEVERITY_WEIGHT[i.severity] * (1 + min(i.rate, 1.0)) for i in issues)
    score = float(max(0.0, 100.0 - penalty))
    critical = any(i.severity == "critical" for i in issues)
    return QualityReport(n_rows=n, score=round(score, 2), passed=(score >= pass_score and not critical), issues=issues)


def clean_batch(df: pd.DataFrame, schema: dict[str, Any]) -> pd.DataFrame:
    """Make a batch safe for inference: drop dups, null out impossible values.

    Remaining NaNs are handled by the model pipeline's imputers; unseen
    categories are handled by ``OneHotEncoder(handle_unknown='ignore')``.
    """
    out = df.drop_duplicates(subset=[c for c in schema["columns"] if c in df.columns]).copy()
    for col, spec in schema["columns"].items():
        if col not in out.columns or spec.get("dtype") not in ("float", "int"):
            continue
        num = pd.to_numeric(out[col], errors="coerce")
        lo, hi = spec.get("min", -np.inf), spec.get("max", np.inf)
        out[col] = num.where((num >= lo) & (num <= hi))
    return out
