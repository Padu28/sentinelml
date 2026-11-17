"""Controlled experiment: Static vs Periodic vs Drift-aware retraining.

For every (drift_type × seed) an identical stream is replayed through each policy
with the same splits and model family. Results are persisted to SQLite and
summarised (mean ± std, 95% CI) in ``reports/``.
"""
from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from sentinelml.common.config import get_settings
from sentinelml.common.logging import get_logger
from sentinelml.common.utils import dump_json, to_jsonable
from sentinelml.ingestion.loaders import load_stream
from sentinelml.retraining.policies import POLICIES, make_policy
from sentinelml.retraining.registry import ModelRegistry
from sentinelml.retraining.simulation import RunResult, simulate
from sentinelml.storage.database import Database, utcnow

log = get_logger("sentinelml.experiments")

SUMMARY_METRICS = ["mean_auc", "post_drift_auc", "mean_f1", "n_retrains", "detection_delay",
                   "false_alarms", "recovery_batches", "retrain_seconds"]


def _persist_run(db: Database, exp_id: int, r: RunResult) -> int:
    s = r.summary
    run_id = db.insert("runs", {
        "experiment_id": exp_id, "policy": r.policy, "drift_type": r.drift_type, "seed": r.seed,
        "mean_auc": s["mean_auc"], "post_drift_auc": s["post_drift_auc"], "mean_f1": s["mean_f1"],
        "n_retrains": s["n_retrains"], "detection_delay": s["detection_delay"],
        "false_alarms": s["false_alarms"],
        "recovery_batches": None if np.isnan(s["recovery_batches"]) else s["recovery_batches"],
        "retrain_seconds": s["retrain_seconds"], "created_at": utcnow(),
    })
    db.insert_many("batch_metrics", [{"run_id": run_id, **b} for b in r.batches])
    db.insert_many("events", [{
        "run_id": run_id, "batch_id": e["batch_id"], "ts": e["ts"], "event_type": e["event_type"],
        "severity": e["severity"], "message": e["message"], "details_json": json.dumps(to_jsonable(e["details"])),
    } for e in r.events])
    return run_id


def ci95(x: pd.Series) -> float:
    x = x.dropna()
    if len(x) < 2:
        return float("nan")
    return float(stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)))


def aggregate(runs: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (dt, pol), g in runs.groupby(["drift_type", "policy"], sort=False):
        row: dict[str, Any] = {"drift_type": dt, "policy": pol, "n_runs": len(g)}
        for m in SUMMARY_METRICS:
            col = pd.to_numeric(g[m], errors="coerce")
            row[f"{m}_mean"] = float(col.mean()) if col.notna().any() else None
            row[f"{m}_std"] = float(col.std(ddof=1)) if col.notna().sum() > 1 else None
            row[f"{m}_ci95"] = ci95(col)
        rows.append(row)
    return pd.DataFrame(rows)


def run_experiments(quick: bool = False, reset: bool = True) -> dict[str, Any]:
    s = get_settings()
    db = Database(s.db_path)
    if reset:
        db.reset()
    seeds = s.monitoring["experiments"]["seeds"][: 2 if quick else None]
    drift_types = s.monitoring["experiments"]["drift_types"]
    exp_id = db.create_experiment("policy_comparison", {"seeds": seeds, "drift_types": drift_types,
                                                        "policies": list(POLICIES), "model": s.model["algorithm"]})
    registry = ModelRegistry(db, s.models_dir)
    for old in s.models_dir.glob("model_v*.joblib"):
        old.unlink()

    summaries: list[dict[str, Any]] = []
    for dt in drift_types:
        for seed in seeds:
            stream = load_stream(dt, seed)
            for pol in POLICIES:
                # The showcase run (sudden drift, first seed, drift-aware) also drives the live model registry.
                use_reg = registry if (dt == "sudden" and seed == seeds[0] and pol == "drift_aware") else None
                res = simulate(stream, make_policy(pol, s.monitoring), dt, seed, registry=use_reg)
                run_id = _persist_run(db, exp_id, res)
                summaries.append({**res.summary, "run_id": run_id, "showcase": use_reg is not None})
                log.info("%-11s | %-11s | seed %d | post-drift AUC %.3f | retrains %d | delay %s",
                         dt, pol, seed, res.summary["post_drift_auc"], res.summary["n_retrains"],
                         res.summary["detection_delay"])

    runs = pd.DataFrame(summaries)
    agg = aggregate(runs)
    out = {"experiment_id": exp_id, "created_at": utcnow(), "seeds": seeds, "drift_types": drift_types,
           "runs": to_jsonable(runs.to_dict(orient="records")), "aggregate": to_jsonable(agg.to_dict(orient="records"))}
    dump_json(out, s.reports_dir / "experiment_summary.json")
    write_markdown_report(agg, s.reports_dir / "experiment_summary.md", seeds)
    try:
        from sentinelml.experiments.figures import make_figures
        make_figures(db, runs, agg, s.reports_dir / "figures")
    except Exception as exc:  # figures are optional (matplotlib)
        log.warning("Figure generation skipped: %s", exc)
    return out


def _fmt(mean: Any, ci: Any, digits: int = 3) -> str:
    if mean is None or (isinstance(mean, float) and np.isnan(mean)):
        return "—"
    if ci is None or (isinstance(ci, float) and np.isnan(ci)):
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f} ± {ci:.{digits}f}"


def write_markdown_report(agg: pd.DataFrame, path, seeds: list[int]) -> None:
    lines = [
        "# SentinelML — Experiment Summary",
        "",
        f"*Generated {utcnow()} · {len(seeds)} seeds per condition · values are mean ± 95% CI*",
        "",
        "> Data: synthetic controlled stream with known drift ground truth. Results characterise",
        "> detector/policy behaviour under controlled conditions and are **not** claims about any real system.",
        "",
        "| Drift | Policy | Post-drift AUC | Mean AUC | Retrains | Detection delay (records) | False alarms | Recovery (batches) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in agg.to_dict(orient="records"):
        lines.append(
            f"| {r['drift_type']} | {r['policy']} | {_fmt(r['post_drift_auc_mean'], r['post_drift_auc_ci95'])} | "
            f"{_fmt(r['mean_auc_mean'], r['mean_auc_ci95'])} | {_fmt(r['n_retrains_mean'], r['n_retrains_ci95'], 1)} | "
            f"{_fmt(r['detection_delay_mean'], r['detection_delay_ci95'], 0)} | {_fmt(r['false_alarms_mean'], None, 1)} | "
            f"{_fmt(r['recovery_batches_mean'], r['recovery_batches_ci95'], 1)} |"
        )
    lines += [
        "",
        "## How to read this",
        "- **H1** (faster recovery): compare *Post-drift AUC* and *Recovery* of `drift_aware` vs `static`.",
        "- **H2** (fewer retrains): compare *Retrains* of `drift_aware` vs `periodic` at similar AUC.",
        "- **H3** (quality ≠ drift): quality-faulted batches are gated and do not count as drift alarms.",
        "",
        "## Limitations",
        "- Single model family (configurable in `configs/model.yaml`); thresholds were set a-priori, not tuned on test streams.",
        "- Labels are assumed to arrive one batch after prediction; real label delay is usually longer.",
        "- Synthetic drift is cleaner than real-world drift; validate on a real dataset (NAB, UCI Electricity) before drawing conclusions.",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
