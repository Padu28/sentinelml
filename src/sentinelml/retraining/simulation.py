"""Replay a stream through a retraining policy and record everything.

The simulation is deliberately explicit so every number on the dashboard can be
traced back to a logged batch: quality → inference → (labels arrive) →
metrics → drift tests → policy decision → candidate validation → promotion.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from sentinelml.common.config import get_settings
from sentinelml.models.train import classification_metrics, latency_profile, predict_proba, train_model
from sentinelml.monitoring.drift import PageHinkley, detect_window_drift
from sentinelml.monitoring.performance import PerformanceMonitor
from sentinelml.retraining.policies import BasePolicy
from sentinelml.retraining.registry import ModelRegistry, promotion_rule
from sentinelml.validation.quality import clean_batch, run_quality_checks


@dataclass
class RunResult:
    policy: str
    drift_type: str
    seed: int
    batches: list[dict[str, Any]] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)


def _rolling(values: list[float], w: int) -> list[float]:
    return pd.Series(values).rolling(w, min_periods=1).mean().tolist()


def recovery_batches(auc: list[float], drift_idx: int, pre_level: float, frac: float, w: int = 3) -> float:
    """Batches from drift onset until rolling AUC returns to ``frac`` × pre-drift level."""
    roll = _rolling(auc, w)[drift_idx:]
    target = frac * pre_level
    below = [i for i, v in enumerate(roll) if v < target]
    if not below:
        return 0.0
    for i in range(below[0], len(roll)):
        if roll[i] >= target:
            return float(i)
    return float("nan")


def simulate(
    stream: pd.DataFrame,
    policy: BasePolicy,
    drift_type: str,
    seed: int,
    registry: ModelRegistry | None = None,
) -> RunResult:
    s = get_settings()
    mon, schema = s.monitoring, s.data["schema"]
    bs = s.data["stream"]["batch_size"]
    n_train_batches = s.data["stream"]["train_records"] // bs
    drift_batch = s.data["synthetic"]["drift_start"] // bs
    window_batches = max(2, mon["policies"]["retrain_window_records"] // bs)
    algo = s.model["algorithm"]

    result = RunResult(policy.name, drift_type, seed)

    # ---- initial champion: chronological 80/20 split of the training period
    train_period = clean_batch(stream[stream["batch_id"] < n_train_batches], schema)
    cut = int(len(train_period) * 0.8)
    model = train_model(train_period.iloc[:cut], algo, seed)
    val_proba = predict_proba(model, train_period.iloc[cut:])
    baseline = classification_metrics(train_period.iloc[cut:][s.target].to_numpy(), val_proba)
    version = "v-init"
    if registry is not None:
        lat = latency_profile(model, train_period.iloc[cut:])
        version = registry.register(model, algo, {**baseline, **lat}, len(train_period.iloc[:cut]),
                                    status="champion", reason="initial training (chronological split)")

    reference = train_period[s.features]
    ph = PageHinkley(**mon["drift"]["page_hinkley"])
    perf = PerformanceMonitor(baseline["roc_auc"], mon["performance"]["rolling_window_batches"],
                              mon["performance"]["degradation_tolerance"])
    labelled: list[pd.DataFrame] = []
    n_retrains = n_promotions = 0
    retrain_seconds = 0.0
    drift_flags: list[tuple[int, bool]] = []

    stream_batches = [(b, g) for b, g in stream.groupby("batch_id", sort=True) if b >= n_train_batches]
    for b, raw in stream_batches:
        ts = str(raw["timestamp"].iloc[0])
        q = run_quality_checks(raw, schema)
        quality_fault = not q.passed
        if quality_fault:
            result.events.append(dict(batch_id=b, ts=ts, event_type="quality", severity="high",
                                      message=f"Quality score {q.score:.0f} — {len(q.issues)} issues; batch cleaned before inference",
                                      details={"issues": [i.message for i in q.issues][:6]}))
        batch = clean_batch(raw, schema)

        proba = predict_proba(model, batch)
        m = classification_metrics(batch[s.target].to_numpy(), proba)
        labelled.append(batch)

        drift = detect_window_drift(reference, batch[s.features], s.numeric_features, s.categorical_features,
                                    mon["drift"]["psi_threshold"], mon["drift"]["ks_alpha"],
                                    mon["drift"]["min_drifted_features"])
        err_alarm = ph.update(1 - m["accuracy"])
        p = perf.update(m["roc_auc"])
        flagged = (drift.drift_detected and not quality_fault) or err_alarm
        drift_flags.append((b, flagged))
        if flagged:
            top = sorted(drift.features, key=lambda f: -f.psi)[:3]
            result.events.append(dict(batch_id=b, ts=ts, event_type="drift", severity="medium",
                                      message=("Error-rate shift (Page-Hinkley)" if err_alarm and not drift.drift_detected
                                               else f"{drift.n_drifted} features drifted; top: " +
                                               ", ".join(f"{f.feature} (PSI {f.psi:.2f})" for f in top)),
                                      details=drift.to_dict()))
        if p["degraded"]:
            result.events.append(dict(batch_id=b, ts=ts, event_type="degradation", severity="high",
                                      message=f"Rolling AUC {p['rolling_auc']:.3f} below baseline {perf.baseline_auc:.3f}",
                                      details=p))

        signals = {"feature_drift": drift.drift_detected, "error_drift": err_alarm,
                   "degraded": p["degraded"], "quality_fault": quality_fault}
        retrained = False
        if policy.should_retrain(b, signals):
            t0 = time.perf_counter()
            if policy.validates:
                start = policy.state.episode_start if policy.state.episode_start is not None else b
                n_use = min(window_batches, b - start + 1)
                train_frames, holdout = labelled[-n_use:-1] or labelled[-2:-1], labelled[-1]
                candidate = train_model(pd.concat(train_frames), algo, seed)
                champ_auc = classification_metrics(holdout[s.target].to_numpy(), predict_proba(model, holdout))["roc_auc"]
                cand_auc = classification_metrics(holdout[s.target].to_numpy(), predict_proba(candidate, holdout))["roc_auc"]
                lat = latency_profile(candidate, holdout, n_trials=10)
                decision = promotion_rule(champ_auc, cand_auc, s.model["registry"]["min_improvement"],
                                          lat["p95_ms"], s.model["registry"]["max_latency_ms_p95"])
                n_retrains += 1
                if decision.promote:
                    # refit on the full window (incl. hold-out) once validated
                    model = train_model(pd.concat(labelled[-n_use:]), algo, seed)
                    n_promotions += 1
                    retrained = True
                    reference = pd.concat(labelled[-n_use:])[s.features]
                    ph.reset()
                    perf.rebase(cand_auc)
                    if registry is not None:
                        version = registry.register(model, algo, {"roc_auc": cand_auc, **lat}, len(reference),
                                                    status="champion", reason=decision.reason)
                    else:
                        version = f"r{n_promotions}"
                elif registry is not None:
                    registry.register(candidate, algo, {"roc_auc": cand_auc, **lat}, len(pd.concat(train_frames)),
                                      status="rejected", reason=decision.reason)
                result.events.append(dict(batch_id=b, ts=ts, event_type="promotion" if decision.promote else "retrain_rejected",
                                          severity="info", message=f"Candidate {'promoted' if decision.promote else 'rejected'}: {decision.reason}",
                                          details={"champion_auc": champ_auc, "candidate_auc": cand_auc}))
                policy.on_retrain(b, decision.promote)
            else:
                window = pd.concat(labelled[-window_batches:])
                model = train_model(window, algo, seed)
                reference = window[s.features]
                n_retrains += 1
                n_promotions += 1
                retrained = True
                version = f"p{n_promotions}"
                ph.reset()
                result.events.append(dict(batch_id=b, ts=ts, event_type="retrain", severity="info",
                                          message=f"Scheduled retrain on last {len(window)} records", details={}))
                policy.on_retrain(b, True)
            retrain_seconds += time.perf_counter() - t0

        result.batches.append(dict(
            batch_id=int(b), ts=ts, n_rows=len(batch), accuracy=m["accuracy"], f1=m["f1"], roc_auc=m["roc_auc"],
            rolling_auc=p["rolling_auc"], quality_score=q.score, max_psi=drift.max_psi, n_drifted=drift.n_drifted,
            drift_flag=int(flagged), retrained=int(retrained), model_version=version,
            concept_alpha=float(raw["concept_alpha"].mean()) if "concept_alpha" in raw else None,
            positive_rate=float(batch[s.target].mean()),
        ))

    # ---------------------------------------------------------------- summary
    auc = [r["roc_auc"] for r in result.batches]
    ids = [r["batch_id"] for r in result.batches]
    d_idx = next((i for i, bid in enumerate(ids) if bid >= drift_batch), len(ids))
    pre = float(np.nanmean(auc[:d_idx])) if d_idx else baseline["roc_auc"]
    flags_after = [bid for bid, f in drift_flags if f and bid >= drift_batch]
    flags_before = [bid for bid, f in drift_flags if f and bid < drift_batch]
    result.summary = dict(
        policy=policy.name, drift_type=drift_type, seed=seed,
        baseline_auc=baseline["roc_auc"], pre_drift_auc=pre,
        mean_auc=float(np.nanmean(auc)),
        post_drift_auc=float(np.nanmean(auc[d_idx:])) if d_idx < len(auc) else float("nan"),
        mean_f1=float(np.nanmean([r["f1"] for r in result.batches])),
        n_retrains=n_retrains, n_promotions=n_promotions,
        detection_delay=(flags_after[0] - drift_batch) * bs if flags_after and drift_type != "none" else None,
        false_alarms=len(flags_before),
        recovery_batches=recovery_batches(auc, d_idx, pre, mon["experiments"]["recovery_fraction"]),
        retrain_seconds=retrain_seconds,
    )
    return result
