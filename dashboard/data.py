"""Cached data access for the dashboard. Every number shown comes from the
experiment database or the model registry — never from hard-coded mocks."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from sentinelml.common.config import get_settings  # noqa: E402
from sentinelml.retraining.registry import ModelRegistry  # noqa: E402
from sentinelml.storage.database import Database  # noqa: E402

settings = get_settings()


def db() -> Database:
    return Database(settings.db_path)


def ensure_pipeline() -> None:
    """Bootstrap the experiment database on first launch."""
    d = db()
    if d.latest_experiment_id() is None:
        with st.status("First launch — running the quick experiment pipeline (≈40 s)…", expanded=True) as s:
            from sentinelml.cli import generate_sample
            from sentinelml.experiments.runner import run_experiments

            st.write("Generating synthetic controlled stream…")
            generate_sample()
            st.write("Replaying 3 retraining policies × 4 drift types × 2 seeds…")
            run_experiments(quick=True)
            s.update(label="Pipeline complete", state="complete")
        st.cache_data.clear()


@st.cache_data(ttl=60)
def runs() -> pd.DataFrame:
    d = db()
    exp = d.latest_experiment_id()
    return d.query_df("SELECT * FROM runs WHERE experiment_id=? ORDER BY drift_type, policy, seed", (exp,))


@st.cache_data(ttl=60)
def batch_metrics(run_id: int) -> pd.DataFrame:
    return db().query_df("SELECT * FROM batch_metrics WHERE run_id=? ORDER BY batch_id", (int(run_id),))


@st.cache_data(ttl=60)
def all_batch_metrics() -> pd.DataFrame:
    d = db()
    exp = d.latest_experiment_id()
    return d.query_df(
        "SELECT b.*, r.policy, r.drift_type, r.seed FROM batch_metrics b JOIN runs r ON r.id=b.run_id "
        "WHERE r.experiment_id=?", (exp,))


@st.cache_data(ttl=60)
def events(run_id: int | None = None, event_type: str | None = None) -> pd.DataFrame:
    sql = ("SELECT e.*, r.policy, r.drift_type, r.seed FROM events e LEFT JOIN runs r ON r.id=e.run_id WHERE 1=1")
    params: list = []
    if run_id is not None:
        sql += " AND e.run_id=?"
        params.append(int(run_id))
    if event_type:
        sql += " AND e.event_type=?"
        params.append(event_type)
    df = db().query_df(sql + " ORDER BY e.id DESC", params)
    df["details"] = df["details_json"].apply(lambda s: json.loads(s) if s else {})
    return df


@st.cache_data(ttl=60)
def model_versions() -> pd.DataFrame:
    rows = ModelRegistry(db(), settings.models_dir).list_versions()
    return pd.DataFrame(rows)


@st.cache_data
def summary_json() -> dict:
    p = settings.reports_dir / "experiment_summary.json"
    return json.loads(p.read_text()) if p.exists() else {}


def showcase_run_id() -> int | None:
    r = runs()
    sel = r[(r.policy == "drift_aware") & (r.drift_type == "sudden")].sort_values("seed")
    return int(sel["id"].iloc[0]) if not sel.empty else None


def run_id_for(drift_type: str, policy: str, seed: int) -> int | None:
    r = runs()
    sel = r[(r.policy == policy) & (r.drift_type == drift_type) & (r.seed == seed)]
    return int(sel["id"].iloc[0]) if not sel.empty else None


def drift_batch() -> int:
    return settings.data["synthetic"]["drift_start"] // settings.data["stream"]["batch_size"]
