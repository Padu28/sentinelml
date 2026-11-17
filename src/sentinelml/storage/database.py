"""SQLite persistence layer (prototype). Swap the DSN for PostgreSQL in deployment.

Tables
------
experiments        one row per experiment campaign
runs               one row per (policy, drift_type, seed)
batch_metrics      per-batch metrics / quality / drift for every run
events             drift, quality, degradation, retrain and promotion events
model_versions     champion / candidate registry
predictions        API prediction log
"""
from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

SCHEMA = """
CREATE TABLE IF NOT EXISTS experiments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    created_at TEXT NOT NULL,
    config_json TEXT
);
CREATE TABLE IF NOT EXISTS runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_id INTEGER REFERENCES experiments(id),
    policy TEXT NOT NULL,
    drift_type TEXT NOT NULL,
    seed INTEGER NOT NULL,
    mean_auc REAL, post_drift_auc REAL, mean_f1 REAL,
    n_retrains INTEGER, detection_delay INTEGER, false_alarms INTEGER,
    recovery_batches INTEGER, retrain_seconds REAL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS batch_metrics (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER REFERENCES runs(id),
    batch_id INTEGER, ts TEXT, n_rows INTEGER,
    accuracy REAL, f1 REAL, roc_auc REAL, rolling_auc REAL,
    quality_score REAL, max_psi REAL, n_drifted INTEGER,
    drift_flag INTEGER, retrained INTEGER, model_version TEXT,
    concept_alpha REAL, positive_rate REAL
);
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER,
    batch_id INTEGER,
    ts TEXT,
    event_type TEXT NOT NULL,
    severity TEXT NOT NULL,
    message TEXT,
    details_json TEXT
);
CREATE TABLE IF NOT EXISTS model_versions (
    version TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    algorithm TEXT, status TEXT,
    trained_on_records INTEGER,
    metrics_json TEXT, path TEXT, reason TEXT
);
CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL, model_version TEXT,
    probability REAL, prediction INTEGER, latency_ms REAL,
    payload_json TEXT
);
CREATE INDEX IF NOT EXISTS ix_bm_run ON batch_metrics(run_id);
CREATE INDEX IF NOT EXISTS ix_ev_run ON events(run_id);
"""


def utcnow() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class Database:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as con:
            con.executescript(SCHEMA)

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        con = sqlite3.connect(self.path, timeout=30)
        con.row_factory = sqlite3.Row
        try:
            yield con
            con.commit()
        finally:
            con.close()

    # ---------------------------------------------------------- generic
    def insert(self, table: str, row: dict[str, Any]) -> int:
        cols = ", ".join(row)
        qs = ", ".join("?" for _ in row)
        with self.connect() as con:
            cur = con.execute(f"INSERT INTO {table} ({cols}) VALUES ({qs})", list(row.values()))
            return int(cur.lastrowid)

    def insert_many(self, table: str, rows: list[dict[str, Any]]) -> None:
        if not rows:
            return
        cols = list(rows[0])
        qs = ", ".join("?" for _ in cols)
        with self.connect() as con:
            con.executemany(
                f"INSERT INTO {table} ({', '.join(cols)}) VALUES ({qs})",
                [[r.get(c) for c in cols] for r in rows],
            )

    def query_df(self, sql: str, params: tuple | list = ()) -> pd.DataFrame:
        with self.connect() as con:
            return pd.read_sql_query(sql, con, params=params)

    def execute(self, sql: str, params: tuple | list = ()) -> None:
        with self.connect() as con:
            con.execute(sql, params)

    # ---------------------------------------------------------- domain helpers
    def create_experiment(self, name: str, config: dict[str, Any]) -> int:
        return self.insert("experiments", {"name": name, "created_at": utcnow(), "config_json": json.dumps(config)})

    def log_event(self, event_type: str, severity: str, message: str, run_id: int | None = None,
                  batch_id: int | None = None, ts: str | None = None, details: dict | None = None) -> None:
        self.insert("events", {
            "run_id": run_id, "batch_id": batch_id, "ts": ts or utcnow(), "event_type": event_type,
            "severity": severity, "message": message, "details_json": json.dumps(details or {}, default=str),
        })

    def latest_experiment_id(self) -> int | None:
        df = self.query_df("SELECT MAX(id) AS id FROM experiments")
        v = df["id"].iloc[0]
        return None if pd.isna(v) else int(v)

    def reset(self) -> None:
        with self.connect() as con:
            for t in ("predictions", "events", "batch_metrics", "runs", "experiments", "model_versions"):
                con.execute(f"DELETE FROM {t}")
