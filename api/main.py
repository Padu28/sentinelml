"""SentinelML REST API.

Run:  uvicorn api.main:app --reload --port 8000     → docs at http://localhost:8000/docs
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Literal

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402
from fastapi import BackgroundTasks, FastAPI, HTTPException, Query  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from sentinelml import __version__  # noqa: E402
from sentinelml.common.config import get_settings  # noqa: E402
from sentinelml.common.utils import timer, to_jsonable  # noqa: E402
from sentinelml.explainability.explain import local_explanation, top_factors  # noqa: E402
from sentinelml.ingestion.loaders import load_stream  # noqa: E402
from sentinelml.retraining.registry import ModelRegistry  # noqa: E402
from sentinelml.storage.database import Database, utcnow  # noqa: E402
from sentinelml.validation.quality import clean_batch, run_quality_checks  # noqa: E402

app = FastAPI(
    title="SentinelML API",
    version=__version__,
    description="Data-quality validation, drift monitoring, model registry and explainable predictions.",
)
app.add_middleware(CORSMiddleware, allow_origins=os.getenv("CORS_ORIGINS", "*").split(","),
                   allow_methods=["*"], allow_headers=["*"])

settings = get_settings()
db = Database(settings.db_path)
registry = ModelRegistry(db, settings.models_dir)
_state: dict[str, Any] = {"experiment_running": False}


# ------------------------------------------------------------------ schemas
class Record(BaseModel):
    sensor_a: float | None = Field(None, examples=[0.4])
    sensor_b: float | None = Field(None, examples=[-0.2])
    sensor_c: float | None = Field(None, examples=[1.1])
    load_kw: float | None = Field(None, examples=[72.5])
    temp_c: float | None = Field(None, examples=[24.0])
    region: str | None = Field(None, examples=["north"])


class Batch(BaseModel):
    records: list[dict[str, Any]] = Field(..., min_length=1)


class PredictRequest(BaseModel):
    record: Record
    explain: bool = True


class PredictResponse(BaseModel):
    probability: float
    prediction: int
    model_version: str
    latency_ms: float
    explanation: dict[str, Any] | None = None
    top_factors: list[str] | None = None


class ExperimentRequest(BaseModel):
    quick: bool = True


# ------------------------------------------------------------------ helpers
def _champion():
    champ = registry.champion()
    if champ is None:
        raise HTTPException(503, "No champion model. Run `python -m sentinelml.cli pipeline` first.")
    return champ


_background_cache: dict[str, pd.DataFrame] = {}


def _background() -> pd.DataFrame:
    if "bg" not in _background_cache:
        df = load_stream("sudden", seed=0)
        _background_cache["bg"] = clean_batch(df[df["batch_id"] < 12], settings.data["schema"]).sample(500, random_state=0)
    return _background_cache["bg"]


# ------------------------------------------------------------------ routes
@app.get("/health", tags=["system"])
def health() -> dict[str, Any]:
    champ = registry.champion()
    return {"status": "ok", "version": __version__, "time": utcnow(),
            "champion": champ[0] if champ else None, "database": str(settings.db_path.name)}


@app.post("/data/validate", tags=["data"])
def validate(batch: Batch) -> dict[str, Any]:
    df = pd.DataFrame(batch.records)
    schema = settings.data["schema"]
    # target column is optional at inference time
    schema = {**schema, "columns": {k: v for k, v in schema["columns"].items() if k in df.columns or k != "label"}}
    report = run_quality_checks(df, schema)
    return to_jsonable(report.to_dict())


@app.post("/predict", response_model=PredictResponse, tags=["inference"])
def predict(req: PredictRequest) -> PredictResponse:
    version, model = _champion()
    row = pd.DataFrame([req.record.model_dump()])
    row = clean_batch(row, {"columns": {k: v for k, v in settings.data["schema"]["columns"].items() if k in row.columns}})
    with timer() as t:
        proba = float(model.predict_proba(row[settings.features])[:, 1][0])
    expl = local_explanation(model, row, _background()) if req.explain else None
    resp = PredictResponse(probability=round(proba, 4), prediction=int(proba >= 0.5), model_version=version,
                           latency_ms=round(t["ms"], 3), explanation=to_jsonable(expl) if expl else None,
                           top_factors=top_factors(expl) if expl else None)
    db.insert("predictions", {"ts": utcnow(), "model_version": version, "probability": proba,
                              "prediction": resp.prediction, "latency_ms": t["ms"],
                              "payload_json": json.dumps(req.record.model_dump())})
    return resp


@app.get("/monitoring/summary", tags=["monitoring"])
def monitoring_summary() -> dict[str, Any]:
    exp_id = db.latest_experiment_id()
    if exp_id is None:
        raise HTTPException(404, "No experiments recorded yet.")
    run = db.query_df("SELECT * FROM runs WHERE experiment_id=? AND policy='drift_aware' AND drift_type='sudden' "
                      "ORDER BY seed LIMIT 1", (exp_id,))
    if run.empty:
        raise HTTPException(404, "Showcase run missing.")
    rid = int(run["id"].iloc[0])
    last = db.query_df("SELECT * FROM batch_metrics WHERE run_id=? ORDER BY batch_id DESC LIMIT 1", (rid,))
    counts = db.query_df("SELECT event_type, COUNT(*) AS n FROM events WHERE run_id=? GROUP BY event_type", (rid,))
    champ = registry.champion()
    return to_jsonable({
        "experiment_id": exp_id,
        "champion": champ[0] if champ else None,
        "latest_batch": last.iloc[0].to_dict() if not last.empty else None,
        "run_summary": run.iloc[0].to_dict(),
        "event_counts": dict(zip(counts.event_type, counts.n)),
    })


@app.get("/events", tags=["monitoring"])
def events(limit: int = Query(50, ge=1, le=500),
           event_type: Literal["drift", "quality", "degradation", "promotion", "retrain", "retrain_rejected"] | None = None
           ) -> list[dict[str, Any]]:
    sql = "SELECT e.*, r.policy, r.drift_type, r.seed FROM events e LEFT JOIN runs r ON r.id=e.run_id"
    params: list[Any] = []
    if event_type:
        sql += " WHERE e.event_type=?"
        params.append(event_type)
    sql += " ORDER BY e.id DESC LIMIT ?"
    params.append(limit)
    df = db.query_df(sql, params)
    return to_jsonable(df.to_dict(orient="records"))


@app.get("/models", tags=["registry"])
def models() -> list[dict[str, Any]]:
    return to_jsonable(registry.list_versions())


@app.post("/experiments/run", status_code=202, tags=["experiments"])
def run_experiment(req: ExperimentRequest, bg: BackgroundTasks) -> dict[str, Any]:
    if os.getenv("APP_ENV", "development") != "development":
        raise HTTPException(403, "Experiments can only be started in the development environment.")
    if _state["experiment_running"]:
        raise HTTPException(409, "An experiment is already running.")

    def _job() -> None:
        from sentinelml.experiments.runner import run_experiments
        _state["experiment_running"] = True
        try:
            run_experiments(quick=req.quick)
        finally:
            _state["experiment_running"] = False

    bg.add_task(_job)
    return {"status": "accepted", "quick": req.quick}
