"""Champion / candidate model registry backed by SQLite + joblib artefacts."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
from sklearn.pipeline import Pipeline

from sentinelml.storage.database import Database, utcnow


@dataclass
class PromotionDecision:
    promote: bool
    reason: str
    champion_score: float
    candidate_score: float


def promotion_rule(champion_score: float, candidate_score: float, min_improvement: float,
                   candidate_p95_ms: float, max_p95_ms: float) -> PromotionDecision:
    if candidate_p95_ms > max_p95_ms:
        return PromotionDecision(False, f"latency p95 {candidate_p95_ms:.1f}ms > budget {max_p95_ms}ms",
                                 champion_score, candidate_score)
    if candidate_score >= champion_score + min_improvement:
        return PromotionDecision(True, f"candidate +{candidate_score - champion_score:.3f} over champion",
                                 champion_score, candidate_score)
    return PromotionDecision(False, f"improvement {candidate_score - champion_score:+.3f} < {min_improvement}",
                             champion_score, candidate_score)


class ModelRegistry:
    def __init__(self, db: Database, models_dir: Path) -> None:
        self.db = db
        self.dir = Path(models_dir)
        self.dir.mkdir(parents=True, exist_ok=True)

    def _next_version(self) -> str:
        df = self.db.query_df("SELECT COUNT(*) AS n FROM model_versions")
        return f"v{int(df['n'].iloc[0]) + 1:03d}"

    def register(self, model: Pipeline, algorithm: str, metrics: dict[str, Any], trained_on: int,
                 status: str = "candidate", reason: str = "") -> str:
        version = self._next_version()
        path = self.dir / f"model_{version}.joblib"
        joblib.dump(model, path)
        self.db.insert("model_versions", {
            "version": version, "created_at": utcnow(), "algorithm": algorithm, "status": status,
            "trained_on_records": trained_on, "metrics_json": json.dumps(metrics, default=float),
            "path": str(path), "reason": reason,
        })
        if status == "champion":
            self.promote(version, reason)
        return version

    def promote(self, version: str, reason: str = "") -> None:
        self.db.execute("UPDATE model_versions SET status='archived' WHERE status='champion' AND version != ?", (version,))
        self.db.execute("UPDATE model_versions SET status='champion', reason=? WHERE version=?", (reason, version))

    def champion(self) -> tuple[str, Pipeline] | None:
        df = self.db.query_df("SELECT version, path FROM model_versions WHERE status='champion' ORDER BY created_at DESC LIMIT 1")
        if df.empty or not Path(df["path"].iloc[0]).exists():
            return None
        return df["version"].iloc[0], joblib.load(df["path"].iloc[0])

    def list_versions(self) -> list[dict[str, Any]]:
        df = self.db.query_df("SELECT * FROM model_versions ORDER BY created_at, version")
        out = df.to_dict(orient="records")
        for r in out:
            r["metrics"] = json.loads(r.pop("metrics_json") or "{}")
        return out
