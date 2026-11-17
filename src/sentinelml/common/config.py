"""Configuration loading and project paths."""
from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(os.getenv("SENTINELML_ROOT", Path(__file__).resolve().parents[3]))
CONFIG_DIR = PROJECT_ROOT / "configs"


def _load_yaml(name: str) -> dict[str, Any]:
    with open(CONFIG_DIR / name, encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


@dataclass(frozen=True)
class Settings:
    data: dict[str, Any]
    model: dict[str, Any]
    monitoring: dict[str, Any]

    @property
    def db_path(self) -> Path:
        override = os.getenv("SENTINELML_DB")
        return Path(override) if override else PROJECT_ROOT / self.monitoring["storage"]["database"]

    @property
    def reports_dir(self) -> Path:
        return PROJECT_ROOT / self.monitoring["storage"]["reports_dir"]

    @property
    def models_dir(self) -> Path:
        return PROJECT_ROOT / self.model["registry"]["dir"]

    @property
    def numeric_features(self) -> list[str]:
        return list(self.data["schema"]["numeric_features"])

    @property
    def categorical_features(self) -> list[str]:
        return list(self.data["schema"]["categorical_features"])

    @property
    def features(self) -> list[str]:
        return self.numeric_features + self.categorical_features

    @property
    def target(self) -> str:
        return self.data["stream"]["target_col"]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings(
        data=_load_yaml("data.yaml"),
        model=_load_yaml("model.yaml"),
        monitoring=_load_yaml("monitoring.yaml"),
    )
