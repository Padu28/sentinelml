"""Data loaders and a timestamp-ordered stream replayer."""
from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from sentinelml.common.config import get_settings
from sentinelml.ingestion.synthetic import StreamSpec, generate_stream


def load_csv(path: str | Path, timestamp_col: str = "timestamp") -> pd.DataFrame:
    df = pd.read_csv(path)
    if timestamp_col in df.columns:
        df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors="coerce")
        df = df.sort_values(timestamp_col, kind="stable").reset_index(drop=True)
    return df


def load_stream(drift_type: str = "sudden", seed: int = 42) -> pd.DataFrame:
    """Load the configured data source as a chronologically ordered frame."""
    s = get_settings()
    cfg = s.data
    if cfg["source"] == "csv":
        df = load_csv(cfg["csv_path"], cfg["stream"]["timestamp_col"])
        df["batch_id"] = df.index // cfg["stream"]["batch_size"]
        return df
    syn = cfg["synthetic"]
    spec = StreamSpec(
        n_records=syn["n_records"],
        drift_type=drift_type,
        drift_start=syn["drift_start"],
        drift_width=syn["drift_width"],
        recurring_period=syn["recurring_period"],
        batch_size=cfg["stream"]["batch_size"],
        quality_fault_batches=list(syn.get("quality_fault_batches", [])),
        seed=seed,
    )
    return generate_stream(spec)


@dataclass
class StreamReplayer:
    """Replay a frame batch-by-batch in timestamp order (simulated live ingestion)."""

    frame: pd.DataFrame
    batch_col: str = "batch_id"

    def __iter__(self) -> Iterator[tuple[int, pd.DataFrame]]:
        for batch_id, batch in self.frame.groupby(self.batch_col, sort=True):
            yield int(batch_id), batch.reset_index(drop=True)

    def __len__(self) -> int:
        return int(self.frame[self.batch_col].nunique())
