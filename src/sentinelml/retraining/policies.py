"""Retraining policies compared in the experiment.

A — Static       : train once, never retrain.
B — Periodic     : retrain every ``k`` batches on a sliding window (no validation gate).
C — Drift-aware  : retrain only after a detected drift / degradation signal, validate the
                   candidate against the champion on held-out recent data, promote if better.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PolicyState:
    batches_seen: int = 0
    last_retrain_batch: int = -10_000
    episode_start: int | None = None   # first alarm batch of the current drift episode


class BasePolicy:
    name = "base"
    validates = False

    def __init__(self) -> None:
        self.state = PolicyState()

    def should_retrain(self, batch_idx: int, signals: dict) -> bool:  # pragma: no cover - interface
        raise NotImplementedError

    def on_retrain(self, batch_idx: int, promoted: bool) -> None:
        self.state.last_retrain_batch = batch_idx
        if promoted:
            self.state.episode_start = None


class StaticPolicy(BasePolicy):
    name = "static"

    def should_retrain(self, batch_idx: int, signals: dict) -> bool:
        return False


class PeriodicPolicy(BasePolicy):
    name = "periodic"

    def __init__(self, every: int = 6) -> None:
        super().__init__()
        self.every = every

    def should_retrain(self, batch_idx: int, signals: dict) -> bool:
        self.state.batches_seen += 1
        return self.state.batches_seen % self.every == 0


class DriftAwarePolicy(BasePolicy):
    name = "drift_aware"
    validates = True

    def __init__(self, cooldown: int = 4, min_episode_batches: int = 2) -> None:
        super().__init__()
        self.cooldown = cooldown
        self.min_episode_batches = min_episode_batches

    def should_retrain(self, batch_idx: int, signals: dict) -> bool:
        # Quality-faulted batches are *not* treated as drift (H3: quality ≠ drift).
        alarm = (
            (signals.get("feature_drift", False) and not signals.get("quality_fault", False))
            or signals.get("error_drift", False)
            or signals.get("degraded", False)
        )
        if alarm and self.state.episode_start is None:
            self.state.episode_start = batch_idx
        if self.state.episode_start is None:
            return False
        enough_data = batch_idx - self.state.episode_start + 1 >= self.min_episode_batches
        cooled = batch_idx - self.state.last_retrain_batch >= self.cooldown
        return bool(enough_data and cooled)


def make_policy(name: str, monitoring_cfg: dict) -> BasePolicy:
    if name == "static":
        return StaticPolicy()
    if name == "periodic":
        return PeriodicPolicy(every=monitoring_cfg["policies"]["periodic_every_batches"])
    if name == "drift_aware":
        return DriftAwarePolicy(cooldown=monitoring_cfg["drift"]["cooldown_batches"])
    raise ValueError(f"Unknown policy '{name}'")


POLICIES = ("static", "periodic", "drift_aware")
