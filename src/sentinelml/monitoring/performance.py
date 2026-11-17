"""Rolling performance monitor: compares live metrics against the training baseline."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np


@dataclass
class PerformanceMonitor:
    baseline_auc: float
    window: int = 4
    tolerance: float = 0.05
    _history: deque = field(default_factory=deque, init=False, repr=False)

    def update(self, auc: float) -> dict[str, float | bool]:
        if not np.isnan(auc):
            self._history.append(auc)
            while len(self._history) > self.window:
                self._history.popleft()
        rolling = float(np.mean(self._history)) if self._history else float("nan")
        degraded = bool(not np.isnan(rolling) and rolling < self.baseline_auc - self.tolerance)
        return {"rolling_auc": rolling, "degraded": degraded}

    def rebase(self, new_baseline: float) -> None:
        self.baseline_auc = new_baseline
        self._history.clear()
