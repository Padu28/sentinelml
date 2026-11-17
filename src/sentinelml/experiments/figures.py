"""Static report figures (matplotlib) saved to reports/figures/."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from sentinelml.storage.database import Database  # noqa: E402

COLORS = {"static": "#94a3b8", "periodic": "#f59e0b", "drift_aware": "#6366f1"}


def make_figures(db: Database, runs: pd.DataFrame, agg: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"figure.dpi": 130, "axes.spines.top": False, "axes.spines.right": False})

    # 1) AUC over time for the showcase scenario
    first = runs[(runs.drift_type == "sudden") & (runs.seed == runs.seed.min())]
    fig, ax = plt.subplots(figsize=(9, 4))
    for r in first.itertuples():
        bm = db.query_df("SELECT batch_id, roc_auc FROM batch_metrics WHERE run_id=? ORDER BY batch_id", (int(r.run_id),))
        ax.plot(bm.batch_id, bm.roc_auc.rolling(2, min_periods=1).mean(), label=r.policy, color=COLORS[r.policy], lw=2)
    ax.axvline(24, ls="--", color="#ef4444", lw=1)
    ax.text(24.3, ax.get_ylim()[0] + 0.02, "drift onset", color="#ef4444", fontsize=8)
    ax.set(title="ROC-AUC per batch — sudden drift (seed 0)", xlabel="batch", ylabel="ROC-AUC")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "auc_over_time_sudden.png")
    plt.close(fig)

    # 2) Post-drift AUC by drift type and policy
    piv = agg.pivot(index="drift_type", columns="policy", values="post_drift_auc_mean")
    err = agg.pivot(index="drift_type", columns="policy", values="post_drift_auc_ci95")
    fig, ax = plt.subplots(figsize=(8, 4))
    piv.plot.bar(ax=ax, yerr=err, color=[COLORS[c] for c in piv.columns], capsize=3, rot=0)
    ax.set(title="Post-drift ROC-AUC (mean ± 95% CI)", xlabel="", ylabel="ROC-AUC", ylim=(0.4, 1.0))
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "post_drift_auc_by_policy.png")
    plt.close(fig)

    # 3) Retrain counts
    piv = agg.pivot(index="drift_type", columns="policy", values="n_retrains_mean")
    fig, ax = plt.subplots(figsize=(8, 4))
    piv.plot.bar(ax=ax, color=[COLORS[c] for c in piv.columns], rot=0)
    ax.set(title="Retraining runs per stream", xlabel="", ylabel="# retrains")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(out_dir / "retrain_counts.png")
    plt.close(fig)
