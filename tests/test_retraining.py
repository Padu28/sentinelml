import pandas as pd

from sentinelml.ingestion.loaders import StreamReplayer, load_stream
from sentinelml.models.train import train_model
from sentinelml.retraining.policies import DriftAwarePolicy, PeriodicPolicy, StaticPolicy
from sentinelml.retraining.registry import ModelRegistry, promotion_rule
from sentinelml.retraining.simulation import recovery_batches, simulate


def test_promotion_rule():
    assert promotion_rule(0.80, 0.85, 0.01, 5, 50).promote
    assert not promotion_rule(0.80, 0.805, 0.01, 5, 50).promote
    assert not promotion_rule(0.80, 0.95, 0.01, 80, 50).promote


def test_policies():
    assert not StaticPolicy().should_retrain(0, {"feature_drift": True})
    p = PeriodicPolicy(every=3)
    assert [p.should_retrain(i, {}) for i in range(6)] == [False, False, True, False, False, True]
    d = DriftAwarePolicy(cooldown=2, min_episode_batches=2)
    assert not d.should_retrain(0, {"feature_drift": True, "quality_fault": True})  # quality ≠ drift
    assert not d.should_retrain(1, {"feature_drift": True})   # first alarm → episode starts
    assert d.should_retrain(2, {})                             # enough evidence


def test_recovery_batches():
    auc = [0.9] * 5 + [0.6, 0.7, 0.88, 0.9]
    assert recovery_batches(auc, 5, 0.9, 0.95, w=1) == 2
    assert recovery_batches([0.9] * 6, 3, 0.9, 0.95) == 0


def test_registry_round_trip(tmp_db, tmp_path, stream):
    reg = ModelRegistry(tmp_db, tmp_path / "models")
    m = train_model(stream[stream.batch_id < 6])
    v1 = reg.register(m, "logistic_regression", {"roc_auc": 0.9}, 1500, status="champion")
    v2 = reg.register(m, "logistic_regression", {"roc_auc": 0.92}, 1500, status="champion")
    assert reg.champion()[0] == v2
    statuses = {r["version"]: r["status"] for r in reg.list_versions()}
    assert statuses == {v1: "archived", v2: "champion"}


def test_replayer_is_chronological():
    df = load_stream("sudden", seed=0)
    ts = [b.timestamp.min() for _, b in StreamReplayer(df.head(1000))]
    assert ts == sorted(ts)


def test_drift_aware_beats_static_on_sudden_drift():
    df = load_stream("sudden", seed=0)
    from sentinelml.common.config import get_settings
    mon = get_settings().monitoring
    static = simulate(df, StaticPolicy(), "sudden", 0).summary
    adaptive = simulate(df, DriftAwarePolicy(cooldown=mon["drift"]["cooldown_batches"]), "sudden", 0).summary
    assert adaptive["post_drift_auc"] > static["post_drift_auc"]
    assert adaptive["detection_delay"] is not None
    assert isinstance(pd.Series([adaptive["mean_auc"]]).iloc[0], float)
