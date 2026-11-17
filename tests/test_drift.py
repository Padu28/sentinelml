import numpy as np

from sentinelml.ingestion.synthetic import StreamSpec, generate_stream
from sentinelml.monitoring.drift import ADWIN, PageHinkley, detect_window_drift, psi_numeric


def test_psi_identical_is_near_zero():
    rng = np.random.default_rng(0)
    a = rng.normal(size=5000)
    assert psi_numeric(a, rng.normal(size=5000)) < 0.02


def test_psi_detects_shift():
    rng = np.random.default_rng(0)
    assert psi_numeric(rng.normal(size=3000), rng.normal(1.5, 1, size=3000)) > 0.5


def test_window_drift_detects_sudden_drift(stream, settings):
    ref = stream[stream.batch_id < 8]
    pre = stream[stream.batch_id == 10]
    post = stream[stream.batch_id == 16]
    args = (settings.numeric_features, settings.categorical_features)
    assert not detect_window_drift(ref, pre, *args).drift_detected
    assert detect_window_drift(ref, post, *args).drift_detected


def test_no_drift_stream_has_few_alarms(settings):
    df = generate_stream(StreamSpec(n_records=5000, drift_type="none", seed=3))
    ref = df[df.batch_id < 8]
    alarms = sum(
        detect_window_drift(ref, g, settings.numeric_features, settings.categorical_features).drift_detected
        for b, g in df.groupby("batch_id") if b >= 8
    )
    assert alarms <= 1


def test_page_hinkley_fires_on_mean_increase():
    ph = PageHinkley(delta=0.005, threshold=2.5, min_samples=4)
    fired_pre = any(ph.update(0.15) for _ in range(20))
    fired_post = any(ph.update(0.45) for _ in range(10))
    assert not fired_pre and fired_post


def test_adwin_detects_change():
    ad = ADWIN(delta=0.002)
    rng = np.random.default_rng(1)
    pre = [ad.update(x) for x in rng.normal(0.1, 0.02, 200)]
    post = [ad.update(x) for x in rng.normal(0.6, 0.02, 100)]
    assert not any(pre[:150]) and any(post)


def test_drift_types_produce_ground_truth():
    for dt in ("sudden", "gradual", "incremental", "recurring"):
        df = generate_stream(StreamSpec(n_records=4000, drift_start=2000, drift_type=dt, seed=0))
        assert df.loc[df.index < 2000, "concept_alpha"].max() == 0
        assert df.loc[df.index >= 2000, "concept_alpha"].max() > 0
