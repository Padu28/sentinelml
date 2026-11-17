import numpy as np
import pandas as pd

from sentinelml.validation.quality import clean_batch, run_quality_checks


def test_clean_batch_passes(stream, settings):
    batch = stream[stream.batch_id == 2]
    rep = run_quality_checks(batch, settings.data["schema"])
    assert rep.passed
    assert rep.score >= 90


def test_faulted_batch_fails_with_expected_issues(stream, settings):
    batch = stream[stream.batch_id == 20]
    rep = run_quality_checks(batch, settings.data["schema"])
    checks = {i.check for i in rep.issues}
    assert not rep.passed
    assert {"nulls", "range", "category", "duplicates"} <= checks


def test_missing_column_is_critical(settings):
    df = pd.DataFrame({"sensor_a": [0.1, 0.2]})
    rep = run_quality_checks(df, settings.data["schema"])
    assert not rep.passed
    assert any(i.severity == "critical" for i in rep.issues)


def test_empty_batch(settings):
    rep = run_quality_checks(pd.DataFrame(columns=list(settings.data["schema"]["columns"])), settings.data["schema"])
    assert rep.score == 0 and not rep.passed


def test_clean_batch_removes_impossible_values(stream, settings):
    batch = stream[stream.batch_id == 20]
    out = clean_batch(batch, settings.data["schema"])
    assert out["temp_c"].max() <= 80
    assert (out["load_kw"].dropna() >= 0).all()
    assert len(out) <= len(batch)
    assert not out.duplicated(subset=list(settings.data["schema"]["columns"])).any()


def test_score_is_bounded(settings):
    df = pd.DataFrame({c: [np.nan] * 10 for c in settings.data["schema"]["columns"]})
    rep = run_quality_checks(df, settings.data["schema"])
    assert 0 <= rep.score <= 100
