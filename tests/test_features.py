import numpy as np
import pandas as pd
import pytest

from sentinelml.features.engineering import build_preprocessor, select_features
from sentinelml.models.train import classification_metrics, evaluate, train_model


def test_preprocessor_handles_nulls_and_unseen_categories(stream, settings):
    prep = build_preprocessor(settings.numeric_features, settings.categorical_features)
    train = stream[stream.batch_id < 8]
    prep.fit(train[settings.features])
    test = train[settings.features].head(5).copy()
    test.loc[test.index[0], "sensor_b"] = np.nan
    test.loc[test.index[1], "region"] = "mars"
    out = prep.transform(test)
    assert not np.isnan(out).any()
    assert out.shape[0] == 5


def test_select_features_raises_on_missing():
    with pytest.raises(KeyError):
        select_features(pd.DataFrame({"a": [1]}), ["a", "b"])


def test_model_beats_chance_before_drift(stream, settings):
    train = stream[stream.batch_id < 8]
    test = stream[(stream.batch_id >= 8) & (stream.batch_id < 11)]
    model = train_model(train)
    m = evaluate(model, test)
    assert m["roc_auc"] > 0.8


def test_metrics_single_class_is_nan():
    m = classification_metrics(np.ones(10), np.linspace(0, 1, 10))
    assert np.isnan(m["roc_auc"])
    assert m["recall"] == pytest.approx(0.5)


def test_no_target_leakage(settings):
    assert settings.target not in settings.features
    assert "concept_alpha" not in settings.features
