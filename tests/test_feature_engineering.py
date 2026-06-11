"""
tests/test_feature_engineering.py
==================================
Tests unitarios para src/features/feature_engineering.py.
"""

import pytest
import numpy as np
import pandas as pd

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import DataConfig, FeatureConfig
from src.features.feature_engineering import (
    create_sequences,
    split_train_test,
    fit_and_scale,
    build_train_test_sequences,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_df() -> pd.DataFrame:
    dates = pd.bdate_range("2020-01-01", periods=100)
    prices = np.linspace(50, 150, 100) + np.random.RandomState(42).randn(100) * 5
    return pd.DataFrame({"Close": prices}, index=dates)


@pytest.fixture
def data_cfg() -> DataConfig:
    return DataConfig(train_ratio=0.80)


@pytest.fixture
def feature_cfg() -> FeatureConfig:
    return FeatureConfig()


@pytest.fixture
def split_data(sample_df, data_cfg):
    return split_train_test(sample_df, data_cfg)


@pytest.fixture
def scaled_data(split_data, feature_cfg):
    train, test = split_data
    return fit_and_scale(train, test, feature_cfg)


# ─────────────────────────────────────────────────────────────────────────────
# Tests: create_sequences
# ─────────────────────────────────────────────────────────────────────────────

class TestCreateSequences:

    def test_output_shape(self):
        series = np.arange(20, dtype=float)
        X, y = create_sequences(series, time_step=5)
        assert X.shape == (15, 5)
        assert y.shape == (15,)

    def test_values_are_correct(self):
        series = np.arange(10, dtype=float)
        X, y = create_sequences(series, time_step=3)
        assert list(X[0]) == [0.0, 1.0, 2.0]
        assert y[0] == 3.0

    def test_time_step_too_large_raises(self):
        series = np.arange(5, dtype=float)
        with pytest.raises(ValueError, match="time_step"):
            create_sequences(series, time_step=10)


# ─────────────────────────────────────────────────────────────────────────────
# Tests: split_train_test
# ─────────────────────────────────────────────────────────────────────────────

class TestSplitTrainTest:

    def test_correct_sizes(self, split_data, sample_df, data_cfg):
        train, test = split_data
        expected_train = int(len(sample_df) * data_cfg.train_ratio)
        assert len(train) == expected_train
        assert len(test)  == len(sample_df) - expected_train

    def test_chronological_order(self, split_data):
        train, test = split_data
        assert train.index[-1] < test.index[0]

    def test_no_overlap(self, split_data):
        train, test = split_data
        assert len(set(train.index) & set(test.index)) == 0

    def test_invalid_ratio_raises(self, sample_df):
        cfg = DataConfig(train_ratio=1.0)
        with pytest.raises(ValueError):
            split_train_test(sample_df, cfg)


# ─────────────────────────────────────────────────────────────────────────────
# Tests: fit_and_scale
# ─────────────────────────────────────────────────────────────────────────────

class TestFitAndScale:

    def test_scaled_column_exists(self, scaled_data):
        train, test, _ = scaled_data
        assert "Close_scaled" in train.columns
        assert "Close_scaled" in test.columns

    def test_train_range_is_0_1(self, scaled_data):
        train, _, _ = scaled_data
        assert train["Close_scaled"].min() == pytest.approx(0.0, abs=1e-6)
        assert train["Close_scaled"].max() == pytest.approx(1.0, abs=1e-6)

    def test_scaler_is_returned(self, scaled_data):
        from sklearn.preprocessing import MinMaxScaler
        _, _, scaler = scaled_data
        assert isinstance(scaler, MinMaxScaler)

    def test_original_close_preserved(self, scaled_data, split_data):
        train_scaled, _, _ = scaled_data
        train_orig, _ = split_data
        pd.testing.assert_series_equal(
            train_scaled["Close"], train_orig["Close"]
        )


# ─────────────────────────────────────────────────────────────────────────────
# Tests: build_train_test_sequences
# ─────────────────────────────────────────────────────────────────────────────

class TestBuildTrainTestSequences:

    def test_output_shapes(self, scaled_data):
        train, test, _ = scaled_data
        time_step = 5
        X_tr, y_tr, X_te, y_te = build_train_test_sequences(train, test, time_step)
        assert X_tr.shape[1:] == (time_step, 1)
        assert X_te.shape[1:] == (time_step, 1)
        assert len(y_tr) == X_tr.shape[0]
        assert len(y_te) == X_te.shape[0]
