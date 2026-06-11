"""
tests/test_data_loader.py
=========================
Tests unitarios para src/data/data_loader.py.
"""

import pytest
import numpy as np
import pandas as pd

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import DataConfig
from src.data.data_loader import (
    generate_synthetic_gbm,
    validate_dataframe,
    compute_descriptive_stats,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def default_cfg() -> DataConfig:
    return DataConfig()


@pytest.fixture
def synthetic_df(default_cfg) -> pd.DataFrame:
    return generate_synthetic_gbm(default_cfg)


# ─────────────────────────────────────────────────────────────────────────────
# Tests: generate_synthetic_gbm
# ─────────────────────────────────────────────────────────────────────────────

class TestGenerateSyntheticGBM:

    def test_returns_dataframe(self, synthetic_df):
        assert isinstance(synthetic_df, pd.DataFrame)

    def test_has_close_column(self, synthetic_df):
        assert "Close" in synthetic_df.columns

    def test_no_nan_values(self, synthetic_df):
        assert not synthetic_df["Close"].isna().any()

    def test_minimum_price_is_positive(self, synthetic_df):
        assert synthetic_df["Close"].min() >= 5.0

    def test_index_is_datetime(self, synthetic_df):
        assert isinstance(synthetic_df.index, pd.DatetimeIndex)

    def test_reproducibility(self, default_cfg):
        df1 = generate_synthetic_gbm(default_cfg)
        df2 = generate_synthetic_gbm(default_cfg)
        pd.testing.assert_frame_equal(df1, df2)

    def test_max_price_around_185(self, synthetic_df):
        # El GBM se escala al máximo de 185 USD
        assert synthetic_df["Close"].max() == pytest.approx(185.0, rel=0.01)


# ─────────────────────────────────────────────────────────────────────────────
# Tests: validate_dataframe
# ─────────────────────────────────────────────────────────────────────────────

class TestValidateDataframe:

    def test_valid_df_passes(self, synthetic_df):
        validate_dataframe(synthetic_df)  # no debe lanzar excepción

    def test_empty_df_raises(self):
        with pytest.raises(ValueError, match="vacío"):
            validate_dataframe(pd.DataFrame())

    def test_missing_close_column_raises(self):
        df = pd.DataFrame({"Open": [1, 2, 3]})
        with pytest.raises(ValueError, match="Close"):
            validate_dataframe(df)

    def test_nan_in_close_raises(self):
        df = pd.DataFrame({"Close": [1.0, np.nan, 3.0]})
        with pytest.raises(ValueError, match="NaN"):
            validate_dataframe(df)


# ─────────────────────────────────────────────────────────────────────────────
# Tests: compute_descriptive_stats
# ─────────────────────────────────────────────────────────────────────────────

class TestComputeDescriptiveStats:

    def test_returns_dataframe(self, synthetic_df):
        stats = compute_descriptive_stats(synthetic_df)
        assert isinstance(stats, pd.DataFrame)

    def test_includes_skewness_and_kurtosis(self, synthetic_df):
        stats = compute_descriptive_stats(synthetic_df)
        assert "skewness" in stats.index
        assert "kurtosis" in stats.index
