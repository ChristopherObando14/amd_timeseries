"""
tests/test_evaluate.py
=======================
Tests unitarios para src/models/evaluate.py.
"""

import pytest
import numpy as np
import pandas as pd

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.models.evaluate import (
    compute_metrics,
    build_results_dataframe,
    get_best_model,
)
from src.models.train import ModelResult


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

def _make_result(model_type, ts, units, mae_target=0.0) -> ModelResult:
    """Factory de ModelResult para testing."""
    n = 50
    y_real = np.linspace(100, 150, n)
    y_pred = y_real + mae_target  # error controlado
    return ModelResult(
        key=f"{model_type}_ts{ts}_u{units}",
        model_type=model_type,
        time_step=ts,
        units=units,
        history={"loss": [0.1, 0.05], "val_loss": [0.12, 0.06]},
        y_pred=y_pred,
        y_real=y_real,
        dates=pd.date_range("2024-01-01", periods=n, freq="B"),
        n_epochs=2,
    )


@pytest.fixture
def sample_results():
    return [
        _make_result("SimpleRNN", 5,  5,  mae_target=10.0),
        _make_result("SimpleRNN", 10, 10, mae_target=5.0),
        _make_result("LSTM",      10, 20, mae_target=3.0),
        _make_result("LSTM",      15, 30, mae_target=7.0),
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestComputeMetrics:

    def test_keys_present(self, sample_results):
        metrics = compute_metrics(sample_results[0])
        for key in ["Modelo", "time_step", "Neuronas", "MAE", "R2", "Épocas"]:
            assert key in metrics

    def test_mae_is_float(self, sample_results):
        metrics = compute_metrics(sample_results[0])
        assert isinstance(metrics["MAE"], float)

    def test_perfect_prediction_mae_is_zero(self):
        r = _make_result("LSTM", 10, 10, mae_target=0.0)
        metrics = compute_metrics(r)
        assert metrics["MAE"] == pytest.approx(0.0, abs=1e-4)


class TestBuildResultsDataframe:

    def test_returns_dataframe(self, sample_results):
        df = build_results_dataframe(sample_results)
        assert isinstance(df, pd.DataFrame)

    def test_correct_number_of_rows(self, sample_results):
        df = build_results_dataframe(sample_results)
        assert len(df) == len(sample_results)

    def test_sorted_by_mae(self, sample_results):
        df = build_results_dataframe(sample_results)
        assert list(df["MAE"]) == sorted(df["MAE"].tolist())


class TestGetBestModel:

    def test_best_by_mae(self, sample_results):
        df = build_results_dataframe(sample_results)
        best = get_best_model(df, criterion="MAE")
        assert best["MAE"] == df["MAE"].min()

    def test_best_by_r2(self, sample_results):
        df = build_results_dataframe(sample_results)
        best = get_best_model(df, criterion="R2")
        assert best["R2"] == df["R2"].max()

    def test_invalid_criterion_raises(self, sample_results):
        df = build_results_dataframe(sample_results)
        with pytest.raises(ValueError, match="criterion"):
            get_best_model(df, criterion="INVALID")
