"""
Tests for the pure logic in each module.

Nothing here touches the network or trains a model - ingestion windowing,
scaling and the backtest are all deterministic and testable on synthetic data,
which is the point of splitting the pipeline into modules.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.backtest import (  # noqa: E402
    confidence_band,
    simulate_trading_strategy,
)
from src.data_ingestion import create_sequences, merge_sentiment  # noqa: E402
from src.model import relative_error, rmse  # noqa: E402
from src.sentiment import coverage  # noqa: E402


# --- ingestion ------------------------------------------------------------- #

def test_create_sequences_shapes_and_targets():
    data = np.arange(20, dtype=float).reshape(10, 2)
    X, y = create_sequences(data, seq_length=3)
    assert X.shape == (7, 3, 2)
    assert y.shape == (7, 1)
    # target is the next step's column 0
    assert y[0, 0] == data[3, 0]
    assert y[-1, 0] == data[-1, 0]


def test_create_sequences_empty_when_window_too_long():
    X, y = create_sequences(np.zeros((5, 2)), seq_length=5)
    assert len(X) == 0 and len(y) == 0


def test_merge_sentiment_fills_missing_days_with_zero():
    prices = pd.DataFrame({
        "Date": pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-03"]),
        "Close": [1.0, 2.0, 3.0],
    })
    sent = pd.DataFrame({
        "Date": pd.to_datetime(["2026-01-02"]),
        "Sentiment": [0.5],
    })
    out = merge_sentiment(prices, sent)
    assert list(out["Sentiment"]) == [0.0, 0.5, 0.0]
    assert len(out) == 3


def test_merge_sentiment_handles_no_sentiment_at_all():
    prices = pd.DataFrame({
        "Date": pd.to_datetime(["2026-01-01"]), "Close": [1.0],
    })
    out = merge_sentiment(prices, pd.DataFrame())
    assert list(out["Sentiment"]) == [0.0]


def test_coverage_reports_the_neutral_fill_ratio():
    frame = pd.DataFrame({"Sentiment": [0.0, 0.0, 0.3, 0.0]})
    cov = coverage(frame)
    assert cov["total_days"] == 4
    assert cov["days_with_sentiment"] == 1
    assert cov["ratio"] == pytest.approx(0.25)


# --- metrics --------------------------------------------------------------- #

def test_rmse_is_zero_for_a_perfect_prediction():
    y = np.array([1.0, 2.0, 3.0])
    assert rmse(y, y) == pytest.approx(0.0)


def test_rmse_matches_the_manual_calculation():
    y_true = np.array([10.0, 20.0])
    y_pred = np.array([12.0, 18.0])
    assert rmse(y_true, y_pred) == pytest.approx(2.0)


def test_relative_error_scales_by_mean_price():
    y_true = np.array([100.0, 100.0])
    y_pred = np.array([90.0, 110.0])
    assert relative_error(y_true, y_pred) == pytest.approx(0.1)


# --- backtest -------------------------------------------------------------- #

def test_confidence_band_brackets_the_prediction():
    y_true = np.array([10.0, 12.0, 11.0, 13.0])
    y_pred = np.array([10.5, 11.5, 11.5, 12.5])
    lower, upper = confidence_band(y_true, y_pred)
    assert np.all(lower <= y_pred) and np.all(upper >= y_pred)
    assert np.allclose(upper - y_pred, y_pred - lower)


def test_no_trades_when_predictions_are_flat():
    prices = np.linspace(100, 110, 20)
    flat = np.full(20, 105.0)
    res = simulate_trading_strategy(prices, flat, initial_capital=1000)
    assert res.trades == 0
    assert res.final_value == pytest.approx(1000.0)


def test_strategy_captures_a_predicted_rise():
    # Prediction leads the actual rise, so the strategy should buy and hold up.
    prices = np.array([100.0, 100.0, 110.0, 120.0, 130.0])
    preds = np.array([100.0, 110.0, 120.0, 130.0, 140.0])
    res = simulate_trading_strategy(prices, preds, initial_capital=1000)
    assert res.trades >= 1
    assert res.final_value > 1000.0


def test_buy_hold_return_is_computed_from_the_actual_series():
    prices = np.array([100.0, 105.0, 110.0, 120.0])
    preds = np.array([100.0, 100.0, 100.0, 100.0])
    res = simulate_trading_strategy(prices, preds)
    assert res.buy_hold_return_pct == pytest.approx(20.0)


def test_portfolio_curve_has_one_point_per_step():
    prices = np.linspace(100, 120, 10)
    res = simulate_trading_strategy(prices, prices, initial_capital=500)
    assert len(res.portfolio_values) == len(prices)


def test_mismatched_lengths_are_rejected():
    with pytest.raises(ValueError):
        simulate_trading_strategy(np.array([1.0, 2.0]), np.array([1.0]))


def test_too_short_series_is_rejected():
    with pytest.raises(ValueError):
        simulate_trading_strategy(np.array([1.0]), np.array([1.0]))
