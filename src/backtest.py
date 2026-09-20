"""
Module 4 of 4 - backtesting.

A paper-trading simulation that turns predicted price moves into long/flat
positions, plus the 95% confidence band used to visualise model uncertainty.

The strategy is deliberately simple so the comparison against buy-and-hold is
readable: go long when the model predicts a rise above `threshold`, close when
it predicts a fall of the same size, never short, no leverage, no costs.
"""

from dataclasses import dataclass, field
from typing import List

import numpy as np


@dataclass
class BacktestResult:
    initial_capital: float
    final_value: float
    strategy_return_pct: float
    buy_hold_return_pct: float
    trades: int
    portfolio_values: List[float] = field(default_factory=list)

    @property
    def beat_buy_hold(self) -> bool:
        return self.strategy_return_pct > self.buy_hold_return_pct

    def summary(self) -> str:
        return (
            f"Final portfolio value : ${self.final_value:,.2f}\n"
            f"Strategy return       : {self.strategy_return_pct:.2f}%\n"
            f"Buy & hold return     : {self.buy_hold_return_pct:.2f}%\n"
            f"Trades                : {self.trades}"
        )


def confidence_band(y_true, y_pred, z: float = 1.96):
    """Symmetric band around the prediction, z standard deviations of residual.

    z=1.96 is the 95% interval under a normal residual assumption. Residuals of
    a price model are usually fat-tailed, so treat this as an indication of
    spread rather than a calibrated probability.
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    std = float(np.std(y_true - y_pred))
    return y_pred - z * std, y_pred + z * std


def simulate_trading_strategy(
    y_true, y_pred, initial_capital: float = 10_000.0, threshold: float = 0.005
) -> BacktestResult:
    """Run the long/flat strategy over the test window.

    At each step the *predicted* move from t to t+1 decides the action, and the
    trade fills at the actual price at t - the decision never sees a price it
    could not have known.
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must be the same length")
    if len(y_true) < 2:
        raise ValueError("need at least two points to simulate")

    capital = float(initial_capital)
    shares = 0.0
    in_position = False
    portfolio_values, trades = [], 0

    for i in range(len(y_true) - 1):
        price_now = y_true[i]
        predicted_change = (y_pred[i + 1] - y_pred[i]) / y_pred[i]

        if predicted_change > threshold and not in_position:
            shares = capital / price_now
            capital = 0.0
            in_position = True
            trades += 1
        elif predicted_change < -threshold and in_position:
            capital = shares * price_now
            shares = 0.0
            in_position = False
            trades += 1

        portfolio_values.append(capital + shares * price_now)

    if in_position:                       # liquidate at the last known price
        capital = shares * y_true[-1]
        shares = 0.0
    portfolio_values.append(capital)

    strategy_return = (capital - initial_capital) / initial_capital * 100
    buy_hold_return = (y_true[-1] - y_true[0]) / y_true[0] * 100

    return BacktestResult(
        initial_capital=initial_capital,
        final_value=capital,
        strategy_return_pct=strategy_return,
        buy_hold_return_pct=buy_hold_return,
        trades=trades,
        portfolio_values=portfolio_values,
    )


def plot_predictions(dates, y_true, y_pred, title="Prediction vs actual", z=1.96):
    """Actual vs predicted with the confidence band shaded."""
    import matplotlib.pyplot as plt

    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    lower, upper = confidence_band(y_true, y_pred, z)

    plt.figure(figsize=(14, 7))
    plt.plot(dates, y_true, label="Actual", color="tab:blue", alpha=0.8)
    plt.plot(dates, y_pred, label="Predicted", color="tab:red", linestyle="--")
    band_label = "95%" if abs(z - 1.96) < 1e-6 else f"{z:g} sigma"
    plt.fill_between(dates, lower, upper, color="tab:red", alpha=0.15,
                     label=f"{band_label} confidence interval")
    plt.title(title)
    plt.ylabel("Price ($)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    return plt


def plot_portfolio(result: BacktestResult, title="Portfolio growth"):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 5))
    plt.plot(result.portfolio_values, label="Strategy")
    plt.axhline(result.initial_capital, color="grey", linestyle="--",
                label="Initial capital")
    plt.title(title)
    plt.ylabel("Value ($)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    return plt
