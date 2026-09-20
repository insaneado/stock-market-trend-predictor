"""
Module 1 of 4 - data ingestion.

Pulls OHLCV price history from yfinance, merges the per-day sentiment series
onto it, then scales and windows the result into LSTM-ready sequences.

Scaling is fit on the training split only and applied to the test split, so no
test-set statistics leak into training.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

DEFAULT_FEATURES = ["Close", "Volume", "Sentiment"]


@dataclass
class Dataset:
    """Everything downstream modules need, in one object."""
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    price_scaler: StandardScaler
    feature_scaler: StandardScaler
    test_dates: np.ndarray
    frame: pd.DataFrame
    features: list
    seq_length: int

    @property
    def n_features(self) -> int:
        return len(self.features)

    def inverse_price(self, scaled):
        """Map scaled Close values back to dollars."""
        return self.price_scaler.inverse_transform(np.asarray(scaled).reshape(-1, 1))


def fetch_prices(ticker: str, start: str = "2020-01-01", end: str = None) -> pd.DataFrame:
    """Download OHLCV history and return it with a flat column index."""
    import yfinance as yf

    df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=False)
    if df.empty:
        raise ValueError(f"yfinance returned no rows for {ticker!r}")
    # yfinance returns a MultiIndex when several tickers are requested.
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.reset_index()
    df["Date"] = pd.to_datetime(df["Date"])
    return df


def merge_sentiment(price_df: pd.DataFrame, sentiment_df: pd.DataFrame) -> pd.DataFrame:
    """Left-join sentiment onto prices; days with no news score 0 (neutral)."""
    if sentiment_df is None or sentiment_df.empty:
        out = price_df.copy()
        out["Sentiment"] = 0.0
        return out
    out = pd.merge(price_df, sentiment_df, on="Date", how="left")
    out["Sentiment"] = out["Sentiment"].fillna(0.0)
    return out


def create_sequences(data: np.ndarray, seq_length: int):
    """Window a (T, F) array into (N, seq_length, F) inputs and (N, 1) targets.

    The target is the next step's column 0 (Close).
    """
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length, 0])
    return np.array(X), np.array(y).reshape(-1, 1)


def build_dataset(
    ticker: str = "AAPL",
    start: str = "2020-01-01",
    sentiment_df: pd.DataFrame = None,
    features: list = None,
    seq_length: int = 30,
    train_split: float = 0.8,
) -> Dataset:
    """Full ingestion pipeline: download -> merge -> split -> scale -> window."""
    features = features or DEFAULT_FEATURES
    frame = merge_sentiment(fetch_prices(ticker, start), sentiment_df)

    missing = [f for f in features if f not in frame.columns]
    if missing:
        raise KeyError(f"features not present in data: {missing}")

    values = frame[features].values
    split = int(train_split * len(values))
    if split <= seq_length:
        raise ValueError(
            f"not enough rows ({len(values)}) for seq_length={seq_length}"
        )
    train_raw, test_raw = values[:split], values[split:]

    # Fit on train only - applying test statistics would leak the future.
    feature_scaler = StandardScaler().fit(train_raw)
    train_scaled = feature_scaler.transform(train_raw)
    test_scaled = feature_scaler.transform(test_raw)

    # Separate scaler for column 0 so predictions can be returned in dollars.
    price_scaler = StandardScaler().fit(train_raw[:, [0]])

    X_train, y_train = create_sequences(train_scaled, seq_length)
    X_test, y_test = create_sequences(test_scaled, seq_length)

    start_idx = split + seq_length
    test_dates = frame["Date"].iloc[start_idx:start_idx + len(y_test)].values

    return Dataset(
        X_train=X_train, y_train=y_train,
        X_test=X_test, y_test=y_test,
        price_scaler=price_scaler, feature_scaler=feature_scaler,
        test_dates=test_dates, frame=frame,
        features=features, seq_length=seq_length,
    )
