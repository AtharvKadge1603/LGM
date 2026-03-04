from __future__ import annotations

import numpy as np
import pandas as pd


def _rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)

    avg_gains = gains.ewm(alpha=1 / period, adjust=False).mean()
    avg_losses = losses.ewm(alpha=1 / period, adjust=False).mean()

    rs = avg_gains / avg_losses.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _macd(close: pd.Series, short: int = 12, long: int = 26, signal: int = 9) -> tuple[pd.Series, pd.Series]:
    short_ema = close.ewm(span=short, adjust=False).mean()
    long_ema = close.ewm(span=long, adjust=False).mean()
    macd_line = short_ema - long_ema
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    return macd_line, signal_line


def create_features(data: pd.DataFrame) -> pd.DataFrame:
    """Build technical and statistical features from OHLCV stock data."""
    df = data.copy()

    if "Adj Close" in df.columns:
        close_col = "Adj Close"
    else:
        close_col = "Close"

    close = df[close_col]

    df["return_1d"] = close.pct_change(1)
    df["return_5d"] = close.pct_change(5)
    df["return_10d"] = close.pct_change(10)

    df["volatility_10d"] = df["return_1d"].rolling(10).std()
    df["volatility_20d"] = df["return_1d"].rolling(20).std()

    df["sma_10"] = close.rolling(10).mean()
    df["sma_20"] = close.rolling(20).mean()
    df["sma_50"] = close.rolling(50).mean()
    df["ema_10"] = close.ewm(span=10, adjust=False).mean()
    df["ema_20"] = close.ewm(span=20, adjust=False).mean()

    df["rsi_14"] = _rsi(close, period=14)
    df["macd"], df["macd_signal"] = _macd(close)

    rolling_std_20 = close.rolling(20).std()
    df["bollinger_high"] = df["sma_20"] + 2 * rolling_std_20
    df["bollinger_low"] = df["sma_20"] - 2 * rolling_std_20
    df["bollinger_width"] = (df["bollinger_high"] - df["bollinger_low"]) / df["sma_20"]

    df["volume_change_1d"] = df["Volume"].pct_change(1)
    df["volume_ma_20"] = df["Volume"].rolling(20).mean()

    df["target_next_close"] = close.shift(-1)

    return df.dropna().copy()


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    features = df.drop(columns=["target_next_close"])

    # Remove raw price columns from the feature matrix to reduce leakage risk.
    drop_cols = [c for c in ["Open", "High", "Low", "Close", "Adj Close"] if c in features.columns]
    if drop_cols:
        features = features.drop(columns=drop_cols)

    return features, df["target_next_close"]
