from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from features import create_features, split_features_target


try:
    from xgboost import XGBRegressor

    HAS_XGBOOST = True
except ImportError:
    from sklearn.ensemble import GradientBoostingRegressor as XGBRegressor

    HAS_XGBOOST = False


def _download_data(symbol: str, start: str, end: str | None) -> pd.DataFrame:
    data = yf.download(symbol, start=start, end=end, progress=False, auto_adjust=False)
    if data.empty:
        raise ValueError(
            f"No data returned for symbol={symbol}. Check ticker and date range."
        )
    return data


def _build_model(random_state: int = 42) -> Pipeline:
    if HAS_XGBOOST:
        xgb = XGBRegressor(
            n_estimators=400,
            learning_rate=0.03,
            max_depth=4,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            random_state=random_state,
        )
    else:
        xgb = XGBRegressor(random_state=random_state)

    rf = RandomForestRegressor(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=2,
        random_state=random_state,
        n_jobs=-1,
    )

    stack = StackingRegressor(
        estimators=[("xgb", xgb), ("rf", rf)],
        final_estimator=Ridge(alpha=1.0),
        passthrough=True,
        n_jobs=-1,
    )

    return Pipeline([("scaler", StandardScaler()), ("model", stack)])


def _time_series_cv_rmse(model: Pipeline, X: pd.DataFrame, y: pd.Series, splits: int = 5) -> float:
    tscv = TimeSeriesSplit(n_splits=splits)
    rmses: list[float] = []
    for train_idx, val_idx in tscv.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model.fit(X_train, y_train)
        pred = model.predict(X_val)
        rmse = mean_squared_error(y_val, pred, squared=False)
        rmses.append(rmse)
    return float(np.mean(rmses))


def train(symbol: str, start: str, end: str | None, model_path: Path, random_state: int = 42) -> None:
    raw_data = _download_data(symbol, start, end)
    featured = create_features(raw_data)
    X, y = split_features_target(featured)

    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    model = _build_model(random_state=random_state)
    cv_rmse = _time_series_cv_rmse(model, X_train, y_train)

    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    rmse = mean_squared_error(y_test, preds, squared=False)
    mae = mean_absolute_error(y_test, preds)
    directional_acc = float(
        np.mean(np.sign(np.diff(y_test.to_numpy())) == np.sign(np.diff(preds)))
    )

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "pipeline": model,
            "feature_columns": X.columns.tolist(),
            "symbol": symbol,
            "train_start": start,
            "train_end": end,
            "metrics": {
                "cv_rmse": cv_rmse,
                "test_rmse": float(rmse),
                "test_mae": float(mae),
                "directional_accuracy": directional_acc,
            },
        },
        model_path,
    )

    print(f"Model saved to: {model_path}")
    print(f"Model family: {'XGBoost + RandomForest + Ridge stack' if HAS_XGBOOST else 'GradientBoosting + RandomForest + Ridge stack'}")
    print(f"CV RMSE: {cv_rmse:.4f}")
    print(f"Test RMSE: {rmse:.4f}")
    print(f"Test MAE: {mae:.4f}")
    print(f"Directional Accuracy: {directional_acc:.2%}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a stock price prediction model.")
    parser.add_argument("--symbol", type=str, default="AAPL", help="Ticker symbol, e.g., AAPL")
    parser.add_argument("--start", type=str, default="2015-01-01", help="Training start date YYYY-MM-DD")
    parser.add_argument("--end", type=str, default=None, help="Training end date YYYY-MM-DD")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("models/stock_price_model.joblib"),
        help="Path to save trained model",
    )
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(
        symbol=args.symbol,
        start=args.start,
        end=args.end,
        model_path=args.model_path,
        random_state=args.random_state,
    )
