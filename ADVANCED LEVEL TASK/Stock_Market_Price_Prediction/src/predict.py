from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import yfinance as yf

from features import create_features, split_features_target


def predict_next_close(model_path: Path, latest_symbol: str | None = None) -> None:
    artifact = joblib.load(model_path)
    model = artifact["pipeline"]
    symbol = latest_symbol or artifact["symbol"]

    raw = yf.download(symbol, period="1y", progress=False, auto_adjust=False)
    feat = create_features(raw)
    X, _ = split_features_target(feat)

    needed_cols = artifact["feature_columns"]
    X = X[needed_cols]

    latest_row = X.iloc[[-1]]
    pred = float(model.predict(latest_row)[0])

    print(f"Symbol: {symbol}")
    print(f"Predicted next trading day close: {pred:.2f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict next day close with a trained stock model.")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("models/stock_price_model.joblib"),
        help="Path to a trained model artifact",
    )
    parser.add_argument("--symbol", type=str, default=None, help="Optional symbol override")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    predict_next_close(model_path=args.model_path, latest_symbol=args.symbol)
