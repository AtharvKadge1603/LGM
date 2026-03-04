from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import joblib
import yfinance as yf
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from features import create_features, split_features_target


def load_artifact(model_path: Path) -> dict:
    if not model_path.exists():
        raise FileNotFoundError(f"Model artifact not found: {model_path}")
    return joblib.load(model_path)


def create_app(model_path: Path) -> FastAPI:
    artifact = load_artifact(model_path)
    pipeline = artifact["pipeline"]
    feature_columns = artifact["feature_columns"]

    app = FastAPI(title="Stock Prediction API", version="1.0.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok", "model_symbol": artifact.get("symbol")}

    @app.get("/predict/{symbol}")
    def predict(symbol: str) -> dict:
        raw = yf.download(symbol, period="1y", progress=False, auto_adjust=False)
        if raw.empty:
            raise HTTPException(status_code=404, detail=f"No market data found for '{symbol}'.")

        feat = create_features(raw)
        if feat.empty:
            raise HTTPException(status_code=400, detail="Not enough data to build features.")

        X, _ = split_features_target(feat)
        missing = [col for col in feature_columns if col not in X.columns]
        if missing:
            raise HTTPException(status_code=500, detail=f"Feature mismatch: missing columns {missing}")

        latest_row = X[feature_columns].iloc[[-1]]
        pred = float(pipeline.predict(latest_row)[0])

        return {
            "symbol": symbol.upper(),
            "predicted_next_close": round(pred, 4),
            "generated_at": datetime.now(tz=timezone.utc).isoformat(),
            "training_symbol": artifact.get("symbol"),
            "metrics": artifact.get("metrics", {}),
        }

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Serve stock prediction model as REST API.")
    parser.add_argument("--model-path", type=Path, default=Path("models/stock_price_model.joblib"))
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    return parser.parse_args()


if __name__ == "__main__":
    import uvicorn

    args = parse_args()
    uvicorn.run(create_app(args.model_path), host=args.host, port=args.port)
