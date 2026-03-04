# Stock Market Price Prediction (Advanced AI/ML)

This project predicts the **next trading day closing price** of a stock using an advanced feature-engineering pipeline and an ensemble model stack.

## What makes it advanced?
- Technical indicator feature engineering (RSI, MACD, Bollinger Bands, rolling volatility).
- Multi-model ensemble:
  - XGBoost (or GradientBoosting fallback)
  - RandomForest
  - Ridge meta-learner (stacking)
- Time-series aware validation (`TimeSeriesSplit`) to avoid leakage.
- Tracks both regression and trading-oriented metrics (Directional Accuracy).

## Project Structure

```text
Stock_Market_Price_Prediction/
├── README.md
├── requirements.txt
└── src/
    ├── features.py
    ├── train.py
    └── predict.py
```

## Setup

```bash
cd "ADVANCED LEVEL TASK/Stock_Market_Price_Prediction"
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Train

```bash
python src/train.py --symbol AAPL --start 2015-01-01 --end 2024-12-31
```

Optional arguments:
- `--model-path` custom model output path.
- `--random-state` reproducibility seed.

## Predict next day close

```bash
python src/predict.py --model-path models/stock_price_model.joblib --symbol AAPL
```

## Notes
- Stock markets are noisy and non-stationary; model performance can degrade over time.
- This is for educational/research use, not financial advice.
- Add sentiment, macroeconomic, and options-flow features to improve signal quality.
