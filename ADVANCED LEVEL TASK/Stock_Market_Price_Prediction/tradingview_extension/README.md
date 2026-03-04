# TradingView Extension Integration

This folder contains a minimal Chrome extension that overlays your model's **predicted next close** on TradingView.

## Important limitation
TradingView Pine Script cannot directly call your external Python model endpoint, so direct in-chart model inference is not possible only with Pine. This extension works around that by querying your local API from the browser.

## 1) Start your model API

```bash
python src/serve.py --model-path models/stock_price_model.joblib --host 0.0.0.0 --port 8000
```

## 2) Load extension in Chrome
1. Open `chrome://extensions`
2. Enable **Developer mode**
3. Click **Load unpacked**
4. Select this folder: `tradingview_extension`

## 3) Open TradingView
- Visit `https://www.tradingview.com/chart/`
- The badge at the top-right will show prediction values from `http://127.0.0.1:8000/predict/<symbol>`.

## Notes
- If data doesn't show, verify API health: `http://127.0.0.1:8000/health`
- If your browser blocks local HTTP requests from HTTPS pages, run Chrome with local CORS allowances in development or host API behind HTTPS tunnel.
