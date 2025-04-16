from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, Query, HTTPException
from fastapi.responses import JSONResponse
import pandas as pd
import yfinance as yf
import joblib
import os
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

symbols = [
    "AAPL", "META", "AMZN", "NFLX", "GOOGL",
    "MSFT", "TSLA", "NVDA", "BRK-B", "JPM",
    "V", "JNJ", "WMT", "UNH", "PG"
]

output_dir = "checkpoints"
os.makedirs(output_dir, exist_ok=True)

def train_and_save_model(symbol: str):
    yf_symbol = symbol.replace("-", ".")  # e.g. BRK-B → BRK.B
    df = yf.download(yf_symbol, start='2014-01-01', end=datetime.today().strftime('%Y-%m-%d'))

    df = df[['Close']].copy().reset_index()
    df.columns = ['Date', 'Close']
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df = df.dropna()

    scaler = MinMaxScaler()
    df_scaled = scaler.fit_transform(df[['Close', 'SMA_50', 'EMA_20']])
    X = df_scaled[:, 1:3]
    y = df_scaled[:, 0]

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    joblib.dump(model, os.path.join(output_dir, f"{symbol}_rf_model.pkl"))
    joblib.dump(scaler, os.path.join(output_dir, f"{symbol}_scaler.pkl"))

    # Predict future 15 days
    future_dates = pd.date_range(datetime.today(), periods=15, freq='D')
    last_row = df.iloc[-1]
    latest_sma_50 = last_row['SMA_50']
    latest_ema_20 = last_row['EMA_20']

    predictions = []
    for date in future_dates:
        scaled_input = scaler.transform([[0, latest_sma_50, latest_ema_20]])[:, 1:3]
        scaled_pred = model.predict(scaled_input)[0]
        close_pred = scaler.inverse_transform([[scaled_pred, latest_sma_50, latest_ema_20]])[0][0]
        predictions.append((date.strftime('%Y-%m-%d'), close_pred))
        latest_ema_20 = (latest_ema_20 * 19 + close_pred) / 20
        latest_sma_50 = (latest_sma_50 * 49 + close_pred) / 50

    pd.DataFrame(predictions, columns=['Date', 'Predicted Close'])\
      .to_csv(os.path.join(output_dir, f"{symbol}_future_predictions.csv"), index=False)

    print(f"✅ Model and predictions saved for {symbol}")

for sym in symbols:
    model_path = os.path.join(output_dir, f"{sym}_rf_model.pkl")
    pred_path = os.path.join(output_dir, f"{sym}_future_predictions.csv")
    if not os.path.exists(model_path) or not os.path.exists(pred_path):
        train_and_save_model(sym)

@app.get("/")
def root():
    return {"message": "✅ Multi-stock Forecast API is running."}

@app.get("/predict")
def get_predictions(symbol: str = Query("AAPL")):
    try:
        path = os.path.join(output_dir, f"{symbol}_future_predictions.csv")
        df = pd.read_csv(path)
        return {
            "symbol": symbol,
            "dates": df["Date"].tolist(),
            "predicted": df["Predicted Close"].tolist(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/historical")
def get_historical(
    period: str = Query("1y", enum=["5y", "3y", "1y", "6mo", "3mo", "1mo", "7d", "1d"]),
    symbol: str = Query("AAPL")
):
    try:
        end_date = datetime.today()
        period_map = {
            "5y": end_date - timedelta(days=5 * 365),
            "3y": end_date - timedelta(days=3 * 365),
            "1y": end_date - timedelta(days=365),
            "6mo": end_date - timedelta(days=180),
            "3mo": end_date - timedelta(days=90),
            "1mo": end_date - timedelta(days=30),
            "7d": end_date - timedelta(days=7),
            "1d": end_date - timedelta(days=2),  # ✅ Fallback: go back 2 days to catch the last available day
        }

        start_date = period_map[period]
        df = yf.download(symbol, start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'))

        if df.empty:
            raise HTTPException(status_code=404, detail=f"No data for {symbol} in period {period}")

        df = df[['Close']].reset_index()
        df.columns = ['Date', 'Close']
        df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d')

        # If it's "1d", return only the last 1–2 entries
        if period == "1d":
            df = df.tail(2)

        return {"symbol": symbol, "data": df.to_dict(orient="records")}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics")
def get_model_metrics(symbol: str = Query("AAPL")):
    try:
        model = joblib.load(os.path.join(output_dir, f"{symbol}_rf_model.pkl"))
        scaler = joblib.load(os.path.join(output_dir, f"{symbol}_scaler.pkl"))
        return {
            "model": f"Random Forest - {symbol}",
            "n_estimators": model.n_estimators,
            "max_depth": model.max_depth,
            "features_range": scaler.feature_range,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Metrics error: {str(e)}")
