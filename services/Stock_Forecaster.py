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
import requests

# === API Keys (you can replace 'demo' with your real keys in environment variables) ===
TWELVE_DATA_API_KEY = os.getenv("TWELVE_API_KEY", "demo")
ALPHA_VANTAGE_API_KEY = os.getenv("ALPHA_API_KEY", "demo")

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


def fetch_fallback_stock_data(symbol: str, start: str, end: str):
    try:
        # === Try Twelve Data ===
        twelve_url = (
            f"https://api.twelvedata.com/time_series?symbol={symbol}&interval=1day"
            f"&start_date={start}&end_date={end}&apikey={TWELVE_DATA_API_KEY}&format=JSON"
        )
        res = requests.get(twelve_url)
        if res.ok:
            json_data = res.json()
            if "values" in json_data:
                df = pd.DataFrame(json_data["values"])
                df["Date"] = pd.to_datetime(df["datetime"]).dt.strftime("%Y-%m-%d")
                df["Close"] = df["close"].astype(float)
                return df[["Date", "Close"]].sort_values("Date")

        # === Try Alpha Vantage ===
        alpha_url = (
            f"https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}"
            f"&outputsize=full&apikey={ALPHA_VANTAGE_API_KEY}"
        )
        res = requests.get(alpha_url)
        if res.ok:
            json_data = res.json().get("Time Series (Daily)", {})
            if json_data:
                records = [
                    {"Date": date, "Close": float(day["4. close"])}
                    for date, day in json_data.items()
                    if start <= date <= end
                ]
                return pd.DataFrame(records).sort_values("Date")
    except Exception as e:
        print("❌ Fallback failed:", str(e))
        return None
    return None


def train_and_save_model(symbol: str):
    yf_symbol = symbol.replace("-", ".")
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
            "1d": end_date - timedelta(days=2),
        }

        start_date = period_map[period]
        df = yf.download(symbol, start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'))

        if df.empty:
            fallback_df = fetch_fallback_stock_data(symbol, start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
            if fallback_df is None or fallback_df.empty:
                raise HTTPException(status_code=404, detail=f"No data for {symbol} in period {period} from any source")
            df = fallback_df
        else:
            df = df[['Close']].reset_index()
            df.columns = ['Date', 'Close']
            df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d')

        if period == "1d":
            df = df.tail(2)

        return {"symbol": symbol, "data": df.to_dict(orient="records")}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Historical fetch error: {str(e)}")


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
