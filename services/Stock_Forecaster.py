from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from typing import List
from datetime import datetime
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import joblib
import os

app = FastAPI()

MODEL_PATH = "services/rf_model.pkl"
SCALER_PATH = "services/rf_scaler.pkl"


# === 🧠 FEATURE ENGINEERING ===
def engineer_features(df):
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    df['Return'] = df['Close'].pct_change()
    df['Volatility'] = df['Return'].rolling(window=10).std()

    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0).rolling(window=14).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
    rs = gain / loss
    df['RSI_14'] = 100 - (100 / (1 + rs))

    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema12 - ema26

    rolling_mean = df['Close'].rolling(window=20).mean()
    rolling_std = df['Close'].rolling(window=20).std()
    df['Bollinger_Upper'] = rolling_mean + 2 * rolling_std
    df['Bollinger_Lower'] = rolling_mean - 2 * rolling_std

    df.dropna(inplace=True)
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col] = df[col].fillna(df[col].mean())

    return df


# === 🏋️‍♂️ TRAIN FUNCTION ===
def train_rf_model(df):
    features = ['SMA_50', 'EMA_20', 'Volatility', 'RSI_14',
                'MACD', 'Bollinger_Upper', 'Bollinger_Lower']
    target = df['Close'].values

    test_size = int(0.2 * len(df))
    train_size = len(df) - test_size

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(df[features])
    x_train, y_train = x_scaled[:train_size], target[:train_size]

    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        min_samples_leaf=2,
        max_features='sqrt',
        random_state=42
    )
    model.fit(x_train, y_train)

    # Save model & scaler
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    return True


# === 🔍 PREDICT FUNCTION ===
def predict_rf(df):
    if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
        raise FileNotFoundError("Model or Scaler not trained yet.")

    features = ['SMA_50', 'EMA_20', 'Volatility', 'RSI_14',
                'MACD', 'Bollinger_Upper', 'Bollinger_Lower']
    target = df['Close'].values

    test_size = int(0.2 * len(df))
    train_size = len(df) - test_size
    test_index = df.iloc[train_size:].index.strftime("%Y-%m-%d").tolist()

    x = df[features]
    scaler = joblib.load(SCALER_PATH)
    model = joblib.load(MODEL_PATH)

    x_scaled = scaler.transform(x)
    x_test, y_test = x_scaled[train_size:], target[train_size:]
    predictions = model.predict(x_test)

    rmse = np.sqrt(mean_squared_error(y_test, predictions))

    return {
        "dates": test_index,
        "actual": y_test.tolist(),
        "predicted": predictions.tolist(),
        "rmse": rmse
    }


# === 🔁 TRAIN ENDPOINT ===
@app.post("/train_rf")
def train_rf(
    symbol: str = Query(..., description="Stock symbol like AAPL"),
    start_date: str = Query(..., description="Start date (YYYY-MM-DD)"),
    end_date: str = Query(..., description="End date (YYYY-MM-DD)")
):
    try:
        df = yf.download(symbol, start=start_date, end=end_date, threads=True)
        if df.empty:
            raise HTTPException(status_code=404, detail="No data fetched")
        df = engineer_features(df)
        train_rf_model(df)
        return {"message": "Model trained and saved successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# === 📈 PREDICT ENDPOINT ===
@app.get("/predict_rf")
def predict_rf_api(
    symbol: str = Query(..., description="Stock symbol like AAPL"),
    start_date: str = Query(..., description="Start date (YYYY-MM-DD)"),
    end_date: str = Query(..., description="End date (YYYY-MM-DD)")
):
    try:
        df = yf.download(symbol, start=start_date, end=end_date, threads=True)
        if df.empty:
            raise HTTPException(status_code=404, detail="No data fetched")
        df = engineer_features(df)
        result = predict_rf(df)
        return {"symbol": symbol, **result}
    except FileNotFoundError as fnf:
        raise HTTPException(status_code=400, detail=str(fnf))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
