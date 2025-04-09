import os
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import ttk, messagebox
from datetime import datetime

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor

import lightgbm as lgb
import xgboost as xgb
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Dropout, Bidirectional
from keras.callbacks import ReduceLROnPlateau
from keras.optimizers import Adam
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from pmdarima import auto_arima
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
import warnings
warnings.filterwarnings('ignore')


# ==== Feature Engineering ====
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

    # Drop rows with NaN from rolling windows
    return df.dropna()


# ==== Model Builders ====
def build_lstm_model(X_train, y_train, path):
    """Builds or loads an LSTM model from disk."""
    if os.path.exists(path):
        print(f"Loading existing LSTM model from {path}...")
        return load_model(path)

    print(f"Training new LSTM model... saving to {path}")
    model = Sequential([
        Bidirectional(LSTM(64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2]))),
        Dropout(0.3),
        Bidirectional(LSTM(32)),
        Dropout(0.3),
        Dense(1)
    ])
    model.compile(optimizer=Adam(0.001), loss='mean_squared_error')
    callbacks = [ReduceLROnPlateau(monitor='loss', factor=0.5, patience=3, min_lr=1e-5)]
    model.fit(X_train, y_train, epochs=25, batch_size=16, verbose=1, callbacks=callbacks)
    model.save(path)
    return model

def build_rf_model(X, y):
    """Builds a RandomForestRegressor."""
    model = RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42)
    model.fit(X, y)
    return model

def build_prophet(df):
    """Builds a Prophet model using the 'Close' column as 'y'."""
    p_df = df.reset_index()[['Date', 'Close']].rename(columns={'Date': 'ds', 'Close': 'y'})
    model = Prophet(yearly_seasonality=True)
    model.fit(p_df)
    return model

def calculate_rmse(y_true, y_pred):
    """Calculates RMSE between two 1-D arrays."""
    return np.sqrt(mean_squared_error(y_true, y_pred))


# ==== Main GUI Logic ====
def analyze_stock():
    symbol = symbol_entry.get().upper()
    start = start_entry.get()
    end = end_entry.get()

    try:
        # 1) Attempt up to 3 times to download
        for attempt in range(3):
            try:
                df = yf.download(symbol, start=start, end=end, threads=False)
                if not df.empty:
                    break
            except Exception as e:
                if attempt == 2:
                    raise e

        if df.empty:
            raise ValueError("No data fetched. Possibly a yfinance parse error.")

        # Feature engineering
        df = engineer_features(df)

        # 2) LSTM needs at least 60 days
        if len(df) < 60:
            raise ValueError("Not enough data for LSTM (need at least 60 days)")

        # Common train/test split
        test_size = int(0.2 * len(df))
        train_size = len(df) - test_size
        test_index = df.index[train_size:]

        features = ['SMA_50','EMA_20','Volatility','RSI_14','MACD','Bollinger_Upper','Bollinger_Lower']
        # Scale for non-LSTM
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df[features])
        y = df['Close'].values

        X_train, X_test = X_scaled[:train_size], X_scaled[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

        # === LightGBM
        lgb_model = lgb.train(
            {'objective': 'regression', 'learning_rate': 0.01},
            lgb.Dataset(X_train, label=y_train),
            num_boost_round=100
        )
        lgb_preds = lgb_model.predict(X_test)

        # === XGBoost
        xgb_model = xgb.XGBRegressor(objective='reg:squarederror', learning_rate=0.01)
        xgb_model.fit(X_train, y_train)
        xgb_preds = xgb_model.predict(X_test)

        # === RandomForest
        rf_model = build_rf_model(X_train, y_train)
        rf_preds = rf_model.predict(X_test)

        # === ARIMA (log-diff)
        log_close = np.log(df['Close'])
        diff_log = log_close.diff().dropna()
        arima_train = diff_log[:train_size]
        arima_model = auto_arima(arima_train, seasonal=False)
        arima_forecast = arima_model.predict(n_periods=test_size)
        last_log = log_close.iloc[train_size - 1]

        # Syntax fix: added final bracket
        arima_preds = np.exp(
            np.r_[last_log, last_log + np.cumsum(arima_forecast)][1:]
        )

        # === Prophet
        # Only train on the training portion
        prophet_model = build_prophet(df.iloc[:train_size])
        future = prophet_model.make_future_dataframe(periods=test_size)
        forecast = prophet_model.predict(future)
        prophet_preds = forecast['yhat'].values[-test_size:]

        # === LSTM
        lstm_data = df[['Close'] + features]
        mm_scaler = MinMaxScaler()
        lstm_scaled = mm_scaler.fit_transform(lstm_data)

        X_lstm, y_lstm = [], []
        for i in range(60, len(lstm_scaled)):
            X_lstm.append(lstm_scaled[i-60:i])
            y_lstm.append(lstm_scaled[i, 0])
        X_lstm = np.array(X_lstm)
        y_lstm = np.array(y_lstm)

        lstm_train_size = train_size - 60
        if lstm_train_size < 0:
            raise ValueError("Not enough data to accommodate 60-day LSTM window + train set.")
        
        X_train_lstm = X_lstm[:lstm_train_size]
        y_train_lstm = y_lstm[:lstm_train_size]
        X_test_lstm  = X_lstm[lstm_train_size:]
        y_test_lstm  = y_lstm[lstm_train_size:]

        lstm_model = build_lstm_model(X_train_lstm, y_train_lstm, f"{symbol}_lstm.keras")
        lstm_preds = lstm_model.predict(X_test_lstm)
        # Inverse scale LSTM predictions
        lstm_preds = mm_scaler.inverse_transform(
            np.concatenate([lstm_preds, np.zeros((len(lstm_preds), len(features)))], axis=1)
        )[:, 0]

        # === Compute RMSE
        lgb_rmse = calculate_rmse(y_test, lgb_preds)
        xgb_rmse = calculate_rmse(y_test, xgb_preds)
        rf_rmse  = calculate_rmse(y_test, rf_preds)
        arima_rmse = calculate_rmse(y_test, arima_preds)
        prophet_rmse = calculate_rmse(y_test, prophet_preds)
        lstm_rmse = calculate_rmse(y_test_lstm, lstm_preds)

        # Print shapes debug
        print(f"Test sizes => Main: {len(y_test)}, LSTM preds: {len(lstm_preds)}, ARIMA preds: {len(arima_preds)}")

        # === Plot
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(test_index, y_test, label='Actual', linewidth=2)
        ax.plot(test_index, arima_preds,    label='ARIMA')
        ax.plot(test_index, rf_preds,       label='RF')
        ax.plot(test_index, xgb_preds,      label='XGBoost')
        ax.plot(test_index, lgb_preds,      label='LightGBM')
        ax.plot(test_index, prophet_preds,  label='Prophet')

        # LSTM might be 60 steps shorter, so we align at the end
        ax.plot(test_index[-len(lstm_preds):], lstm_preds, label='LSTM', color='green')
        ax.legend()
        ax.set_title(f"{symbol} Model Predictions")

        # Send to GUI
        for widget in plot_frame.winfo_children():
            widget.destroy()
        canvas = FigureCanvasTkAgg(fig, master=plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Show metrics
        metrics_label.config(text=(
            f"RMSE:\nLSTM: {lstm_rmse:.2f} | ARIMA: {arima_rmse:.2f} | RF: {rf_rmse:.2f}\n"
            f"XGBoost: {xgb_rmse:.2f} | LightGBM: {lgb_rmse:.2f} | Prophet: {prophet_rmse:.2f}"
        ))

    except Exception as e:
        messagebox.showerror("Error", str(e))


# GUI Setup
root = tk.Tk()
root.title("Stock Forecasting GUI")
root.geometry("1200x800")

frame_input = ttk.Frame(root, padding=10)
frame_input.pack()

symbol_entry = ttk.Entry(frame_input)
symbol_entry.insert(0, "AAPL")
symbol_entry.grid(row=0, column=1)
ttk.Label(frame_input, text="Stock Symbol:").grid(row=0, column=0)

start_entry = ttk.Entry(frame_input)
start_entry.insert(0, "2015-01-01")
start_entry.grid(row=1, column=1)
ttk.Label(frame_input, text="Start Date:").grid(row=1, column=0)

end_entry = ttk.Entry(frame_input)
end_entry.insert(0, "2024-01-01")
end_entry.grid(row=2, column=1)
ttk.Label(frame_input, text="End Date:").grid(row=2, column=0)

ttk.Button(frame_input, text="Analyze", command=analyze_stock).grid(row=3, column=0, columnspan=2, pady=10)

metrics_label = ttk.Label(root, text="Model Performance Metrics will appear here.", font=("Arial", 12))
metrics_label.pack(pady=10)

plot_frame = tk.Frame(root, bg="white", relief=tk.SUNKEN, borderwidth=2)
plot_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

root.mainloop()
