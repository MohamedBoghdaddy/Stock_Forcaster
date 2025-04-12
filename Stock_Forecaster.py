# === 📦 IMPORTS ===
import os
import warnings
from datetime import datetime
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import tkinter as tk
from tkinter import ttk, messagebox

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb
import xgboost as xgb

from keras.models import Sequential, load_model
from keras.layers import LSTM, Dense, Dropout, Bidirectional
from keras.callbacks import ReduceLROnPlateau
from keras.optimizers import Adam

from pmdarima import auto_arima

warnings.filterwarnings('ignore')



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

    threshold = 0.7 * len(df)
    df = df.dropna(axis=1, thresh=int(threshold))

    fill_strategies = {
        'SMA_50': 'median', 'EMA_20': 'mean', 'Return': 'mean', 'Volatility': 'median',
        'RSI_14': 'mean', 'MACD': 'mean', 'Bollinger_Upper': 'mean', 'Bollinger_Lower': 'mean'
    }

    for col, strategy in fill_strategies.items():
        if col in df.columns:
            if strategy == 'mean':
                df[col] = df[col].fillna(df[col].mean())
            elif strategy == 'median':
                df[col] = df[col].fillna(df[col].median())

    for col in df.select_dtypes(include=[np.number]).columns:
        df[col] = df[col].fillna(df[col].mean())

    df.dropna(inplace=True)
    return df

# Function to check if any predictions are NaN
def check_for_nan(predictions):
    return np.any(np.isnan(predictions))

# Validate and calculate RMSE
if check_for_nan(lgb_preds) or check_for_nan(xgb_preds) or check_for_nan(rf_preds) or check_for_nan(arima_preds) or check_for_nan(lstm_preds):
    messagebox.showerror("Error", "Model predictions contain NaN values.")
    return np.any(np.isnan(predictions))




# === 🔁 MODEL BUILDERS ===
def build_rf_model(x, y):
    model = RandomForestRegressor(
        n_estimators=200, max_depth=15, min_samples_leaf=2,
        max_features='sqrt', random_state=42
    )
    model.fit(x, y)
    return model

def build_lstm_model(x_train, y_train, path):
    if np.isnan(x_train).any() or np.isnan(y_train).any():
        if os.path.exists(path):
            os.remove(path)

    if os.path.exists(path):
        return load_model(path)

    model = Sequential([ 
        Bidirectional(LSTM(64, return_sequences=True, input_shape=(x_train.shape[1], x_train.shape[2]))),
        Dropout(0.3),
        Bidirectional(LSTM(32)),
        Dropout(0.3),
        Dense(1)
    ])
    model.compile(optimizer=Adam(0.001), loss='mean_squared_error')
    model.fit(x_train, y_train, epochs=25, batch_size=16, verbose=0,
              callbacks=[ReduceLROnPlateau(monitor='loss', factor=0.5, patience=3, min_lr=1e-5)])
    model.save(path)
    return model

def calculate_rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))




# === 📊 ANALYSIS ===
def validate_dates(start_date, end_date):
    try:
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        if start >= end:
            raise ValueError("Start date must be earlier than end date.")
        return True
    except ValueError as e:
        print(f"❌ Error: {e}")
        return False

# === 📊 ANALYSIS ===
def analyze_stock():
    symbol = symbol_entry.get().upper()
    start = start_entry.get()
    end = end_entry.get()
    
    if not validate_dates(start, end):
        return false
    
    try:
        df = yf.download(symbol, start=start, end=end, threads=True)
        if df.empty:
            raise ValueError("No data fetched. Possibly invalid symbol/date.")
        
        df = engineer_features(df)
        features = ['SMA_50', 'EMA_20', 'Volatility', 'RSI_14', 'MACD', 'Bollinger_Upper', 'Bollinger_Lower']
        target = df['Close'].values
        test_size = int(0.2 * len(df))
        train_size = len(df) - test_size
        test_index = df.iloc[train_size:].index

        scaler = StandardScaler()
        x_scaled = scaler.fit_transform(df[features])

        x_train, x_test = x_scaled[:train_size], x_scaled[train_size:]
        y_train, y_test = target[:train_size], target[train_size:]

        # LightGBM with Early Stopping
        try:
            lgb_model = lgb.train({'objective': 'regression', 'learning_rate': 0.01},
                                lgb.Dataset(x_train, label=y_train), 
                                num_boost_round=1000, 
                                early_stopping_rounds=50,  # Fixed typo here
                                valid_sets=[lgb.Dataset(x_test, label=y_test)], 
                                verbose_eval=False)
            lgb_preds = lgb_model.predict(x_test, num_iteration=lgb_model.best_iteration)
        except Exception as e:
            print(f"⚠️ Error in LightGBM: {e}")
            lgb_preds = np.full_like(y_test, np.nan)  # Return NaN if model fails

        # XGBoost with Early Stopping
        try:
            xgb_model = xgb.XGBRegressor(objective='reg:squarederror', learning_rate=0.01, n_estimators=1000)
            xgb_model.fit(x_train, y_train, 
                        eval_set=[(x_test, y_test)], 
                        early_stopping_rounds=50,  # Fixed typo here
                        verbose=False)
            xgb_preds = xgb_model.predict(x_test)
        except Exception as e:
            print(f"⚠️ Error in XGBoost: {e}")
            xgb_preds = np.full_like(y_test, np.nan)

        # Random Forest
        try:
            rf_model = build_rf_model(x_train, y_train)
            rf_preds = rf_model.predict(x_test)
        except Exception as e:
            print(f"⚠️ Error in Random Forest: {e}")
            rf_preds = np.full_like(y_test, np.nan)

        # ARIMA
        try:
            log_close = np.log(df['Close'])
            diff_log = log_close.diff().dropna()
            arima_model = auto_arima(diff_log[:train_size], seasonal=False)
            arima_forecast = arima_model.predict(n_periods=test_size)
            arima_preds = np.exp(np.r_[log_close.iloc[train_size - 1],
                                        log_close.iloc[train_size - 1] + np.cumsum(arima_forecast)][1:])
        except Exception as e:
            print(f"⚠️ Error in ARIMA: {e}")
            arima_preds = np.full_like(y_test, np.nan)

        # LSTM
        try:
            lstm_data = df[['Close'] + features]
            mm_scaler = MinMaxScaler()
            close_scaled = mm_scaler.fit_transform(lstm_data[['Close']])
            feature_scaled = MinMaxScaler().fit_transform(lstm_data[features])
            lstm_scaled = np.hstack([close_scaled, feature_scaled])

            x_lstm, y_lstm = [], []
            for i in range(60, len(lstm_scaled)):
                x_lstm.append(lstm_scaled[i - 60:i])
                y_lstm.append(lstm_scaled[i, 0])
            x_lstm, y_lstm = np.array(x_lstm), np.array(y_lstm)

            lstm_train_size = train_size - 60
            x_train_lstm = x_lstm[:lstm_train_size]
            y_train_lstm = y_lstm[:lstm_train_size]
            x_test_lstm = x_lstm[lstm_train_size:]
            y_test_lstm = y_lstm[lstm_train_size:]

            model_path = f"{symbol}_{start}_{end}_lstm.keras"
            lstm_model = build_lstm_model(x_train_lstm, y_train_lstm, model_path)
            lstm_preds = lstm_model.predict(x_test_lstm)

            # Inverse scale LSTM
            dummy_zeros = np.zeros((len(lstm_preds), len(features)))
            lstm_preds = mm_scaler.inverse_transform(np.concatenate([lstm_preds, dummy_zeros], axis=1))[:, 0]
            y_test_lstm = mm_scaler.inverse_transform(np.concatenate([y_test_lstm.reshape(-1, 1), dummy_zeros], axis=1))[:, 0]
        except Exception as e:
            print(f"⚠️ Error in LSTM: {e}")
            lstm_preds = np.full_like(y_test, np.nan)

        # Handle NaN predictions before plotting
        lgb_preds = np.nan_to_num(lgb_preds, nan=np.nan)
        xgb_preds = np.nan_to_num(xgb_preds, nan=np.nan)
        rf_preds = np.nan_to_num(rf_preds, nan=np.nan)
        arima_preds = np.nan_to_num(arima_preds, nan=np.nan)
        lstm_preds = np.nan_to_num(lstm_preds, nan=np.nan)

        # Plot
        min_len = min(len(test_index), len(lgb_preds), len(xgb_preds), len(rf_preds), len(arima_preds), len(lstm_preds))
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.plot(test_index[:min_len], y_test[:min_len], label='Actual')
        ax.plot(test_index[:min_len], lgb_preds[:min_len], label='LightGBM')
        ax.plot(test_index[:min_len], xgb_preds[:min_len], label='XGBoost')
        ax.plot(test_index[:min_len], rf_preds[:min_len], label='Random Forest')
        ax.plot(test_index[:min_len], arima_preds[:min_len], label='ARIMA')
        ax.plot(test_index[:min_len], lstm_preds[:min_len], label='LSTM', linestyle='--')
        ax.set_title(f"{symbol} Price Forecast")
        ax.legend()

        for widget in plot_frame.winfo_children():
            widget.destroy()

        canvas = FigureCanvasTkAgg(fig, master=plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        metrics_label.config(text=f"RMSE:\n"
                                    f"LSTM: {calculate_rmse(y_test_lstm[:min_len], lstm_preds[:min_len]):.2f} | "
                                    f"ARIMA: {calculate_rmse(y_test[:min_len], arima_preds[:min_len]):.2f} | "
                                    f"RF: {calculate_rmse(y_test[:min_len], rf_preds[:min_len]):.2f} | "
                                    f"XGBoost: {calculate_rmse(y_test[:min_len], xgb_preds[:min_len]):.2f} | "
                                    f"LightGBM: {calculate_rmse(y_test[:min_len], lgb_preds[:min_len]):.2f}")

    except Exception as e:
        messagebox.showerror("Error", str(e))
        print(f"❌ Error: {e}")


# === 🖥️ GUI SETUP ===
root = tk.Tk()
root.title("📈 Stock Forecasting Dashboard")
root.geometry("1200x800")

frame_input = ttk.Frame(root, padding=10)
frame_input.pack()

ttk.Label(frame_input, text="Stock Symbol:").grid(row=0, column=0)
symbol_entry = ttk.Entry(frame_input)
symbol_entry.insert(0, "AAPL")
symbol_entry.grid(row=0, column=1)

ttk.Label(frame_input, text="Start Date:").grid(row=1, column=0)
start_entry = ttk.Entry(frame_input)
start_entry.insert(0, "2014-01-01")
start_entry.grid(row=1, column=1)

ttk.Label(frame_input, text="End Date:").grid(row=2, column=0)
end_entry = ttk.Entry(frame_input)
end_entry.insert(0, "2024-01-01")
end_entry.grid(row=2, column=1)

ttk.Button(frame_input, text="Analyze", command=analyze_stock).grid(row=3, column=0, columnspan=2, pady=10)

metrics_label = ttk.Label(root, text="Model Performance Metrics will appear here.", font=("Arial", 12))
metrics_label.pack(pady=10)

plot_frame = tk.Frame(root, bg="white", relief=tk.SUNKEN, borderwidth=2)
plot_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

root.mainloop()
