import tkinter as tk
from tkinter import ttk, messagebox
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from keras.models import Sequential, load_model
from keras.layers import LSTM, Dropout, Dense, Bidirectional
from keras.callbacks import ReduceLROnPlateau
from keras.optimizers import Adam
from statsmodels.tsa.arima.model import ARIMA
import os

# Function for LSTM preprocessing
def preprocess_lstm(df):
    df['Date'] = df.index
    data = df[['Date', 'Close', 'Volume']].dropna()
    data['SMA_50'] = data['Close'].rolling(window=50).mean()
    data['EMA_20'] = data['Close'].ewm(span=20, adjust=False).mean()
    data.fillna(method='ffill', inplace=True)
    data.dropna(inplace=True)

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data[['Close', 'SMA_50', 'EMA_20']])
    return data, scaled_data, scaler

# Function for ARIMA preprocessing
def preprocess_arima(df):
    df['Close'] = df['Close'].fillna(method='ffill')
    return df

# Function for Random Forest preprocessing
def preprocess_rf(df):
    df['Date'] = df.index
    data = df[['Date', 'Close', 'Volume']].dropna()
    data['SMA_50'] = data['Close'].rolling(window=50).mean()
    data['EMA_20'] = data['Close'].ewm(span=20, adjust=False).mean()
    data['Volatility'] = data['Close'].rolling(window=30).std()
    data.fillna(method='ffill', inplace=True)
    data.dropna(inplace=True)
    return data

# Function to build and train LSTM model
def build_lstm_model(X_train, y_train, model_path):
    if os.path.exists(model_path):
        print("Loading existing LSTM model...")
        lstm_model = load_model(model_path)
    else:
        print("Training new LSTM model...")
        lstm_model = Sequential([
            Bidirectional(LSTM(units=64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2]))),
            Dropout(0.3),
            Bidirectional(LSTM(units=32, return_sequences=False)),
            Dropout(0.3),
            Dense(units=1)
        ])
        lstm_model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
        reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=3, min_lr=1e-5, verbose=1)
        lstm_model.fit(X_train, y_train, epochs=25, batch_size=16, verbose=1, callbacks=[reduce_lr])
        lstm_model.save(model_path)
        print(f"LSTM model saved to {model_path}")
    return lstm_model

# Function to build ARIMA model
def build_arima_model(data):
    train_data = data['Close'][:int(len(data) * 0.8)]
    arima_model = ARIMA(train_data, order=(5, 1, 0)).fit()
    return arima_model

# Function to build Random Forest model
def build_rf_model(X_train_rf, y_train_rf):
    rf_model = RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train_rf, y_train_rf)
    return rf_model

# Function to calculate accuracy
def calculate_accuracy(y_actual, y_pred):
    mean_actual = np.mean(y_actual)
    rmse = np.sqrt(mean_squared_error(y_actual, y_pred))
    accuracy = (1 - (rmse / mean_actual)) * 100
    return rmse, accuracy

# Function to analyze stock data
def analyze_stock():
    try:
        # Get user inputs
        start_date = start_date_entry.get()
        end_date = end_date_entry.get()
        stock_symbol = stock_symbol_entry.get()

        # Validate dates
        datetime.strptime(start_date, '%Y-%m-%d')
        datetime.strptime(end_date, '%Y-%m-%d')

        # Fetch stock data
        df = yf.download(stock_symbol, start=start_date, end=end_date, threads=True)

        if df.empty:
            messagebox.showerror("Error", "No data found for the given stock symbol and date range.")
            return

        # LSTM Preprocessing
        lstm_data, lstm_scaled_data, lstm_scaler = preprocess_lstm(df)
        train_size = int(len(lstm_scaled_data) * 0.8)
        train_data = lstm_scaled_data[:train_size]
        test_data = lstm_scaled_data[train_size:]

        X_train, y_train = [], []
        for i in range(60, len(train_data)):
            X_train.append(train_data[i - 60:i])
            y_train.append(train_data[i, 0])
        X_train, y_train = np.array(X_train), np.array(y_train)

        # Define the model path
        model_path = f"{stock_symbol}_{start_date}_{end_date}_lstm_model.keras"

        # Build and Train LSTM Model (or load if already trained)
        lstm_model = build_lstm_model(X_train, y_train, model_path)
        X_test = np.array([test_data[i - 60:i] for i in range(60, len(test_data))])
        lstm_predictions = lstm_model.predict(X_test)
        lstm_predictions = lstm_scaler.inverse_transform(
            np.concatenate((lstm_predictions, np.zeros((lstm_predictions.shape[0], 2))), axis=1))[:, 0]

        # Plot LSTM training performance
        plt.figure(figsize=(10, 6))
        plt.plot(y_train, label="Training Data", color='blue')
        plt.title("LSTM Training Data")
        plt.legend()
        plt.show()

        # ARIMA Preprocessing
        arima_data = preprocess_arima(df)
        arima_model = build_arima_model(arima_data)
        arima_predictions = arima_model.forecast(steps=len(arima_data) - train_size)

        # Plot ARIMA predictions
        plt.figure(figsize=(10, 6))
        plt.plot(arima_predictions, label="ARIMA Predictions", color='red')
        plt.title("ARIMA Model Predictions")
        plt.legend()
        plt.show()

        # Random Forest Preprocessing
        rf_data = preprocess_rf(df)
        X_rf = rf_data[['SMA_50', 'EMA_20', 'Volatility']].iloc[:train_size]
        y_rf = rf_data['Close'].iloc[:train_size]
        rf_model = build_rf_model(X_rf, y_rf)
        rf_predictions = rf_model.predict(rf_data[['SMA_50', 'EMA_20', 'Volatility']].iloc[train_size:])

        # Plot Random Forest predictions
        plt.figure(figsize=(10, 6))
        plt.plot(rf_predictions, label="Random Forest Predictions", color='green')
        plt.title("Random Forest Model Predictions")
        plt.legend()
        plt.show()

        # Align Predictions and Calculate Metrics
        y_actual = rf_data['Close'].iloc[train_size:]
        min_length = min(len(y_actual), len(lstm_predictions), len(arima_predictions), len(rf_predictions))
        y_actual = y_actual.iloc[:min_length]
        lstm_predictions = lstm_predictions[:min_length]
        arima_predictions = arima_predictions[:min_length]
        rf_predictions = rf_predictions[:min_length]

        lstm_rmse, lstm_accuracy = calculate_accuracy(y_actual, lstm_predictions)
        arima_rmse, arima_accuracy = calculate_accuracy(y_actual, arima_predictions)
        rf_rmse, rf_accuracy = calculate_accuracy(y_actual, rf_predictions)

        # Display Results
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(rf_data['Date'].iloc[train_size:train_size + min_length], y_actual, label='Actual Prices', color='blue')
        ax.plot(rf_data['Date'].iloc[train_size:train_size + min_length], lstm_predictions, label='LSTM Predictions', color='green')
        ax.plot(rf_data['Date'].iloc[train_size:train_size + min_length], arima_predictions, label='ARIMA Predictions', color='red')
        ax.plot(rf_data['Date'].iloc[train_size:train_size + min_length], rf_predictions, label='RF Predictions', color='orange')
        ax.set_title(f'{stock_symbol} Model Predictions vs Actual Prices')
        ax.set_xlabel('Date')
        ax.set_ylabel('Price')
        ax.legend()
        canvas = FigureCanvasTkAgg(fig, master=plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Update Metrics
        lstm_label.config(text=f"LSTM: RMSE = {lstm_rmse:.2f} | Accuracy = {lstm_accuracy:.2f}%")
        rf_label.config(text=f"Random Forest: RMSE = {rf_rmse:.2f} | Accuracy = {rf_accuracy:.2f}%")
        arima_label.config(text=f"ARIMA: RMSE = {arima_rmse:.2f} | Accuracy = {arima_accuracy:.2f}%")

    except Exception as e:
        messagebox.showerror("Error", str(e))

# GUI Setup
root = tk.Tk()
root.title("Stock Price Prediction")
root.geometry("1000x800")

# Input Frame
input_frame = ttk.Frame(root, padding=10)
input_frame.pack()

ttk.Label(input_frame, text="Start Date (YYYY-MM-DD):").grid(row=0, column=0, sticky="w")
start_date_entry = ttk.Entry(input_frame)
start_date_entry.grid(row=0, column=1, padx=10)

ttk.Label(input_frame, text="End Date (YYYY-MM-DD):").grid(row=1, column=0, sticky="w")
end_date_entry = ttk.Entry(input_frame)
end_date_entry.grid(row=1, column=1, padx=10)

ttk.Label(input_frame, text="Stock Symbol:").grid(row=2, column=0, sticky="w")
stock_symbol_entry = ttk.Entry(input_frame)
stock_symbol_entry.grid(row=2, column=1, padx=10)

analyze_button = ttk.Button(input_frame, text="Analyze", command=analyze_stock)
analyze_button.grid(row=3, column=0, columnspan=2, pady=10)

# Metrics Frame
metrics_frame = ttk.Frame(root, padding=10)
metrics_frame.pack()

lstm_label = ttk.Label(metrics_frame, text="LSTM: RMSE = N/A | Accuracy = N/A%", font=("Arial", 12))
lstm_label.pack()
rf_label = ttk.Label(metrics_frame, text="Random Forest: RMSE = N/A | Accuracy = N/A%", font=("Arial", 12))
rf_label.pack()
arima_label = ttk.Label(metrics_frame, text="ARIMA: RMSE = N/A | Accuracy = N/A%", font=("Arial", 12))
arima_label.pack()

# Plot Frame
plot_frame = tk.Frame(root, bg="white", relief=tk.SUNKEN, borderwidth=2)
plot_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

# Run the Application
root.mainloop()
