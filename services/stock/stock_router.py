from fastapi import FastAPI, Query, HTTPException, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import pandas as pd
import yfinance as yf
import joblib
import os
import requests
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from functools import lru_cache
import time
import logging
from typing import Dict, List, Optional, Tuple, Union
import threading
import numpy as np
import os
from dotenv import load_dotenv

# 🌍 Load environment variables from .env
load_dotenv()

# 📝 Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 🛑 Track persistent failures
persistent_failures = set()

# ✅ Required Environment Keys
required_keys = [
    "ALPHA_VANTAGE_API_KEY",
    "TWELVE_DATA_API_KEY",
    "MARKETSTACK_API_KEY",
    "MEDIASTACK_API_KEY",
    "MONGO_URL"
]

# 🔍 Check for missing critical env vars
missing_keys = [key for key in required_keys if not os.getenv(key)]
if missing_keys:
    logger.critical(f"❌ Missing environment variables: {', '.join(missing_keys)}")
    raise RuntimeError("Missing critical environment variables")

# ✅ API Keys Dictionary
APIS = {
    "alpha_vantage": os.getenv("ALPHA_VANTAGE_API_KEY"),
    "marketstack": os.getenv("MARKETSTACK_API_KEY"),
    "twelve_data": os.getenv("TWELVE_DATA_API_KEY"),
    "mediastack": os.getenv("MEDIASTACK_API_KEY"),
    "finnhub": os.getenv("FINNHUB_API_KEY"),
    "binance_api": os.getenv("BINANCE_API_KEY"),
    "binance_secret": os.getenv("BINANCE_SECRET_KEY"),
    "huggingface": os.getenv("HUGGINGFACEHUB_API_TOKEN"),
    "gemini": os.getenv("GEMINI_API_KEY")
}


# === Setup ===
app = FastAPI(
    title="Enhanced Stock Analysis API",
    description="Comprehensive API for stock market data analysis, predictions, and technical indicators",
    version="2.0.0"
)

router = APIRouter(prefix="/api/stocks", tags=["Stocks"])
app.include_router(router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Configuration ===
MODEL_DIR = "checkpoints"
os.makedirs(MODEL_DIR, exist_ok=True)

# Expanded list of supported symbols
symbols = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", 
    "TSLA", "NVDA", "JPM", "V", "JNJ",
    "WMT", "UNH", "PG", "DIS", "BAC",
    "XOM", "HD", "INTC", "BRK-B", "CSCO",
    "PEP", "KO", "ABT", "T", "ABBV"
]

# === Enhanced Caching Setup ===
@lru_cache(maxsize=1024)
def cached_request(url: str) -> Optional[requests.Response]:
    """Cache API requests with rate limit handling"""
    try:
        logger.info(f"Fetching: {url.split('?')[0]}")
        response = requests.get(url, timeout=15)
        
        if response.status_code == 200:
            return response
        elif response.status_code == 429:  # Rate limited
            logger.warning(f"Rate limited: {url.split('?')[0]}")
            time.sleep(30)  # Wait before retry
            return cached_request(url)
        else:
            logger.warning(f"API request failed: {response.status_code}")
            return None
            
    except Exception as e:
        logger.error(f"Request failed: {str(e)}")
        return None

# Cache for models and data
model_cache: Dict[str, RandomForestRegressor] = {}
scaler_cache: Dict[str, MinMaxScaler] = {}
prediction_cache: Dict[str, List[Tuple[str, float]]] = {}
historical_cache: Dict[str, Dict[str, List[Dict]]] = {}

def clear_cache_periodically(interval: int = 1800):
    """Periodically clear cache to free memory"""
    while True:
        time.sleep(interval)
        model_cache.clear()
        scaler_cache.clear()
        prediction_cache.clear()
        historical_cache.clear()
        cached_request.cache_clear()
        logger.info("Cache cleared")

# Start cache clearing thread
cache_thread = threading.Thread(target=clear_cache_periodically, daemon=True)
cache_thread.start()

# === Helper Functions ===
def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate various technical indicators for stock data"""
    if df.empty:
        return df
    
    # Moving Averages
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['SMA_200'] = df['Close'].rolling(window=200).mean()
    
    # Exponential Moving Averages
    df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
    
    # MACD
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
    
    # RSI
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Bollinger Bands
    df['BB_MA'] = df['Close'].rolling(window=20).mean()
    df['BB_UPPER'] = df['BB_MA'] + 2 * df['Close'].rolling(window=20).std()
    df['BB_LOWER'] = df['BB_MA'] - 2 * df['Close'].rolling(window=20).std()
    
    # Volume Weighted Average Price
    df['VWAP'] = (df['Volume'] * (df['High'] + df['Low'] + df['Close']) / 3).cumsum() / df['Volume'].cumsum()
    
    return df.dropna()

def parse_date(date_str: str) -> Optional[datetime]:
    """Robust date parsing with multiple formats"""
    if isinstance(date_str, datetime):
        return date_str
        
    if date_str.isdigit() and len(date_str) == 10:  # Unix timestamp
        return datetime.fromtimestamp(int(date_str))
        
    formats = [
        "%Y-%m-%d",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M:%S%z",
        "%m/%d/%Y",
        "%d-%m-%Y"
    ]
    
    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    return None

def fetch_from_yfinance(
    symbol: str,
    start_dt: datetime,
    end_dt: datetime,
    interval: str = "1d"
) -> Optional[pd.DataFrame]:
    """Fetch data from Yahoo Finance with enhanced error handling"""
    try:
        yf_symbol = symbol.replace("-", ".")
        df = yf.download(
            yf_symbol,
            start=start_dt,
            end=end_dt,
            interval=interval,
            progress=False
        )
        
        if not df.empty:
            df = df.reset_index()
            df['Date'] = pd.to_datetime(df['Date']).dt.strftime("%Y-%m-%d")
            return df
            
    except Exception as e:
        logger.error(f"Yahoo Finance error for {symbol}: {str(e)}")
        time.sleep(5)  # Add delay after failure
    return None

def fetch_from_twelve_data(
    symbol: str,
    start_dt: datetime,
    end_dt: datetime,
    interval: str = "1day"
) -> Optional[pd.DataFrame]:
    """Fetch data from Twelve Data API with improved parsing"""
    if not APIS["twelve_data"]:
        return None

    url = (
        f"https://api.twelvedata.com/time_series?symbol={symbol}&interval={interval}"
        f"&start_date={start_dt.strftime('%Y-%m-%d')}"
        f"&end_date={end_dt.strftime('%Y-%m-%d')}&apikey={APIS['twelve_data']}"
    )
    res = cached_request(url)
    
    if res and res.status_code == 200:
        try:
            data = res.json()
            if "values" in data:
                df = pd.DataFrame(data["values"])
                
                # Handle different column naming conventions
                col_map = {
                    'datetime': 'Date',
                    'open': 'Open',
                    'high': 'High',
                    'low': 'Low',
                    'close': 'Close',
                    'volume': 'Volume'
                }
                
                df = df.rename(columns={k:v for k,v in col_map.items() if k in df.columns})
                
                if 'Date' not in df.columns and 'datetime' in df.columns:
                    df['Date'] = df['datetime']
                
                df["Date"] = pd.to_datetime(df["Date"])
                numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
                
                for col in numeric_cols:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                
                df = df[(df["Date"] >= start_dt) & (df["Date"] <= end_dt)]
                if not df.empty:
                    df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")
                    return df[["Date", "Open", "High", "Low", "Close", "Volume"]].sort_values("Date")
                    
        except Exception as e:
            logger.error(f"Twelve Data parse error: {str(e)}")
    return None

def fetch_stock_data(
    symbol: str, 
    start: str, 
    end: str, 
    interval: str = "1day",
    outputsize: str = "compact"
) -> Optional[pd.DataFrame]:
    """
    Enhanced stock data fetcher with multiple fallback sources
    Returns DataFrame with OHLCV data and technical indicators
    """
    try:
        cache_key = f"{symbol}_{start}_{end}_{interval}"
        if cache_key in historical_cache:
            return pd.DataFrame(historical_cache[cache_key])
        
        start_dt = parse_date(start)
        end_dt = parse_date(end)
        
        if not start_dt or not end_dt:
            logger.error(f"Invalid date format: start={start}, end={end}")
            return None

        # Try data sources with delays
        df = None
        sources = [
            lambda: fetch_from_yfinance(symbol, start_dt, end_dt, interval),
            lambda: fetch_from_twelve_data(symbol, start_dt, end_dt, interval),
            lambda: fetch_from_alpha_vantage(symbol, start_dt, end_dt, interval, outputsize),
        ]
        
        if interval == "1day":
            sources.append(lambda: fetch_from_marketstack(symbol, start_dt, end_dt))
            
        for source in sources:
            df = source()
            if df is not None and not df.empty:
                break
            time.sleep(2)  # Delay between sources

        if df is None or df.empty:
            logger.error(f"All data sources failed for {symbol} from {start} to {end}")
            return None

        # Calculate technical indicators
        df = calculate_technical_indicators(df)
        
        # Cache the result
        historical_cache[cache_key] = df.to_dict(orient="records")
        
        return df
    except Exception as e:
        logger.exception("Data fetch error")
        return None

def train_and_save_model(symbol: str) -> bool:
    """Train and save an enhanced Random Forest model for the given symbol"""
    try:
        logger.info(f"Training enhanced model for {symbol}")
        
        # Fetch historical data with technical indicators
        df = fetch_stock_data(
            symbol, 
            "2014-01-01", 
            datetime.today().strftime("%Y-%m-%d"),
            interval="1day",
            outputsize="full"
        )
        
        if df is None or df.empty:
            logger.warning(f"No data available for {symbol}")
            return False

        # Prepare features and target
        features = df[[
            'SMA_20', 'SMA_50', 'SMA_200',
            'EMA_12', 'EMA_26',
            'MACD', 'MACD_Signal', 'MACD_Hist',
            'RSI',
            'BB_MA', 'BB_UPPER', 'BB_LOWER',
            'VWAP'
        ]].values
        
        target = df['Close'].values
        
        # Scale features
        scaler = MinMaxScaler()
        scaled_features = scaler.fit_transform(features)
        
        # Train model with enhanced parameters
        model = RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        model.fit(scaled_features, target)

        # Save artifacts
        model_path = os.path.join(MODEL_DIR, f"{symbol}_rf_model.pkl")
        scaler_path = os.path.join(MODEL_DIR, f"{symbol}_scaler.pkl")
        
        joblib.dump(model, model_path)
        joblib.dump(scaler, scaler_path)

        # Generate future predictions (60 days)
        future_dates = pd.date_range(datetime.today(), periods=60)
        last_row = df.iloc[-1]
        
        # Initialize with last known values
        preds = []
        current_features = last_row[[
            'SMA_20', 'SMA_50', 'SMA_200',
            'EMA_12', 'EMA_26',
            'MACD', 'MACD_Signal', 'MACD_Hist',
            'RSI',
            'BB_MA', 'BB_UPPER', 'BB_LOWER',
            'VWAP'
        ]].values.reshape(1, -1)
        
        for date in future_dates:
            # Scale features and predict
            scaled_input = scaler.transform(current_features)
            pred_scaled = model.predict(scaled_input)[0]
            
            # Inverse transform to get actual price
            close_pred = pred_scaled  # Since we didn't scale target
            
            preds.append((date.strftime('%Y-%m-%d'), close_pred))
            
            # Update indicators for next prediction
            current_features[0][0] = (current_features[0][0] * 19 + close_pred) / 20  # SMA_20
            current_features[0][1] = (current_features[0][1] * 49 + close_pred) / 50  # SMA_50
            current_features[0][2] = (current_features[0][2] * 199 + close_pred) / 200  # SMA_200
            current_features[0][3] = (current_features[0][3] * 11 + close_pred) / 12  # EMA_12
            current_features[0][4] = (current_features[0][4] * 25 + close_pred) / 26  # EMA_26

        # Save predictions
        pred_path = os.path.join(MODEL_DIR, f"{symbol}_future_predictions.csv")
        pd.DataFrame(preds, columns=["Date", "Predicted Close"]).to_csv(pred_path, index=False)

        # Update cache
        model_cache[symbol] = model
        scaler_cache[symbol] = scaler
        prediction_cache[symbol] = preds

        logger.info(f"Successfully trained enhanced model for {symbol}")
        return True
    except Exception as e:
        logger.exception(f"Error training model for {symbol}")
        persistent_failures.add(symbol)
        return False

# === API Endpoints ===
@router.get("/predict")
def predict(
    symbol: str = Query("AAPL", description="Stock ticker symbol"),
    days: int = Query(15, ge=1, le=60, description="Number of days to predict")
):
    """
    Get enhanced stock price predictions with technical analysis metrics.
    Returns predicted prices, returns, and technical indicators.
    """
    try:
        # Get from cache or disk
        model = get_cached_model(symbol)
        scaler = get_cached_scaler(symbol)
        preds = get_cached_predictions(symbol)
        
        if not all([model, scaler, preds]):
            raise HTTPException(
                status_code=404,
                detail=f"Model not trained for symbol {symbol}. Available symbols: {', '.join(symbols)}"
            )
        
        # Get historical data for technical indicators
        hist_data = fetch_stock_data(
            symbol,
            (datetime.today() - timedelta(days=365)).strftime("%Y-%m-%d"),
            datetime.today().strftime("%Y-%m-%d")
        )
        
        if hist_data is None:
            raise HTTPException(
                status_code=404,
                detail=f"Could not fetch historical data for {symbol}"
            )
        
        # Get predictions
        dates, prices = zip(*preds[:days])
        latest_close = hist_data['Close'].iloc[-1]
        final_price = prices[-1]
        predicted_return_pct = ((final_price - latest_close) / latest_close) * 100

        # Get latest technical indicators
        latest_tech = hist_data.iloc[-1][[
            'SMA_20', 'SMA_50', 'SMA_200',
            'EMA_12', 'EMA_26',
            'MACD', 'MACD_Signal', 'MACD_Hist',
            'RSI',
            'BB_MA', 'BB_UPPER', 'BB_LOWER',
            'VWAP'
        ]].to_dict()

        return {
            "symbol": symbol,
            "days": days,
            "latest_close": round(latest_close, 2),
            "final_prediction": round(final_price, 2),
            "predicted_return_pct": round(predicted_return_pct, 2),
            "dates": dates,
            "predicted": [round(p, 2) for p in prices],
            "technical_indicators": {
                k: round(v, 4) for k, v in latest_tech.items()
            },
            "model_metrics": {
                "n_estimators": model.n_estimators,
                "max_depth": model.max_depth,
                "min_samples_split": model.min_samples_split,
                "features": list(latest_tech.keys())
            }
        }
    except Exception as e:
        logger.exception("Prediction error")
        raise HTTPException(
            status_code=500,
            detail=f"Prediction error: {str(e)}"
        )

from fastapi import FastAPI, Query, HTTPException, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import pandas as pd
import yfinance as yf
import joblib
import os
import requests
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler
from functools import lru_cache
import time
import logging
from typing import Dict, List, Optional, Tuple, Union
import threading
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Track persistent failures
persistent_failures = set()

# === API KEYS ===
APIS = {
    "alpha_vantage": os.getenv("ALPHA_VANTAGE_API_KEY", "demo"),
    "marketstack": os.getenv("MARKETSTACK_API_KEY"),
    "twelve_data": os.getenv("TWELVE_DATA_API_KEY", "demo"),
    "mediastack": os.getenv("MEDIASTACK_API_KEY"),
}

# === Setup ===
app = FastAPI(
    title="Enhanced Stock Analysis API",
    description="Comprehensive API for stock market data analysis, predictions, and technical indicators",
    version="2.0.0"
)

router = APIRouter(prefix="/api/stocks", tags=["Stocks"])
app.include_router(router)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Configuration ===
MODEL_DIR = "checkpoints"
os.makedirs(MODEL_DIR, exist_ok=True)

# Expanded list of supported symbols
symbols = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", 
    "TSLA", "NVDA", "JPM", "V", "JNJ",
    "WMT", "UNH", "PG", "DIS", "BAC",
    "XOM", "HD", "INTC", "BRK-B", "CSCO",
    "PEP", "KO", "ABT", "T", "ABBV"
]

# === Enhanced Caching Setup ===
@lru_cache(maxsize=1024)
def cached_request(url: str) -> Optional[requests.Response]:
    """Cache API requests with rate limit handling"""
    try:
        logger.info(f"Fetching: {url.split('?')[0]}")
        response = requests.get(url, timeout=15)
        
        if response.status_code == 200:
            return response
        elif response.status_code == 429:  # Rate limited
            logger.warning(f"Rate limited: {url.split('?')[0]}")
            time.sleep(30)  # Wait before retry
            return cached_request(url)
        else:
            logger.warning(f"API request failed: {response.status_code}")
            return None
            
    except Exception as e:
        logger.error(f"Request failed: {str(e)}")
        return None

# Cache for models and data
model_cache: Dict[str, RandomForestRegressor] = {}
scaler_cache: Dict[str, MinMaxScaler] = {}
prediction_cache: Dict[str, List[Tuple[str, float]]] = {}
historical_cache: Dict[str, Dict[str, List[Dict]]] = {}

def clear_cache_periodically(interval: int = 1800):
    """Periodically clear cache to free memory"""
    while True:
        time.sleep(interval)
        model_cache.clear()
        scaler_cache.clear()
        prediction_cache.clear()
        historical_cache.clear()
        cached_request.cache_clear()
        logger.info("Cache cleared")

# Start cache clearing thread
cache_thread = threading.Thread(target=clear_cache_periodically, daemon=True)
cache_thread.start()

# === Helper Functions ===
def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate various technical indicators for stock data"""
    if df.empty:
        return df
    
    # Moving Averages
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['SMA_200'] = df['Close'].rolling(window=200).mean()
    
    # Exponential Moving Averages
    df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()
    
    # MACD
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
    
    # RSI
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Bollinger Bands
    df['BB_MA'] = df['Close'].rolling(window=20).mean()
    df['BB_UPPER'] = df['BB_MA'] + 2 * df['Close'].rolling(window=20).std()
    df['BB_LOWER'] = df['BB_MA'] - 2 * df['Close'].rolling(window=20).std()
    
    # Volume Weighted Average Price
    df['VWAP'] = (df['Volume'] * (df['High'] + df['Low'] + df['Close']) / 3).cumsum() / df['Volume'].cumsum()
    
    return df.dropna()

def parse_date(date_str: str) -> Optional[datetime]:
    """Robust date parsing with multiple formats"""
    if isinstance(date_str, datetime):
        return date_str
        
    if date_str.isdigit() and len(date_str) == 10:  # Unix timestamp
        return datetime.fromtimestamp(int(date_str))
        
    formats = [
        "%Y-%m-%d",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M:%S%z",
        "%m/%d/%Y",
        "%d-%m-%Y"
    ]
    
    for fmt in formats:
        try:
            return datetime.strptime(date_str, fmt)
        except ValueError:
            continue
    return None

def fetch_from_yfinance(
    symbol: str,
    start_dt: datetime,
    end_dt: datetime,
    interval: str = "1d"
) -> Optional[pd.DataFrame]:
    """Fetch data from Yahoo Finance with enhanced error handling"""
    try:
        yf_symbol = symbol.replace("-", ".")
        df = yf.download(
            yf_symbol,
            start=start_dt,
            end=end_dt,
            interval=interval,
            progress=False
        )
        
        if not df.empty:
            df = df.reset_index()
            df['Date'] = pd.to_datetime(df['Date']).dt.strftime("%Y-%m-%d")
            return df
            
    except Exception as e:
        logger.error(f"Yahoo Finance error for {symbol}: {str(e)}")
        time.sleep(5)  # Add delay after failure
    return None

def fetch_from_twelve_data(
    symbol: str,
    start_dt: datetime,
    end_dt: datetime,
    interval: str = "1day"
) -> Optional[pd.DataFrame]:
    """Fetch data from Twelve Data API with improved parsing"""
    if not APIS["twelve_data"]:
        return None

    url = (
        f"https://api.twelvedata.com/time_series?symbol={symbol}&interval={interval}"
        f"&start_date={start_dt.strftime('%Y-%m-%d')}"
        f"&end_date={end_dt.strftime('%Y-%m-%d')}&apikey={APIS['twelve_data']}"
    )
    res = cached_request(url)
    
    if res and res.status_code == 200:
        try:
            data = res.json()
            if "values" in data:
                df = pd.DataFrame(data["values"])
                
                # Handle different column naming conventions
                col_map = {
                    'datetime': 'Date',
                    'open': 'Open',
                    'high': 'High',
                    'low': 'Low',
                    'close': 'Close',
                    'volume': 'Volume'
                }
                
                df = df.rename(columns={k:v for k,v in col_map.items() if k in df.columns})
                
                if 'Date' not in df.columns and 'datetime' in df.columns:
                    df['Date'] = df['datetime']
                
                df["Date"] = pd.to_datetime(df["Date"])
                numeric_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
                
                for col in numeric_cols:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                
                df = df[(df["Date"] >= start_dt) & (df["Date"] <= end_dt)]
                if not df.empty:
                    df["Date"] = df["Date"].dt.strftime("%Y-%m-%d")
                    return df[["Date", "Open", "High", "Low", "Close", "Volume"]].sort_values("Date")
                    
        except Exception as e:
            logger.error(f"Twelve Data parse error: {str(e)}")
    return None

def fetch_stock_data(
    symbol: str, 
    start: str, 
    end: str, 
    interval: str = "1day",
    outputsize: str = "compact"
) -> Optional[pd.DataFrame]:
    """
    Enhanced stock data fetcher with multiple fallback sources
    Returns DataFrame with OHLCV data and technical indicators
    """
    try:
        cache_key = f"{symbol}_{start}_{end}_{interval}"
        if cache_key in historical_cache:
            return pd.DataFrame(historical_cache[cache_key])
        
        start_dt = parse_date(start)
        end_dt = parse_date(end)
        
        if not start_dt or not end_dt:
            logger.error(f"Invalid date format: start={start}, end={end}")
            return None

        # Try data sources with delays
        df = None
        sources = [
            lambda: fetch_from_yfinance(symbol, start_dt, end_dt, interval),
            lambda: fetch_from_twelve_data(symbol, start_dt, end_dt, interval),
            lambda: fetch_from_alpha_vantage(symbol, start_dt, end_dt, interval, outputsize),
        ]
        
        if interval == "1day":
            sources.append(lambda: fetch_from_marketstack(symbol, start_dt, end_dt))
            
        for source in sources:
            df = source()
            if df is not None and not df.empty:
                break
            time.sleep(2)  # Delay between sources

        if df is None or df.empty:
            logger.error(f"All data sources failed for {symbol} from {start} to {end}")
            return None

        # Calculate technical indicators
        df = calculate_technical_indicators(df)
        
        # Cache the result
        historical_cache[cache_key] = df.to_dict(orient="records")
        
        return df
    except Exception as e:
        logger.exception("Data fetch error")
        return None

def train_and_save_model(symbol: str) -> bool:
    """Train and save an enhanced Random Forest model for the given symbol"""
    try:
        logger.info(f"Training enhanced model for {symbol}")
        
        # Fetch historical data with technical indicators
        df = fetch_stock_data(
            symbol, 
            "2014-01-01", 
            datetime.today().strftime("%Y-%m-%d"),
            interval="1day",
            outputsize="full"
        )
        
        if df is None or df.empty:
            logger.warning(f"No data available for {symbol}")
            return False

        # Prepare features and target
        features = df[[
            'SMA_20', 'SMA_50', 'SMA_200',
            'EMA_12', 'EMA_26',
            'MACD', 'MACD_Signal', 'MACD_Hist',
            'RSI',
            'BB_MA', 'BB_UPPER', 'BB_LOWER',
            'VWAP'
        ]].values
        
        target = df['Close'].values
        
        # Scale features
        scaler = MinMaxScaler()
        scaled_features = scaler.fit_transform(features)
        
        # Train model with enhanced parameters
        model = RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        model.fit(scaled_features, target)

        # Save artifacts
        model_path = os.path.join(MODEL_DIR, f"{symbol}_rf_model.pkl")
        scaler_path = os.path.join(MODEL_DIR, f"{symbol}_scaler.pkl")
        
        joblib.dump(model, model_path)
        joblib.dump(scaler, scaler_path)

        # Generate future predictions (60 days)
        future_dates = pd.date_range(datetime.today(), periods=60)
        last_row = df.iloc[-1]
        
        # Initialize with last known values
        preds = []
        current_features = last_row[[
            'SMA_20', 'SMA_50', 'SMA_200',
            'EMA_12', 'EMA_26',
            'MACD', 'MACD_Signal', 'MACD_Hist',
            'RSI',
            'BB_MA', 'BB_UPPER', 'BB_LOWER',
            'VWAP'
        ]].values.reshape(1, -1)
        
        for date in future_dates:
            # Scale features and predict
            scaled_input = scaler.transform(current_features)
            pred_scaled = model.predict(scaled_input)[0]
            
            # Inverse transform to get actual price
            close_pred = pred_scaled  # Since we didn't scale target
            
            preds.append((date.strftime('%Y-%m-%d'), close_pred))
            
            # Update indicators for next prediction
            current_features[0][0] = (current_features[0][0] * 19 + close_pred) / 20  # SMA_20
            current_features[0][1] = (current_features[0][1] * 49 + close_pred) / 50  # SMA_50
            current_features[0][2] = (current_features[0][2] * 199 + close_pred) / 200  # SMA_200
            current_features[0][3] = (current_features[0][3] * 11 + close_pred) / 12  # EMA_12
            current_features[0][4] = (current_features[0][4] * 25 + close_pred) / 26  # EMA_26

        # Save predictions
        pred_path = os.path.join(MODEL_DIR, f"{symbol}_future_predictions.csv")
        pd.DataFrame(preds, columns=["Date", "Predicted Close"]).to_csv(pred_path, index=False)

        # Update cache
        model_cache[symbol] = model
        scaler_cache[symbol] = scaler
        prediction_cache[symbol] = preds

        logger.info(f"Successfully trained enhanced model for {symbol}")
        return True
    except Exception as e:
        logger.exception(f"Error training model for {symbol}")
        persistent_failures.add(symbol)
        return False

# === API Endpoints ===
@router.get("/predict")
def predict(
    symbol: str = Query("AAPL", description="Stock ticker symbol"),
    days: int = Query(15, ge=1, le=60, description="Number of days to predict")
):
    """
    Get enhanced stock price predictions with technical analysis metrics.
    Returns predicted prices, returns, and technical indicators.
    """
    try:
        # Get from cache or disk
        model = get_cached_model(symbol)
        scaler = get_cached_scaler(symbol)
        preds = get_cached_predictions(symbol)
        
        if not all([model, scaler, preds]):
            raise HTTPException(
                status_code=404,
                detail=f"Model not trained for symbol {symbol}. Available symbols: {', '.join(symbols)}"
            )
        
        # Get historical data for technical indicators
        hist_data = fetch_stock_data(
            symbol,
            (datetime.today() - timedelta(days=365)).strftime("%Y-%m-%d"),
            datetime.today().strftime("%Y-%m-%d")
        )
        
        if hist_data is None:
            raise HTTPException(
                status_code=404,
                detail=f"Could not fetch historical data for {symbol}"
            )
        
        # Get predictions
        dates, prices = zip(*preds[:days])
        latest_close = hist_data['Close'].iloc[-1]
        final_price = prices[-1]
        predicted_return_pct = ((final_price - latest_close) / latest_close) * 100

        # Get latest technical indicators
        latest_tech = hist_data.iloc[-1][[
            'SMA_20', 'SMA_50', 'SMA_200',
            'EMA_12', 'EMA_26',
            'MACD', 'MACD_Signal', 'MACD_Hist',
            'RSI',
            'BB_MA', 'BB_UPPER', 'BB_LOWER',
            'VWAP'
        ]].to_dict()

        return {
            "symbol": symbol,
            "days": days,
            "latest_close": round(latest_close, 2),
            "final_prediction": round(final_price, 2),
            "predicted_return_pct": round(predicted_return_pct, 2),
            "dates": dates,
            "predicted": [round(p, 2) for p in prices],
            "technical_indicators": {
                k: round(v, 4) for k, v in latest_tech.items()
            },
            "model_metrics": {
                "n_estimators": model.n_estimators,
                "max_depth": model.max_depth,
                "min_samples_split": model.min_samples_split,
                "features": list(latest_tech.keys())
            }
        }
    except Exception as e:
        logger.exception("Prediction error")
        raise HTTPException(
            status_code=500,
            detail=f"Prediction error: {str(e)}"
        )

# ... (keep other endpoints like historical, metrics, health the same as before)

# === Initialize Models ===
def initialize_models():
    """Train models for all symbols with rate limit handling"""
    logger.info("⚙️ Initializing enhanced stock prediction models...")
    
    # Validate environment
    required_env_vars = ["ALPHA_VANTAGE_API_KEY", "MONGO_URL"]
    missing = [var for var in required_env_vars if not os.getenv(var)]
    if missing:
        logger.critical(f"Missing environment variables: {', '.join(missing)}")
        raise RuntimeError("Missing critical environment variables")
    
    for i, symbol in enumerate(symbols):
        try:
            if symbol in persistent_failures:
                continue
                
            model_path = os.path.join(MODEL_DIR, f"{symbol}_rf_model.pkl")
            pred_path = os.path.join(MODEL_DIR, f"{symbol}_future_predictions.csv")
            
            # Skip if already exists and recent
            if os.path.exists(model_path) and os.path.exists(pred_path):
                model_age = (datetime.now() - datetime.fromtimestamp(os.path.getmtime(model_path))).days
                if model_age <= 7:  # Only retrain weekly
                    get_cached_model(symbol)
                    get_cached_scaler(symbol)
                    get_cached_predictions(symbol)
                    continue
                    
            # Train with delay
            if i > 0:
                delay = 30 if i % 5 == 0 else 10  # Longer delay every 5 symbols
                logger.info(f"Waiting {delay} seconds before next symbol...")
                time.sleep(delay)
                
            success = train_and_save_model(symbol)
            if not success:
                persistent_failures.add(symbol)
                
        except Exception as e:
            logger.error(f"Failed to initialize model for {symbol}: {str(e)}")
            persistent_failures.add(symbol)
            
    logger.info("✅ Model initialization complete")

# Run model initialization when the app starts
initialize_models()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
# === Initialize Models ===
def initialize_models():
    """Train models for all symbols with rate limit handling"""
    logger.info("⚙️ Initializing enhanced stock prediction models...")
    
    # Validate environment
    required_env_vars = ["ALPHA_VANTAGE_API_KEY"]
    missing = [var for var in required_env_vars if not os.getenv(var)]
    if missing:
        logger.critical(f"Missing environment variables: {', '.join(missing)}")
        raise RuntimeError("Missing critical environment variables")
    
    for i, symbol in enumerate(symbols):
        try:
            if symbol in persistent_failures:
                logger.info(f"Skipping {symbol} due to previous failures")
                continue
                
            model_path = os.path.join(MODEL_DIR, f"{symbol}_rf_model.pkl")
            pred_path = os.path.join(MODEL_DIR, f"{symbol}_future_predictions.csv")
            
            # Skip if model exists and is recent
            if os.path.exists(model_path) and os.path.exists(pred_path):
                model_age = (datetime.now() - datetime.fromtimestamp(os.path.getmtime(model_path))).days
                if model_age <= 7:
                    logger.info(f"Using cached model for {symbol}")
                    get_cached_model(symbol)
                    get_cached_scaler(symbol)
                    get_cached_predictions(symbol)
                    continue
                    
            # Add API rate limiting
            if i > 0:
                delay = 15  # Seconds between API calls
                logger.info(f"Rate limit protection: Waiting {delay} seconds...")
                time.sleep(delay)
                
            success = train_and_save_model(symbol)
            if success:
                logger.info(f"✅ Successfully trained model for {symbol}")
            else:
                persistent_failures.add(symbol)
                logger.warning(f"❌ Failed to train model for {symbol}")
                
        except Exception as e:
            logger.error(f"Failed to initialize model for {symbol}: {str(e)}")
            persistent_failures.add(symbol)
            
    logger.info("✅ Model initialization complete")
# Run model initialization when the app starts
initialize_models()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
    
    