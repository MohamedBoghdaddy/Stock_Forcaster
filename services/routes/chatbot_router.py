from fastapi import FastAPI, APIRouter, Request, HTTPException, Body, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import google.generativeai as genai
import os
import json
import logging
import hashlib
import asyncio
import pandas as pd
import httpx
import time
from cachetools import TTLCache
from dotenv import load_dotenv
from scipy.stats import linregress
import joblib
import numpy as np
from datetime import datetime, timedelta

# Load environment variables
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("❌ GEMINI_API_KEY not set in environment")

router = APIRouter(prefix="/chatbot", tags=["Chatbot"])

# === Configure logging ===
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# === Create main app ===
app = FastAPI()

# CORS configuration
origins = ["http://localhost:3000"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Gemini Configuration ===
try:
    genai.configure(api_key=GEMINI_API_KEY)
    gemini_model = genai.GenerativeModel("gemini-1.5-flash")
    logger.info("✅ Gemini chatbot model initialized")
except Exception as e:
    logger.error(f"❌ Gemini initialization failed: {str(e)}")
    raise RuntimeError(f"Gemini initialization failed: {str(e)}")

# === Global Caches ===
RESPONSE_CACHE = TTLCache(maxsize=1000, ttl=300)  # 5-minute cache
STOCK_DATA_CACHE = TTLCache(maxsize=200, ttl=3600)  # 1-hour cache
SESSION_HISTORY = TTLCache(maxsize=1000, ttl=3600)  # 1-hour session cache
SESSION_LOCK = asyncio.Lock()

# === Stock Knowledge Base ===
RISK_CATEGORIES = {
    "conservative": {
        "stocks": ["JNJ", "PG", "KO", "PEP", "WMT", "MCD", "T", "VZ", "SO", "DUK"],
        "description": "stable blue-chip companies with consistent dividends",
        "max_volatility": 0.3,
        "allocation": "15-25% of portfolio"
    },
    "balanced": {
        "stocks": ["MSFT", "AAPL", "V", "MA", "DIS", "HD", "LOW", "COST", "JPM", "BAC"],
        "description": "balanced growth stocks with moderate volatility",
        "max_volatility": 0.5,
        "allocation": "25-40% of portfolio"
    },
    "growth": {
        "stocks": ["GOOGL", "AMZN", "TSLA", "META", "NFLX", "ADBE", "CRM", "PYPL", "AVGO", "QCOM"],
        "description": "high-growth potential stocks",
        "max_volatility": 0.8,
        "allocation": "20-35% of portfolio"
    },
    "aggressive": {
        "stocks": ["NVDA", "AMD", "SNOW", "CRWD", "PLTR", "MRNA", "BILL", "DDOG", "NET", "ZS"],
        "description": "high-risk/high-reward innovative companies",
        "max_volatility": 1.2,
        "allocation": "10-20% of portfolio"
    }
}

STOCK_TIPS = [
    "📈 Invest in diversified stock portfolios to reduce risk",
    "⏳ Consider dollar-cost averaging to reduce market timing risk",
    "🌍 Diversify internationally to hedge against country-specific risks",
    "📅 Rebalance your stock portfolio at least once per year",
    "🔍 Research P/E ratios and growth rates before investing",
    "💡 Focus on companies with sustainable competitive advantages",
    "📉 Maintain a long-term perspective during market volatility",
    "💰 Reinvest dividends to benefit from compound growth"
]

STOCK_FAQS = {
    "how to analyze stocks": "🔍 Look at P/E ratio, growth rates, competitive advantage, and management quality",
    "best long term stocks": "🌱 Consider companies with strong moats, consistent growth, and innovation",
    "how much to invest in stocks": "📈 Allocate (100 - your age)% in stocks, e.g., 70% stocks if you're 30",
    "when to sell stocks": "💸 Consider selling when fundamentals deteriorate or you reach price targets",
    "how to pick stocks": "✅ Focus on companies with growing earnings, strong balance sheets, and good management",
    "best stock sectors": "🚀 Technology and healthcare often show strong growth potential",
    "stock market basics": "📊 Stocks represent ownership in companies. Prices fluctuate based on supply/demand",
    "what affects stock prices": "📉 Company performance, economic conditions, interest rates, and market sentiment"
}

# === API Configuration ===
PREFERRED_API_ORDER = [
    "twelve_data",
    "marketstack",
    "alpha_vantage"
]

APIS = {
    "alpha_vantage": os.getenv("ALPHA_VANTAGE_API_KEY"),
    "marketstack": os.getenv("MARKETSTACK_API_KEY"),
    "twelve_data": os.getenv("TWELVE_DATA_API_KEY"),
    "mediastack": os.getenv("MEDIASTACK_API_KEY"),
}

# === Path Constants ===
AI_INSIGHTS_PATH = "ai_insights.json"
LOCAL_STOCK_DATASET_PATH = "stocks_dataset"  # Updated path
MODEL_PATH = "rf_model.pkl"  # Updated path
SCALER_PATH = "rf_scaler.pkl"  # Updated path

# ========================
# === Helper Functions ===
# ========================
def build_prompt(user_input: dict, goal: str) -> str:
    """Build prompt for financial advice (stocks only)"""
    logger.info("🔧 Building financial advice prompt")
    income = user_input.get("income", "0")
    savings = user_input.get("savingAmount", "0")
    predictions = user_input.get("modelPredictions", {})
    volatility = user_input.get("marketVolatility", {})

    prompt = (
        "You are a professional financial advisor specialized in stocks and portfolio management.\n\n"
        "Your task is to analyze the user's financial profile and return advice **only** based on their investment goals.\n\n"
        "Use predicted investment returns and market volatility to guide stock selection.\n"
        "Responses must be realistic, actionable, and tailored to the user.\n\n"
        "Respond in valid **JSON format only**.\n\n"
        "User Profile:\n"
        f"- Monthly Income: {income} EGP\n"
        f"- Monthly Savings: {savings} EGP\n"
        "\nPredicted Investment Returns:\n" +
        "".join([f"- {asset}: {value}\n" for asset, value in predictions.items()]) +
        "\nMarket Volatility:\n" +
        "".join([f"- {asset}: Variance = {value}\n" for asset, value in volatility.items()]) +
        f"\n\nUser Goal: {goal}\n\n"
        "### Respond only in this JSON format:\n"
        "{\n"
        "  \"investment_plan\": [\"...\"]\n"
        "}\n"
        "Focus exclusively on stock market investments."
    )
    return prompt

def load_ai_insights() -> Dict[str, Dict]:
    """Load predicted returns and volatility from file with caching"""
    logger.info("📂 Loading AI insights...")
    try:
        if os.path.exists(AI_INSIGHTS_PATH):
            logger.info(f"✅ Found AI insights file at {AI_INSIGHTS_PATH}")
            with open(AI_INSIGHTS_PATH, "r") as f:
                return json.load(f)
    except Exception as e:
        logger.warning(f"⚠️ Failed to load AI insights file: {str(e)}")
    
    logger.warning("⚠️ Using default AI insights")
    return {
        "predicted_returns": {"stocks": "8.9%"},
        "market_volatility": {"stocks": "0.06"}
    }

async def fetch_user_profile(token: str) -> dict:
    """Fetch user profile from localhost:4000"""
    logger.info("🔍 Fetching user profile from localhost:4000")
    headers = {"Authorization": f"Bearer {token}"}
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                "http://localhost:4000/api/profile/me",
                headers=headers,
                timeout=5
            )
            logger.debug(f"🔧 Profile API response: {response.status_code}")
            if response.status_code != 200:
                logger.error(f"❌ Profile API error: {response.text}")
                return {}
            return response.json()
    except Exception as e:
        logger.error(f"❌ Profile fetch error: {str(e)}")
        return {}

async def fetch_twelve_data(symbol: str, days: int) -> dict:
    """Fetch stock data from Twelve Data API"""
    logger.info(f"🌐 Fetching Twelve Data for {symbol} ({days} days)")
    api_key = APIS.get("twelve_data")
    if not api_key:
        logger.warning("❌ Twelve Data API key not configured")
        return None
        
    params = {
        "symbol": symbol,
        "interval": "1day",
        "outputsize": days,
        "apikey": api_key
    }
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                "https://api.twelvedata.com/time_series",
                params=params,
                timeout=10
            )
            logger.debug(f"🔧 Twelve Data response status: {response.status_code}")
            data = response.json()
            
        if "values" not in data:
            logger.warning(f"⚠️ Twelve Data response missing 'values' key: {data}")
            return None
            
        logger.info(f"✅ Successfully fetched Twelve Data for {symbol}")
        closes = [float(item["close"]) for item in data["values"]]
        volumes = [float(item["volume"]) for item in data["values"]]
        highs = [float(item["high"]) for item in data["values"]]
        lows = [float(item["low"]) for item in data["values"]]
        
        return {
            "Close": closes,
            "Volume": volumes,
            "High": highs,
            "Low": lows,
            "Current": closes[-1]
        }
    except Exception as e:
        logger.error(f"❌ Twelve Data error: {str(e)}")
        return None

async def fetch_marketstack_data(symbol: str, days: int) -> dict:
    """Fetch stock data from Marketstack API"""
    logger.info(f"🌐 Fetching Marketstack data for {symbol} ({days} days)")
    api_key = APIS.get("marketstack")
    if not api_key:
        logger.warning("❌ Marketstack API key not configured")
        return None
        
    params = {
        "access_key": api_key,
        "symbols": symbol,
        "limit": days
    }
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                "http://api.marketstack.com/v1/eod",
                params=params,
                timeout=10
            )
            logger.debug(f"🔧 Marketstack response status: {response.status_code}")
            data = response.json()
            
        if "data" not in data or not data["data"]:
            logger.warning(f"⚠️ Marketstack response missing 'data' key: {data}")
            return None
            
        logger.info(f"✅ Successfully fetched Marketstack data for {symbol}")
        closes = [float(item["close"]) for item in data["data"]]
        volumes = [float(item["volume"]) for item in data["data"]]
        highs = [float(item["high"]) for item in data["data"]]
        lows = [float(item["low"]) for item in data["data"]]
        
        return {
            "Close": closes,
            "Volume": volumes,
            "High": highs,
            "Low": lows,
            "Current": closes[-1]
        }
    except Exception as e:
        logger.error(f"❌ Marketstack error: {str(e)}")
        return None

async def fetch_alpha_vantage_data(symbol: str, days: int) -> dict:
    """Fetch stock data from Alpha Vantage API"""
    logger.info(f"🌐 Fetching Alpha Vantage data for {symbol} ({days} days)")
    api_key = APIS.get("alpha_vantage")
    if not api_key:
        logger.warning("❌ Alpha Vantage API key not configured")
        return None
        
    params = {
        "function": "TIME_SERIES_DAILY_ADJUSTED",
        "symbol": symbol,
        "apikey": api_key,
        "outputsize": "compact" if days <= 100 else "full"
    }
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                "https://www.alphavantage.co/query",
                params=params,
                timeout=10
            )
            logger.debug(f"🔧 Alpha Vantage response status: {response.status_code}")
            data = response.json()
            
        if "Time Series (Daily)" not in data:
            logger.warning(f"⚠️ Alpha Vantage response missing time series: {data}")
            return None
            
        logger.info(f"✅ Successfully fetched Alpha Vantage data for {symbol}")
        time_series = data["Time Series (Daily)"]
        sorted_dates = sorted(time_series.keys(), reverse=True)[:days]
        
        closes = [float(time_series[date]["4. close"]) for date in sorted_dates]
        volumes = [float(time_series[date]["6. volume"]) for date in sorted_dates]
        highs = [float(time_series[date]["2. high"]) for date in sorted_dates]
        lows = [float(time_series[date]["3. low"]) for date in sorted_dates]
        
        return {
            "Close": closes,
            "Volume": volumes,
            "High": highs,
            "Low": lows,
            "Current": closes[0]
        }
    except Exception as e:
        logger.error(f"❌ Alpha Vantage error: {str(e)}")
        return None

async def fetch_stock_data(symbol: str, days: int = 30) -> dict:
    """Fetch stock data with robust fallback mechanism"""
    logger.info(f"📊 Fetching stock data for {symbol} ({days} days)")
    cache_key = f"{symbol}_{days}d"
    if cache_key in STOCK_DATA_CACHE:
        logger.info(f"♻️ Using cached stock data for {symbol}")
        return STOCK_DATA_CACHE[cache_key]
    
    logger.info(f"🔍 No cache found for {symbol}, querying APIs...")
    for api_name in PREFERRED_API_ORDER:
        try:
            logger.info(f"🔁 Trying {api_name} for {symbol}")
            if api_name == "twelve_data":
                data = await fetch_twelve_data(symbol, days)
            elif api_name == "marketstack":
                data = await fetch_marketstack_data(symbol, days)
            elif api_name == "alpha_vantage":
                data = await fetch_alpha_vantage_data(symbol, days)
            
            if data:
                logger.info(f"✅ Successfully got data from {api_name} for {symbol}")
                STOCK_DATA_CACHE[cache_key] = data
                return data
        except Exception as e:
            logger.warning(f"⚠️ {api_name} failed for {symbol}: {str(e)}")
            continue
    
    logger.warning(f"⚠️ All APIs failed for {symbol}, trying Gemini fallback...")
    try:
        prompt = f"""
        You are a financial data expert. Provide stock data for {symbol} for the last {days} days in JSON format:
        {{
          "Close": [list of closing prices],
          "Volume": [list of trading volumes],
          "High": [list of daily highs],
          "Low": [list of daily lows],
          "Current": [most recent closing price]
        }}
        Make realistic estimates based on historical trends if needed.
        """
        response = gemini_model.generate_content(prompt)  # Fixed: use gemini_model
        data = json.loads(response.text)
        logger.info(f"✅ Generated stock data for {symbol} using Gemini")
        STOCK_DATA_CACHE[cache_key] = data
        return data
    except Exception as e:
        logger.error(f"❌ Gemini stock data generation failed: {str(e)}")
        return {}

async def fetch_multiple_stocks(symbols: List[str]) -> Dict[str, Any]:
    """Fetch multiple stocks in parallel"""
    logger.info(f"📡 Fetching multiple stocks: {', '.join(symbols)}")
    tasks = [fetch_stock_data(sym) for sym in symbols]
    results = await asyncio.gather(*tasks)
    result_dict = {sym: result for sym, result in zip(symbols, results) if result}
    logger.info(f"✅ Successfully fetched {len(result_dict)}/{len(symbols)} stocks")
    return result_dict

async def fetch_stock_price(symbol: str) -> str:
    """Fetch stock price using fallback logic"""
    logger.info(f"💵 Fetching stock price for {symbol}")
    for api_name in PREFERRED_API_ORDER:
        try:
            logger.info(f"🔁 Trying {api_name} for price of {symbol}")
            if api_name == "twelve_data":
                api_key = APIS.get("twelve_data")
                if not api_key: continue
                    
                async with httpx.AsyncClient() as client:
                    response = await client.get(
                        "https://api.twelvedata.com/time_series",
                        params={
                            "symbol": symbol,
                            "interval": "1day",
                            "outputsize": 1,
                            "apikey": api_key
                        },
                        timeout=5
                    )
                data = response.json()
                if "values" in data and data["values"]:
                    latest = data["values"][0]
                    logger.info(f"✅ Got price from Twelve Data for {symbol}")
                    return f"📈 {symbol} Price: ${latest['close']} (as of {latest['datetime']})"
            
            elif api_name == "marketstack":
                api_key = APIS.get("marketstack")
                if not api_key: continue
                    
                async with httpx.AsyncClient() as client:
                    response = await client.get(
                        "http://api.marketstack.com/v1/eod",
                        params={
                            "access_key": api_key,
                            "symbols": symbol,
                            "limit": 1
                        },
                        timeout=5
                    )
                data = response.json()
                if "data" in data and data["data"]:
                    stock_data = data["data"][0]
                    logger.info(f"✅ Got price from Marketstack for {symbol}")
                    return f"📈 {symbol} Price: ${stock_data['close']} (as of {stock_data['date']})"
            
            elif api_name == "alpha_vantage":
                api_key = APIS.get("alpha_vantage")
                if not api_key: continue
                    
                async with httpx.AsyncClient() as client:
                    response = await client.get(
                        "https://www.alphavantage.co/query",
                        params={
                            "function": "TIME_SERIES_DAILY_ADJUSTED",
                            "symbol": symbol,
                            "apikey": api_key
                        },
                        timeout=5
                    )
                data = response.json()
                if "Time Series (Daily)" in data:
                    latest_date = sorted(data["Time Series (Daily)"].keys())[-1]
                    close_price = data["Time Series (Daily)"][latest_date]["4. close"]
                    logger.info(f"✅ Got price from Alpha Vantage for {symbol}")
                    return f"📈 {symbol} Price: ${close_price} (as of {latest_date})"
                    
        except Exception as e:
            logger.warning(f"⚠️ {api_name} price fetch failed for {symbol}: {str(e)}")
            continue
    
    logger.warning(f"⚠️ All APIs failed for {symbol}, using Gemini")
    try:
        prompt = f"What is the current price of {symbol} stock? Respond ONLY with the price in USD."
        response = gemini_model.generate_content(prompt)  # Fixed: use gemini_model
        return f"📈 {symbol} Price: ${response.text.strip()}"
    except Exception as e:
        logger.error(f"❌ Gemini price fetch failed: {str(e)}")
        return f"⚠️ Couldn't fetch price for {symbol}"

async def fetch_finance_news() -> str:
    """Fetch financial news from Mediastack"""
    logger.info("📰 Fetching financial news")
    mediastack_key = APIS.get("mediastack")
    if not mediastack_key:
        logger.error("❌ Mediastack API key not configured")
        return "⚠️ Mediastack API key not configured"
    
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(
                "http://api.mediastack.com/v1/news",
                params={
                    "access_key": mediastack_key,
                    "categories": "business",
                    "languages": "en",
                    "limit": 3,
                    "keywords": "stocks,investing,market"
                },
                timeout=5
            )
            logger.debug(f"🔧 Mediastack response status: {response.status_code}")
            data = response.json()
            
        if "data" in data and data["data"]:
            logger.info("✅ Successfully fetched financial news")
            articles = data["data"]
            news_items = [
                f"📰 {art['title']}\n{art['description']}\n🔗 {art['url']}"
                for art in articles
            ]
            return "📢 Latest Stock News:\n\n" + "\n\n".join(news_items)
        else:
            logger.warning("⚠️ Mediastack returned no news data")
            return ""
    except Exception as e:
        logger.error(f"❌ News error: {str(e)}")
        return ""

def get_risk_category(risk_score: int) -> str:
    """Categorize risk tolerance"""
    if risk_score < 3: return "conservative"
    if risk_score < 6: return "balanced"
    if risk_score < 9: return "growth"
    return "aggressive"

def calculate_technical_metrics(prices: List[float]) -> dict:
    """Calculate technical indicators for stocks"""
    if len(prices) < 5:
        return {}
    
    series = pd.Series(prices)
    
    # Calculate moving averages
    sma_10 = series.rolling(10).mean().iloc[-1] if len(prices) >= 10 else None
    sma_20 = series.rolling(20).mean().iloc[-1] if len(prices) >= 20 else None
    
    # Calculate RSI (simplified)
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(14).mean().iloc[-1] if len(prices) >= 14 else None
    avg_loss = loss.rolling(14).mean().iloc[-1] if len(prices) >= 14 else None
    rsi = 100 - (100 / (1 + (avg_gain / avg_loss))) if avg_loss != 0 and avg_loss is not None else None
    
    # Calculate trend
    x = range(len(prices))
    slope, _, _, _, _ = linregress(x, prices)
    
    return {
        "sma_10": sma_10,
        "sma_20": sma_20,
        "rsi": rsi,
        "trend_slope": slope
    }

def load_prediction_model():
    """Load RF model and scaler for stock predictions"""
    logger.info("🤖 Loading prediction model")
    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        logger.info("✅ Prediction model loaded successfully")
        return model, scaler
    except Exception as e:
        logger.error(f"❌ Model loading failed: {str(e)}")
        return None, None

async def predict_stock(symbol: str, days: int = 30) -> dict:
    """Predict stock performance using ML model"""
    logger.info(f"🔮 Predicting stock for {symbol}")
    model, scaler = load_prediction_model()
    if not model or not scaler:
        return {}
    
    stock_data = await fetch_stock_data(symbol, days)
    if not stock_data or not stock_data.get("Close"):
        return {}
    
    try:
        # Prepare features
        features = np.array([
            stock_data["Close"][-1],  # latest close
            np.mean(stock_data["Close"]),  # average price
            np.std(stock_data["Close"]),  # volatility
            stock_data["Volume"][-1],  # latest volume
            np.mean(stock_data["Volume"])  # average volume
        ]).reshape(1, -1)
        
        # Scale features and predict
        scaled_features = scaler.transform(features)
        prediction = model.predict(scaled_features)[0]
        
        return {
            "symbol": symbol,
            "predicted_return": f"{prediction:.2%}",
            "confidence": "High" if abs(prediction) > 0.05 else "Medium"
        }
    except Exception as e:
        logger.error(f"❌ Prediction failed for {symbol}: {str(e)}")
        return {}

# ===================================
# === Enhanced Recommendation Engine ===
# ===================================
async def generate_stock_recommendation(user_profile: dict) -> str:
    """Generate personalized stock recommendation with improved analysis"""
    logger.info("💡 Generating stock recommendation...")
    risk = user_profile.get("riskTolerance", 5)
    investment_horizon = user_profile.get("investmentHorizon", 5)
    logger.debug(f"👤 User profile - risk: {risk}, horizon: {investment_horizon}")
    
    ai_insights = load_ai_insights()
    predicted_return = ai_insights["predicted_returns"].get("stocks", "8.9%")
    volatility = ai_insights["market_volatility"].get("stocks", "0.06")
    
    risk_category = get_risk_category(risk)
    category_data = RISK_CATEGORIES[risk_category]
    logger.info(f"📊 Risk category: {risk_category} - {category_data['description']}")
    
    candidate_symbols = category_data["stocks"]
    stocks_data = await fetch_multiple_stocks(candidate_symbols)
    
    # Add predictions
    predictions = {}
    for symbol in candidate_symbols:
        prediction = await predict_stock(symbol)
        if prediction:
            predictions[symbol] = prediction
    
    candidates = []
    logger.info(f"🔍 Processing {len(stocks_data)} candidate stocks")
    for symbol in candidate_symbols:
        data = stocks_data.get(symbol)
        if not data or not data.get('Close'):
            logger.warning(f"⚠️ Skipping {symbol} - no data")
            continue
            
        prices = data['Close']
        current_price = data['Current']
        trend = ((prices[-1] - prices[0]) / prices[0]) * 100
        volatility_val = pd.Series(prices).pct_change().std() * 100
        technicals = calculate_technical_metrics(prices)
        
        if volatility_val > category_data["max_volatility"]:
            logger.info(f"⚠️ Skipping {symbol} - too volatile ({volatility_val:.2f} > {category_data['max_volatility']})")
            continue
            
        score = 0
        if technicals.get('trend_slope') and technicals['trend_slope'] > 0:
            score += 3
        if technicals.get('rsi') and 30 < technicals['rsi'] < 70:
            score += 2
        if current_price > technicals.get('sma_20', 0):
            score += 2
        if current_price > technicals.get('sma_10', 0):
            score += 1
            
        if symbol in predictions:
            pred = predictions[symbol]
            if pred['confidence'] == "High":
                score += 3
            else:
                score += 1
            
        logger.debug(f"📊 {symbol} score: {score} (price: ${current_price:.2f}, trend: {trend:.1f}%, volatility: {volatility_val:.1f}%)")
        candidates.append({
            "symbol": symbol,
            "price": current_price,
            "trend": trend,
            "volatility": volatility_val,
            "technicals": technicals,
            "score": score
        })
    
    if not candidates:
        logger.warning("⚠️ No suitable investments found")
        return "No suitable investments found. Try again later."
    
    best_stock = sorted(candidates, key=lambda x: x['score'], reverse=True)[0]
    logger.info(f"🏆 Selected best stock: {best_stock['symbol']} with score {best_stock['score']}")
    
    # Add prediction to report if available
    prediction_info = ""
    if best_stock['symbol'] in predictions:
        pred = predictions[best_stock['symbol']]
        prediction_info = f"\n📊 ML Prediction: {pred['predicted_return']} return ({pred['confidence']} confidence)"
    
    report = (
        f"📈 Recommendation for {risk_category.title()} Investor\n\n"
        f"Stock: {best_stock['symbol']}\n"
        f"Price: ${best_stock['price']:.2f}\n"
        f"30-Day Trend: {best_stock['trend']:.1f}%\n"
        f"Volatility: {best_stock['volatility']:.1f}%"
        f"{prediction_info}\n\n"
        f"Technical Indicators:\n"
        f"- 10-Day SMA: ${best_stock['technicals'].get('sma_10', 'N/A'):.2f}\n"
        f"- 20-Day SMA: ${best_stock['technicals'].get('sma_20', 'N/A'):.2f}\n"
        f"- RSI: {best_stock['technicals'].get('rsi', 'N/A'):.1f}\n"
        f"- Trend Strength: {'Strong' if best_stock['technicals'].get('trend_slope', 0) > 0.5 else 'Moderate'}\n\n"
        f"Portfolio Advice:\n"
        f"- Allocate {category_data['allocation']}\n"
        f"- Ideal for {'long-term' if investment_horizon > 3 else 'short-term'} investment\n"
        f"- Predicted Market Return: {predicted_return}\n"
        f"- Market Volatility: {volatility}"
    )
    
    logger.info("✅ Stock recommendation generated")
    return report

# =====================
# === Request Models ===
# =====================
class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = ""

class AdviceRequest(BaseModel):
    goal: str

# =========================
# === Financial Advice Routes ===
# =========================
@app.post("/generate/investment")
async def generate_investment_advice(request: Request):
    logger.info("🚀 /generate/investment route HIT")
    try:
        # Get authorization token
        auth_header = request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Missing or invalid authorization token")
        
        token = auth_header.split(" ")[1]
        
        # Fetch user profile
        profile = await fetch_user_profile(token)
        if not profile:
            raise HTTPException(status_code=404, detail="User profile not found")
        
        logger.info(f"👤 User profile: {profile}")
        
        # Build prompt with stock-only focus
        prompt = build_prompt(profile, "investment")
        response = gemini_model.generate_content(prompt)  # Fixed: use gemini_model
        result_text = response.text
        
        try:
            cleaned = result_text.strip().removeprefix("```json").removesuffix("```").strip()
            result_json = json.loads(cleaned)
            logger.info("✅ /generate/investment SUCCESS")
            return JSONResponse(content=result_json, status_code=200)
        except json.JSONDecodeError:
            logger.info("⚠️ /generate/investment returned plain text")
            return JSONResponse(content={"output": result_text}, status_code=200)
            
    except Exception as e:
        logger.error(f"❌ /generate/investment ERROR: {str(e)}")
        return JSONResponse(
            content={"error": str(e)}, 
            status_code=500
        )

# =====================
# === Chatbot Routes ===
# =====================
@app.post("/chatbot/chat")
async def chat_with_bot(request: ChatRequest, authorization: str = Depends(lambda x: x.headers.get("Authorization"))):
    logger.info("🚀 /chatbot/chat route HIT")
    try:
        user_message = request.message.strip()
        session_id = request.session_id or ""
        logger.debug(f"📩 Message: {user_message}, Session: {session_id}")
        
        if not user_message:
            logger.warning("⚠️ Empty message received")
            return JSONResponse(
                content={"error": "Empty message"}, 
                status_code=400
            )
        
        # Fetch user profile using authorization token
        if not authorization or not authorization.startswith("Bearer "):
            profile = {}
        else:
            token = authorization.split(" ")[1]
            profile = await fetch_user_profile(token) or {}
        
        logger.debug(f"👤 User profile: {profile}")
        
        cache_key = hashlib.sha256(
            f"{user_message}-{json.dumps(profile, sort_keys=True)}".encode()
        ).hexdigest()
        
        if cache_key in RESPONSE_CACHE:
            logger.info("♻️ Serving response from cache")
            logger.info("✅ /chatbot/chat SUCCESS (cached)")
            return JSONResponse(content={
                "output": RESPONSE_CACHE[cache_key],
                "session_id": session_id
            }, status_code=200)
        
        async with SESSION_LOCK:
            if session_id and session_id in SESSION_HISTORY:
                conversation_history = SESSION_HISTORY[session_id]
                if len(conversation_history) > 6:
                    conversation_history = conversation_history[-6:]
            else:
                session_id = hashlib.sha256(f"{time.time()}{user_message}".encode()).hexdigest()[:16]
                conversation_history = []
            
            conversation_history.append({"role": "user", "content": user_message})
            SESSION_HISTORY[session_id] = conversation_history
        
        lower_msg = user_message.lower()
        
        # FAQ handling
        if lower_msg in STOCK_FAQS:
            response_text = STOCK_FAQS[lower_msg]
            RESPONSE_CACHE[cache_key] = response_text
            logger.info("✅ /chatbot/chat SUCCESS (FAQ)")
            return JSONResponse(content={
                "output": response_text,
                "session_id": session_id
            }, status_code=200)
        
        # Stock price requests
        stock_symbols = {
            "apple": "AAPL", "microsoft": "MSFT", "google": "GOOGL",
            "amazon": "AMZN", "tesla": "TSLA", "nvidia": "NVDA",
            "meta": "META", "netflix": "NFLX", "amd": "AMD",
            "intel": "INTC", "coca": "KO", "pepsi": "PEP", "walmart": "WMT"
        }
        
        for keyword, symbol in stock_symbols.items():
            if keyword in lower_msg:
                response_text = await fetch_stock_price(symbol)
                RESPONSE_CACHE[cache_key] = response_text
                logger.info(f"✅ /chatbot/chat SUCCESS (Stock price: {symbol})")
                return JSONResponse(content={
                    "output": response_text,
                    "session_id": session_id
                }, status_code=200)
        
        # Investment advice
        investment_triggers = [
            "invest", "stock", "portfolio", "recommend", 
            "which stock", "buy", "should i invest", "opportunity"
        ]
        
        if any(trigger in lower_msg for trigger in investment_triggers):
            response_text = await generate_stock_recommendation(profile)
            RESPONSE_CACHE[cache_key] = response_text
            logger.info("✅ /chatbot/chat SUCCESS (Investment advice)")
            return JSONResponse(content={
                "output": response_text,
                "session_id": session_id
            }, status_code=200)
        
        # News requests
        if "news" in lower_msg or "update" in lower_msg:
            response_text = await fetch_finance_news()
            RESPONSE_CACHE[cache_key] = response_text
            logger.info("✅ /chatbot/chat SUCCESS (News)")
            return JSONResponse(content={
                "output": response_text or "No stock news available",
                "session_id": session_id
            }, status_code=200)
        
        # Default Gemini response
        context_prompt = """
You are an expert financial advisor specialized in stocks and investments. Follow these rules:

1. Scope:
   - Only respond to stock market and investment questions
   - For non-finance topics: "I specialize in stock market advice"

2. Personalization:
   - Consider user profile: {profile}
   - Maintain conversation context

3. Response Style:
   - Be concise (2-3 sentences)
   - Use natural language with occasional emojis
   - Include one relevant stock tip if appropriate

CONVERSATION HISTORY:
{history}

USER QUESTION:
"{current_message}"
"""
        history_text = "\n".join(
            [f"{msg['role'].capitalize()}: {msg['content']}" 
            for msg in conversation_history[:-1]]
        )
        
        formatted_prompt = context_prompt.format(
            profile=json.dumps(profile, indent=2),
            history=history_text,
            current_message=user_message
        )
        
        response = gemini_model.generate_content(formatted_prompt)  # Fixed: use gemini_model
        response_text = response.text.strip()
        
        # Add stock tip if response is short
        if len(response_text.split()) < 30:
            tip = STOCK_TIPS[int(time.time()) % len(STOCK_TIPS)]
            response_text += f"\n\n💡 Stock Tip: {tip}"
        
        async with SESSION_LOCK:
            conversation_history.append({"role": "assistant", "content": response_text})
            SESSION_HISTORY[session_id] = conversation_history
        
        RESPONSE_CACHE[cache_key] = response_text
        logger.info("✅ /chatbot/chat SUCCESS (Gemini response)")
        return JSONResponse(content={
            "output": response_text,
            "session_id": session_id
        }, status_code=200)
        
    except Exception as e:
        logger.exception(f"❌ /chatbot/chat ERROR: {str(e)}")
        return JSONResponse(
            content={"error": "Service unavailable. Please try again later."},
            status_code=500
        )

@app.post("/chatbot/generate/{goal}")
async def generate_goal_advice(goal: str, authorization: str = Depends(lambda x: x.headers.get("Authorization"))):
    logger.info(f"🚀 /chatbot/generate/{goal} route HIT")
    try:
        # Fetch user profile using authorization token
        if not authorization or not authorization.startswith("Bearer "):
            profile = {}
        else:
            token = authorization.split(" ")[1]
            profile = await fetch_user_profile(token) or {}
        
        logger.debug(f"👤 Profile: {profile}")
        
        cache_key = hashlib.sha256(
            f"{goal}-{json.dumps(profile, sort_keys=True)}".encode()
        ).hexdigest()
        
        if cache_key in RESPONSE_CACHE:
            logger.info("♻️ Serving response from cache")
            logger.info(f"✅ /chatbot/generate/{goal} SUCCESS (cached)")
            return JSONResponse(content={"advice": RESPONSE_CACHE[cache_key]}, status_code=200)
        
        ai_insights = load_ai_insights()
        predicted_returns = ai_insights["predicted_returns"]
        market_volatility = ai_insights["market_volatility"]
        
        prompt = f"""
You are a professional financial advisor specialized in stocks. Your task:

1. Analyze user's financial health:
   - Risk tolerance: {profile.get('riskTolerance', 5)}/10
   - Investment horizon: {profile.get('investmentHorizon', 5)} years
   - Financial goals: {profile.get('financialGoals', 'Not specified')}

2. Provide personalized stock advice for goal: {goal}

3. Incorporate market insights:
   - Predicted returns: {predicted_returns}
   - Market volatility: {market_volatility}

4. Response guidelines:
   - Use natural language without markdown
   - Be specific and actionable
   - Limit to 3-5 key recommendations
   - Focus exclusively on stock market investments

USER PROFILE SUMMARY:
- Risk category: {get_risk_category(profile.get('riskTolerance', 5))}
- Investment experience: {profile.get('experience', 'Not specified')}
- Current portfolio size: {profile.get('portfolioSize', 'Not specified')}
- Primary investment goal: {goal}

ADVICE:
"""
        response = gemini_model.generate_content(prompt)  # Fixed: use gemini_model
        response_text = response.text.strip()
        RESPONSE_CACHE[cache_key] = response_text
        
        logger.info(f"✅ /chatbot/generate/{goal} SUCCESS")
        return JSONResponse(content={"advice": response_text}, status_code=200)
        
    except Exception as e:
        logger.exception(f"❌ /chatbot/generate/{goal} ERROR: {str(e)}")
        return JSONResponse(
            content={"error": "Service unavailable. Please try again later."},
            status_code=500
        )

# ===================
# === Health Check ===
# ===================
@app.get("/health")
async def health_check():
    logger.info("🩺 /health route HIT")
    try:
        # Test API connectivity
        api_status = {}
        for api_name in PREFERRED_API_ORDER:
            api_key = APIS.get(api_name)
            api_status[api_name] = bool(api_key)
        
        # Test stock data loading
        test_stock = await fetch_stock_data('AAPL', 1)
        test_news = await fetch_finance_news()
        
        # Test model loading
        test_model, test_scaler = load_prediction_model()
        
        status = {
            "status": "operational",
            "services": {
                "stock_data": bool(test_stock),
                "news_service": bool(test_news),
                "prediction_model": bool(test_model and test_scaler),
                "api_connectivity": api_status
            },
            "cache_stats": {
                "response_cache": len(RESPONSE_CACHE),
                "stock_data_cache": len(STOCK_DATA_CACHE),
                "active_sessions": len(SESSION_HISTORY)
            }
        }
        logger.info("✅ /health SUCCESS")
        return JSONResponse(content=status, status_code=200)
    except Exception as e:
        logger.error(f"❌ /health ERROR: {str(e)}")
        return JSONResponse(
            content={"status": "degraded", "error": str(e)},
            status_code=500
        )

# ===============
# === Startup ===
# ===============
@app.on_event("startup")
async def startup_event():
    logger.info("🚀 Service starting up...")
    try:
        logger.info("🔍 Preloading essential data...")
        await fetch_stock_data('AAPL')
        await fetch_finance_news()
        load_prediction_model()
        logger.info("✅ Service initialized successfully")
    except Exception as e:
        logger.error(f"❌ Startup failed: {str(e)}")