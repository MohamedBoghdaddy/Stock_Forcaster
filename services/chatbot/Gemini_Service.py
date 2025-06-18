from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime, timedelta
from typing import Optional, Dict, Any, List
import google.generativeai as genai
import os
import json
import logging
import hashlib
import asyncio
import pandas as pd
import httpx
import glob
import time
from cachetools import TTLCache
from dotenv import load_dotenv
from textblob import TextBlob
from scipy.stats import linregress

# Load environment variables
load_dotenv()

# === Setup ===
app = FastAPI(title="Stock Forecaster API", version="1.1.0")
print("✅ FastAPI application initialized")

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
print("🔒 CORS middleware configured")

# API Configuration
PREFERRED_API_ORDER = [
    "twelve_data",
    "marketstack",
    "alpha_vantage"
]

APIS = {
    "alpha_vantage": os.getenv("ALPHA_VANTAGE_API_KEY"),
    "finnhub": os.getenv("FINNHUB_API_KEY"),
    "marketstack": os.getenv("MARKETSTACK_API_KEY"),
    "twelve_data": os.getenv("TWELVE_DATA_API_KEY"),
    "binance": {
        "key": os.getenv("BINANCE_API_KEY"),
        "secret": os.getenv("BINANCE_SECRET_KEY")
    },
    "mediastack": os.getenv("MEDIASTACK_API_KEY"),
}

# Initialize Gemini
print("🧠 Configuring Gemini model...")
try:
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    model = genai.GenerativeModel("gemini-1.5-flash")
    print("✅ Gemini configured successfully")
except Exception as e:
    print(f"❌ Gemini configuration failed: {str(e)}")
    raise RuntimeError("Gemini initialization failed")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
print("📝 Logging configured")

# Create caches
RESPONSE_CACHE = TTLCache(maxsize=1000, ttl=300)  # 5-minute cache
STOCK_DATA_CACHE = TTLCache(maxsize=200, ttl=3600)  # 1-hour cache
FUNDAMENTALS_CACHE = TTLCache(maxsize=200, ttl=86400)  # 24-hour cache
print("💾 Caches initialized")

# Session management
SESSION_HISTORY = TTLCache(maxsize=1000, ttl=3600)  # 1-hour session cache
SESSION_LOCK = asyncio.Lock()
print("🔐 Session management initialized")

# Financial knowledge base
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

FINANCIAL_TIPS = [
    "💰 Save at least 20% of your income each month.",
    "📉 Avoid impulse buying by waiting 24 hours before making a purchase.",
    "📊 Invest in diversified assets to reduce risk.",
    "🏦 Use high-yield savings accounts for emergency funds.",
    "💳 Pay off high-interest debt as soon as possible to avoid extra fees.",
    "📈 Consider dollar-cost averaging to reduce market timing risk.",
    "🌍 Diversify internationally to hedge against country-specific risks.",
    "📅 Rebalance your portfolio at least once per year.",
    "🧾 Keep investment expenses below 0.5% of assets annually.",
    "🛡️ Maintain 3-6 months of living expenses in cash equivalents."
]

FAQS = {
    "how to save money": "💰 Save at least 20% of your income each month and avoid impulse purchases.",
    "best way to invest": "📊 Diversify your investments and consider low-cost index funds.",
    "how to improve credit score": "✅ Pay bills on time and keep credit utilization below 30%.",
    "how to start budgeting": "📋 Track your expenses and allocate your income into savings, needs, and wants.",
    "what is dollar cost averaging": "⏳ Invest fixed amounts regularly to reduce market timing risk.",
    "how much to invest in stocks": "📈 Allocate (100 - your age)% in stocks, e.g., 70% stocks if you're 30.",
    "best long term investments": "🌱 Consider index funds, blue-chip stocks, and real estate for long-term growth.",
    "how to analyze stocks": "🔍 Look at P/E ratio, growth rates, competitive advantage, and management quality."
}

print("💡 Financial knowledge base loaded")

# === Helper Functions ===
async def fetch_twelve_data(symbol: str, days: int) -> dict:
    """Fetch data from Twelve Data API"""
    api_key = APIS.get("twelve_data")
    if not api_key:
        return None
        
    print(f"📈 Using Twelve Data for {symbol}")
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
            data = response.json()
            
        if "values" not in data:
            return None
            
        # Process data into consistent format
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
        print(f"❌ Twelve Data error: {str(e)}")
        return None

async def fetch_marketstack_data(symbol: str, days: int) -> dict:
    """Fetch data from Marketstack API"""
    api_key = APIS.get("marketstack")
    if not api_key:
        return None
        
    print(f"📊 Using Marketstack for {symbol}")
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
            data = response.json()
            
        if "data" not in data or not data["data"]:
            return None
            
        # Process data into consistent format
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
        print(f"❌ Marketstack error: {str(e)}")
        return None

async def fetch_alpha_vantage_data(symbol: str, days: int) -> dict:
    """Fetch data from Alpha Vantage API"""
    api_key = APIS.get("alpha_vantage")
    if not api_key:
        return None
        
    print(f"📉 Using Alpha Vantage for {symbol}")
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
            data = response.json()
            
        if "Time Series (Daily)" not in data:
            return None
            
        # Process data into consistent format
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
        print(f"❌ Alpha Vantage error: {str(e)}")
        return None

async def fetch_stock_data(symbol: str, days: int = 30) -> dict:
    """Fetch stock data with robust fallback mechanism"""
    # Check cache first
    cache_key = f"{symbol}_{days}d"
    if cache_key in STOCK_DATA_CACHE:
        return STOCK_DATA_CACHE[cache_key]
    
    # Try preferred APIs in order
    for api_name in PREFERRED_API_ORDER:
        try:
            if api_name == "twelve_data":
                data = await fetch_twelve_data(symbol, days)
            elif api_name == "marketstack":
                data = await fetch_marketstack_data(symbol, days)
            elif api_name == "alpha_vantage":
                data = await fetch_alpha_vantage_data(symbol, days)
            
            if data:
                STOCK_DATA_CACHE[cache_key] = data
                return data
        except Exception as e:
            print(f"⚠️ {api_name} failed: {str(e)}")
            continue
    
    return {}

async def fetch_multiple_stocks(symbols: List[str]) -> Dict[str, Any]:
    """Fetch multiple stocks in parallel"""
    tasks = [fetch_stock_data(sym) for sym in symbols]
    results = await asyncio.gather(*tasks)
    return {sym: result for sym, result in zip(symbols, results) if result}

async def analyze_stock_fundamentals_safe(symbol: str) -> dict:
    """Fallback-safe fundamentals using available APIs"""
    try:
        # Try Twelve Data first
        twelve_data = await fetch_twelve_data(symbol, 1)
        if twelve_data:
            return {
                "pe_ratio": None,  # Twelve Data doesn't provide this
                "peg_ratio": None,
                "debt_equity": None,
                "roe": None,
                "dividend": None,
                "beta": None,
                "profit_margins": None,
                "recommendation": None,
                "target_price": None,
                "current_price": twelve_data["Current"]
            }
            
        # Fallback to Alpha Vantage
        alpha_data = await fetch_alpha_vantage_data(symbol, 1)
        if alpha_data:
            return {
                "pe_ratio": None,  # Alpha Vantage requires separate endpoint
                "peg_ratio": None,
                "debt_equity": None,
                "roe": None,
                "dividend": None,
                "beta": None,
                "profit_margins": None,
                "recommendation": None,
                "target_price": None,
                "current_price": alpha_data["Current"]
            }
            
        return {}
    except Exception as e:
        print(f"❌ Fundamental fetch failed for {symbol}: {str(e)}")
        return {}

def get_risk_category(risk_score: int) -> str:
    """Categorize risk tolerance"""
    if risk_score < 3: return "conservative"
    if risk_score < 6: return "balanced"
    if risk_score < 9: return "growth"
    return "aggressive"

def calculate_technical_metrics(prices: List[float]) -> dict:
    """Calculate technical indicators"""
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
    rsi = 100 - (100 / (1 + (avg_gain / avg_loss))) if avg_loss != 0 else None
    
    # Calculate trend
    x = range(len(prices))
    slope, _, _, _, _ = linregress(x, prices)
    
    return {
        "sma_10": sma_10,
        "sma_20": sma_20,
        "rsi": rsi,
        "trend_slope": slope
    }

async def fetch_crypto_price(symbol: str) -> str:
    """Fetch cryptocurrency price from Binance"""
    clean_symbol = symbol.replace(r"\W+", "", symbol).upper() + "USDT"
    try:
        print(f"💰 Fetching crypto price for {symbol}")
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"https://api.binance.com/api/v3/ticker/price?symbol={clean_symbol}",
                timeout=5
            )
        data = response.json()
        if 'price' in data:
            print(f"✅ Crypto price fetched: {data['price']}")
            return f"🚀 **{symbol.upper()} Price**: **${data['price']}**"
        return ""
    except Exception as e:
        print(f"❌ Crypto price error: {str(e)}")
        return ""

async def fetch_stock_price(symbol: str) -> str:
    """Fetch stock price using fallback logic"""
    # Try preferred APIs in order
    for api_name in PREFERRED_API_ORDER:
        try:
            if api_name == "twelve_data":
                api_key = APIS.get("twelve_data")
                if not api_key:
                    continue
                    
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
                    return f"📈 **{symbol} Price**: **${latest['close']}** (as of {latest['datetime']})"
            
            elif api_name == "marketstack":
                api_key = APIS.get("marketstack")
                if not api_key:
                    continue
                    
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
                    return f"📈 **{symbol} Price**: **${stock_data['close']}** (as of {stock_data['date']})"
            
            elif api_name == "alpha_vantage":
                api_key = APIS.get("alpha_vantage")
                if not api_key:
                    continue
                    
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
                    return f"📈 **{symbol} Price**: **${close_price}** (as of {latest_date})"
                    
        except Exception as e:
            print(f"⚠️ {api_name} price fetch failed: {str(e)}")
            continue
    
    return f"⚠️ Sorry, couldn't fetch price for **{symbol}** right now."

async def fetch_currency_rates(base: str = "USD", target: str = "EUR") -> str:
    """Fetch currency exchange rates from Alpha Vantage"""
    alpha_vantage_key = APIS.get("alpha_vantage")
    if not alpha_vantage_key:
        return "⚠️ Alpha Vantage API key not configured"
    
    try:
        print(f"💱 Fetching currency rates: {base} to {target}")
        async with httpx.AsyncClient() as client:
            response = await client.get(
                "https://www.alphavantage.co/query",
                params={
                    "function": "CURRENCY_EXCHANGE_RATE",
                    "from_currency": base,
                    "to_currency": target,
                    "apikey": alpha_vantage_key
                },
                timeout=5
            )
        data = response.json()
        if "Realtime Currency Exchange Rate" in data:
            rate = data["Realtime Currency Exchange Rate"]["5. Exchange Rate"]
            print(f"✅ Currency rate fetched: 1 {base} = {rate} {target}")
            return f"💱 **Exchange Rate**: **1 {base} = {rate} {target}**"
        else:
            error_msg = data.get("Error Message", "No exchange rate data")
            print(f"❌ Currency error: {error_msg}")
            return ""
    except Exception as e:
        print(f"❌ Currency error: {str(e)}")
        return ""

async def fetch_metal_prices(metal: str) -> str:
    """Fetch metal prices from Finnhub"""
    finnhub_key = APIS.get("finnhub")
    if not finnhub_key:
        return "⚠️ Finnhub API key not configured"
    
    metal_symbols = {"GOLD": "GC1!", "SILVER": "SI1!"}
    symbol = metal_symbols.get(metal.upper())
    if not symbol:
        return f"⚠️ Unknown metal: {metal}"
    
    try:
        print(f"🥇 Fetching metal price for {metal}")
        async with httpx.AsyncClient() as client:
            response = await client.get(
                "https://finnhub.io/api/v1/quote",
                params={
                    "symbol": symbol,
                    "token": finnhub_key
                },
                timeout=5
            )
        data = response.json()
        if "c" in data:
            print(f"✅ Metal price fetched: ${data['c']} per ounce")
            return f"🥇 **{metal.capitalize()} Price**: **${data['c']} per ounce**"
        else:
            print(f"❌ Metal price error: {data.get('error', 'No price data')}")
            return ""
    except Exception as e:
        print(f"❌ Metal price error: {str(e)}")
        return ""

async def fetch_finance_news() -> str:
    """Fetch financial news from Mediastack"""
    mediastack_key = APIS.get("mediastack")
    if not mediastack_key:
        return "⚠️ Mediastack API key not configured"
    
    try:
        print("📰 Fetching financial news")
        async with httpx.AsyncClient() as client:
            response = await client.get(
                "http://api.mediastack.com/v1/news",
                params={
                    "access_key": mediastack_key,
                    "categories": "business",
                    "languages": "en",
                    "limit": 5
                },
                timeout=5
            )
        data = response.json()
        if "data" in data and data["data"]:
            articles = data["data"]
            print(f"✅ Retrieved {len(articles)} news articles")
            news_items = [
                f"📰 **{art['title']}**\n{art['description']}\n🔗 [Read more]({art['url']})"
                for art in articles
            ]
            return "📢 **Latest Financial News**:\n\n" + "\n\n".join(news_items)
        else:
            error_msg = data.get("error", "No news data")
            print(f"❌ News error: {error_msg}")
            return ""
    except Exception as e:
        print(f"❌ News error: {str(e)}")
        return ""

def analyze_sentiment(text: str) -> str:
    """Analyze text sentiment using TextBlob"""
    print(f"😊 Analyzing sentiment for: {text[:50]}...")
    analysis = TextBlob(text)
    if analysis.sentiment.polarity > 0.1:
        sentiment = "😊 Positive"
    elif analysis.sentiment.polarity < -0.1:
        sentiment = "😞 Negative"
    else:
        sentiment = "😐 Neutral"
    print(f"✅ Sentiment: {sentiment}")
    return sentiment

# === Enhanced Recommendation Engine ===
async def generate_stock_recommendation(user_profile: dict) -> str:
    """Generate personalized stock recommendation with parallel data fetching"""
    risk = user_profile.get("riskTolerance", 5)
    investment_horizon = user_profile.get("investmentHorizon", 5)
    print(f"📈 Generating stock recommendation for risk level: {risk}/10")
    
    # Determine risk category
    risk_category = get_risk_category(risk)
    category_data = RISK_CATEGORIES[risk_category]
    
    # Fetch all candidate stocks in parallel
    candidate_symbols = category_data["stocks"]
    stocks_data = await fetch_multiple_stocks(candidate_symbols)
    
    # Process candidates
    candidates = []
    for symbol in candidate_symbols:
        data = stocks_data.get(symbol)
        if not data or not data.get('Close'):
            continue
            
        # Calculate metrics
        prices = data['Close']
        current_price = data['Current']
        trend = ((prices[-1] - prices[0]) / prices[0]) * 100
        volatility = pd.Series(prices).pct_change().std() * 100
        fundamentals = await analyze_stock_fundamentals_safe(symbol)
        technicals = calculate_technical_metrics(prices)
        
        # Skip if too volatile for category
        if volatility > category_data["max_volatility"]:
            continue
            
        # Calculate composite score
        score = 0
        if fundamentals.get('peg_ratio') and fundamentals['peg_ratio'] < 2:
            score += 3
        if fundamentals.get('pe_ratio') and fundamentals['pe_ratio'] < 25:
            score += 2
        if fundamentals.get('roe') and fundamentals['roe'] > 15:
            score += 2
        if fundamentals.get('recommendation') and fundamentals['recommendation'] == "Buy":
            score += 2
        if technicals.get('trend_slope') and technicals['trend_slope'] > 0:
            score += 1
            
        candidates.append({
            "symbol": symbol,
            "name": data.get('Name', symbol),
            "price": current_price,
            "trend": trend,
            "volatility": volatility,
            "fundamentals": fundamentals,
            "technicals": technicals,
            "score": score
        })
    
    if not candidates:
        return "I couldn't find suitable investments matching your profile right now. Please try again later."
    
    # Select best candidate
    best_stock = sorted(candidates, key=lambda x: x['score'], reverse=True)[0]
    fundamentals = best_stock['fundamentals']
    technicals = best_stock['technicals']
    
    # Generate recommendation report
    report = (
        f"📈 **Recommendation for {risk_category.title()} Investor**\n\n"
        f"**{best_stock['name']} ({best_stock['symbol']})**\n"
        f"Current Price: ${best_stock['price']:.2f}\n"
        f"30-Day Trend: {best_stock['trend']:.1f}%\n"
        f"Volatility: {best_stock['volatility']:.1f}%\n\n"
        f"📊 **Fundamentals**\n"
        f"- P/E Ratio: {fundamentals.get('pe_ratio', 'N/A'):.1f}\n"
        f"- PEG Ratio: {fundamentals.get('peg_ratio', 'N/A'):.1f}\n"
        f"- ROE: {fundamentals.get('roe', 'N/A'):.1f}%\n"
        f"- Dividend Yield: {fundamentals.get('dividend', 0):.1f}%\n"
        f"- Debt/Equity: {fundamentals.get('debt_equity', 'N/A'):.2f}\n"
        f"- Beta: {fundamentals.get('beta', 'N/A'):.2f}\n"
        f"- Profit Margins: {fundamentals.get('profit_margins', 'N/A'):.1f}%\n"
        f"- Target Price: ${fundamentals.get('target_price', 'N/A'):.2f}\n"
        f"- Analyst Consensus: {fundamentals.get('recommendation', 'N/A')}\n\n"
        f"📉 **Technical Indicators**\n"
        f"- 10-Day SMA: ${technicals.get('sma_10', 'N/A'):.2f}\n"
        f"- 20-Day SMA: ${technicals.get('sma_20', 'N/A'):.2f}\n"
        f"- RSI: {technicals.get('rsi', 'N/A'):.1f}\n"
        f"- Trend Slope: {technicals.get('trend_slope', 'N/A'):.4f}\n\n"
        f"💡 **Why This Stock?**\n"
        f"Matches your risk profile ({risk}/10) as a {category_data['description']}."
    )
    
    # Add portfolio advice
    horizon_advice = "short-term trades" if investment_horizon < 3 else "long-term holding"
    report += (
        f"\n\n🔮 **Portfolio Advice**\n"
        f"- Consider allocating {category_data['allocation']} to this position\n"
        f"- Ideal for {horizon_advice} ({investment_horizon} year horizon)\n"
        f"- Monitor quarterly earnings reports for performance updates\n"
        f"- Set stop-loss at {best_stock['price'] * 0.9:.2f} (10% below current)"
    )
    
    print(f"✅ Recommendation generated: {best_stock['symbol']}")
    return report

# === Models ===
class ChatRequest(BaseModel):
    message: str
    profile: Optional[Dict] = {}
    session_id: Optional[str] = ""

# === Routes ===
@app.post("/chat")
async def chat_with_bot(request: ChatRequest):
    print("\n" + "="*50)
    print(f"💬 New chat request: {request.message[:100]}...")
    print(f"🧾 Session ID: {request.session_id or 'None'}")
    print(f"👤 User profile: {json.dumps(request.profile, indent=2) if request.profile else 'None'}")
    
    try:
        user_message = request.message.strip()
        profile = request.profile or {}
        session_id = request.session_id or ""
        
        if not user_message:
            print("⚠️ Empty message received")
            raise HTTPException(status_code=400, detail="Empty message")
        
        lower_msg = user_message.lower()
        
        # Check FAQs
        if lower_msg in FAQS:
            print(f"ℹ️ FAQ matched: {lower_msg}")
            return JSONResponse(content={
                "output": FAQS[lower_msg],
                "session_id": session_id
            })
        
        # Analyze sentiment
        sentiment = analyze_sentiment(user_message)
        response_parts = [f"🔍 **Sentiment**: {sentiment}"]
        
        # Check for investment advice request
        investment_request = (
            "invest" in lower_msg or 
            "stock" in lower_msg or 
            "long term" in lower_msg or 
            "what should i buy" in lower_msg or
            "recommend" in lower_msg or
            "portfolio" in lower_msg or 
            "diversify" in lower_msg or 
            "buy stock" in lower_msg
        )
        
        if investment_request:
            print("🔍 Investment advice request detected")
            response_text = await generate_stock_recommendation(profile)
            print(f"📤 Sending investment recommendation: {response_text[:100]}...")
            return JSONResponse(content={
                "output": response_text,
                "session_id": session_id
            })
        
        # Check for specific data requests
        # Check for crypto
        if "crypto" in lower_msg or "bitcoin" in lower_msg or "btc" in lower_msg:
            crypto = "BTC" if "btc" in lower_msg else "ETH"
            print(f"🔍 Crypto request detected: {crypto}")
            response_parts.append(await fetch_crypto_price(crypto))
        
        # Check for stock
        elif "stock" in lower_msg:
            symbol = "AAPL"  # Default to Apple
            if "microsoft" in lower_msg or "msft" in lower_msg:
                symbol = "MSFT"
            elif "google" in lower_msg or "googl" in lower_msg:
                symbol = "GOOGL"
            elif "amazon" in lower_msg or "amzn" in lower_msg:
                symbol = "AMZN"
            print(f"🔍 Stock request detected: {symbol}")
            response_parts.append(await fetch_stock_price(symbol))
        
        # Check for currency
        elif "currency" in lower_msg or "exchange" in lower_msg:
            print("🔍 Currency request detected")
            response_parts.append(await fetch_currency_rates())
        
        # Check for metals
        elif "gold" in lower_msg or "silver" in lower_msg:
            metal = "gold" if "gold" in lower_msg else "silver"
            print(f"🔍 Metal request detected: {metal}")
            response_parts.append(await fetch_metal_prices(metal))
        
        # Check for news
        elif "news" in lower_msg:
            print("🔍 News request detected")
            response_parts.append(await fetch_finance_news())
        
        # If we have API responses, use them
        if len(response_parts) > 1:
            response_text = "\n\n".join(response_parts)
            print(f"📤 Sending API-based response: {response_text[:100]}...")
            return JSONResponse(content={
                "output": response_text,
                "session_id": session_id
            })
        
        # Default financial advice using Gemini
        print("🤖 Generating advice with Gemini")
        prompt = (
            "You are a certified financial advisor. "
            "Provide concise investment advice (1-2 paragraphs). "
            "Focus on stocks, ETFs, or long-term investments. "
            f"User profile: {json.dumps(profile)}\n\n"
            f"Question: {user_message}"
        )
        
        print(f"📝 Gemini prompt: {prompt[:150]}...")
        start_time = time.time()
        response = model.generate_content(prompt)
        gen_time = time.time() - start_time
        response_text = response.text.strip()
        print(f"✅ Gemini response generated in {gen_time:.2f}s: {response_text[:100]}...")
        
        # Add financial tip if response is short
        if len(response_text.split()) < 30:
            tip = FINANCIAL_TIPS[int(time.time()) % len(FINANCIAL_TIPS)]
            print(f"💡 Adding financial tip: {tip}")
            response_text += f"\n\n💡 **Financial Tip**: {tip}"
        
        print(f"📤 Sending Gemini response: {response_text[:100]}...")
        return JSONResponse(content={
            "output": response_text,
            "session_id": session_id
        })
        
    except Exception as e:
        logger.exception(f"Chat error: {str(e)}")
        print(f"❌ Critical error: {str(e)}")
        return JSONResponse(
            content={"error": "Financial advice service unavailable. Please try again later."},
            status_code=500
        )

# === Health Check ===
@app.get("/health")
async def health_check():
    print("🩺 Health check requested")
    return {
        "status": "operational",
        "version": "1.1.0",
        "model": "gemini-1.5-flash",
        "api_priority": PREFERRED_API_ORDER,
        "services": {
            "alpha_vantage": bool(APIS["alpha_vantage"]),
            "finnhub": bool(APIS["finnhub"]),
            "marketstack": bool(APIS["marketstack"]),
            "twelve_data": bool(APIS["twelve_data"]),
            "binance": bool(APIS["binance"]["key"]),
            "mediastack": bool(APIS["mediastack"])
        },
        "cache_stats": {
            "response_cache": len(RESPONSE_CACHE),
            "stock_data_cache": len(STOCK_DATA_CACHE),
            "fundamentals_cache": len(FUNDAMENTALS_CACHE)
        },
        "uptime": round(time.time() - app.startup_time, 2) if hasattr(app, 'startup_time') else 0
    }

# === Startup ===
@app.on_event("startup")
async def startup_event():
    app.startup_time = time.time()
    print("🚀 Application startup complete")

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)