from fastapi import APIRouter, Request, HTTPException, Depends
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
router = APIRouter(prefix="/chatbot", tags=["Chatbot"])


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
try:
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
    model = genai.GenerativeModel("gemini-1.5-flash")
except Exception as e:
    raise RuntimeError(f"Gemini initialization failed: {str(e)}")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create caches
RESPONSE_CACHE = TTLCache(maxsize=1000, ttl=300)  # 5-minute cache
STOCK_DATA_CACHE = TTLCache(maxsize=200, ttl=3600)  # 1-hour cache
FUNDAMENTALS_CACHE = TTLCache(maxsize=200, ttl=86400)  # 24-hour cache

# Session management
SESSION_HISTORY = TTLCache(maxsize=1000, ttl=3600)  # 1-hour session cache
SESSION_LOCK = asyncio.Lock()

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

AI_INSIGHTS_PATH = "ai_insights.json"

# === Helper Functions ===
def load_ai_insights() -> Dict[str, Dict]:
    """Load predicted returns and volatility from file with caching"""
    try:
        if os.path.exists(AI_INSIGHTS_PATH):
            with open(AI_INSIGHTS_PATH, "r") as f:
                return json.load(f)
    except Exception:
        logger.warning("Failed to load AI insights file")
    
    # Default values if file can't be loaded
    return {
        "predicted_returns": {
            "stocks": "8.9%",
        },
        "market_volatility": {
            "stocks": "0.06",
        }
    }

async def fetch_twelve_data(symbol: str, days: int) -> dict:
    """Fetch data from Twelve Data API"""
    api_key = APIS.get("twelve_data")
    if not api_key:
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
        logger.error(f"Twelve Data error: {str(e)}")
        return None

async def fetch_marketstack_data(symbol: str, days: int) -> dict:
    """Fetch data from Marketstack API"""
    api_key = APIS.get("marketstack")
    if not api_key:
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
        logger.error(f"Marketstack error: {str(e)}")
        return None

async def fetch_alpha_vantage_data(symbol: str, days: int) -> dict:
    """Fetch data from Alpha Vantage API"""
    api_key = APIS.get("alpha_vantage")
    if not api_key:
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
        logger.error(f"Alpha Vantage error: {str(e)}")
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
            logger.warning(f"{api_name} failed: {str(e)}")
            continue
    
    return await get_local_stock_data(symbol)


async def fetch_multiple_stocks(symbols: List[str]) -> Dict[str, Any]:
    """Fetch multiple stocks in parallel"""
    tasks = [fetch_stock_data(sym) for sym in symbols]
    results = await asyncio.gather(*tasks)
    return {sym: result for sym, result in zip(symbols, results) if result}

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
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"https://api.binance.com/api/v3/ticker/price?symbol={clean_symbol}",
                timeout=5
            )
        data = response.json()
        if 'price' in data:
            return f"🚀 **{symbol.upper()} Price**: **${data['price']}**"
        return ""
    except Exception as e:
        logger.error(f"Crypto price error: {str(e)}")
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
            logger.warning(f"{api_name} price fetch failed: {str(e)}")
            continue
    
    return f"⚠️ Sorry, couldn't fetch price for **{symbol}** right now."

async def fetch_currency_rates(base: str = "USD", target: str = "EUR") -> str:
    """Fetch currency exchange rates from Alpha Vantage"""
    alpha_vantage_key = APIS.get("alpha_vantage")
    if not alpha_vantage_key:
        return "⚠️ Alpha Vantage API key not configured"
    
    try:
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
            return f"💱 **Exchange Rate**: **1 {base} = {rate} {target}**"
        else:
            return ""
    except Exception as e:
        logger.error(f"Currency error: {str(e)}")
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
            return f"🥇 **{metal.capitalize()} Price**: **${data['c']} per ounce**"
        else:
            return ""
    except Exception as e:
        logger.error(f"Metal price error: {str(e)}")
        return ""

async def fetch_finance_news() -> str:
    """Fetch financial news from Mediastack"""
    mediastack_key = APIS.get("mediastack")
    if not mediastack_key:
        return "⚠️ Mediastack API key not configured"
    
    try:
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
            news_items = [
                f"📰 **{art['title']}**\n{art['description']}\n🔗 [Read more]({art['url']})"
                for art in articles
            ]
            return "📢 **Latest Financial News**:\n\n" + "\n\n".join(news_items)
        else:
            return ""
    except Exception as e:
        logger.error(f"News error: {str(e)}")
        return ""

def analyze_sentiment(text: str) -> str:
    """Analyze text sentiment using TextBlob"""
    analysis = TextBlob(text)
    if analysis.sentiment.polarity > 0.1:
        return "😊 Positive"
    elif analysis.sentiment.polarity < -0.1:
        return "😞 Negative"
    return "😐 Neutral"

# === Enhanced Recommendation Engine ===
async def generate_stock_recommendation(user_profile: dict) -> str:
    """Generate personalized stock recommendation with parallel data fetching"""
    risk = user_profile.get("riskTolerance", 5)
    investment_horizon = user_profile.get("investmentHorizon", 5)
    
    # Load AI insights
    ai_insights = load_ai_insights()
    predicted_return = ai_insights["predicted_returns"].get("stocks", "N/A")
    volatility = ai_insights["market_volatility"].get("stocks", "N/A")
    
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
        technicals = calculate_technical_metrics(prices)
        
        # Skip if too volatile for category
        if volatility > category_data["max_volatility"]:
            continue
            
        # Calculate composite score
        score = 0
        if technicals.get('trend_slope') and technicals['trend_slope'] > 0:
            score += 3
        if technicals.get('rsi') and 30 < technicals['rsi'] < 70:
            score += 2
        if current_price > technicals.get('sma_20', 0):
            score += 2
            
        candidates.append({
            "symbol": symbol,
            "price": current_price,
            "trend": trend,
            "volatility": volatility,
            "technicals": technicals,
            "score": score
        })
    
    if not candidates:
        return "I couldn't find suitable investments matching your profile right now. Please try again later."
    
    # Select best candidate
    best_stock = sorted(candidates, key=lambda x: x['score'], reverse=True)[0]
    technicals = best_stock['technicals']
    
    # Generate recommendation report
    report = (
        f"📈 **Recommendation for {risk_category.title()} Investor**\n\n"
        f"**{best_stock['symbol']}**\n"
        f"Current Price: ${best_stock['price']:.2f}\n"
        f"30-Day Trend: {best_stock['trend']:.1f}%\n"
        f"Volatility: {best_stock['volatility']:.1f}%\n\n"
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
        f"- Set stop-loss at {best_stock['price'] * 0.9:.2f} (10% below current)\n"
        f"- Market-wide predicted return: {predicted_return}\n"
        f"- Market volatility: {volatility}"
    )
    
    return report

async def generate_asset_recommendation(asset_type: str, user_profile: dict) -> str:
    """Generate recommendation for non-stock assets"""
    # Load AI insights
    ai_insights = load_ai_insights()
    predicted_return = ai_insights["predicted_returns"].get(asset_type, "N/A")
    volatility = ai_insights["market_volatility"].get(asset_type, "N/A")
    
    # Generate recommendation based on asset type
    if asset_type == "real_estate":
        return (
            f"🏠 **Real Estate Investment Recommendation**\n\n"
            f"- Predicted Return: {predicted_return}\n"
            f"- Market Volatility: {volatility}\n\n"
            "💡 **Why Real Estate?**\n"
            "Real estate provides stable long-term growth and acts as a hedge against inflation. "
            "Consider properties in developing areas with good infrastructure projects."
        )
    elif asset_type == "gold":
        return (
            f"🥇 **Gold Investment Recommendation**\n\n"
            f"- Predicted Return: {predicted_return}\n"
            f"- Market Volatility: {volatility}\n\n"
            "💡 **Why Gold?**\n"
            "Gold is a safe-haven asset that preserves value during market turmoil. "
            "Allocate 5-10% of your portfolio to gold for diversification."
        )
    else:
        return "⚠️ Unsupported asset type"

# === Models ===
class ChatRequest(BaseModel):
    message: str
    profile: Optional[Dict] = {}
    session_id: Optional[str] = ""

class AdviceRequest(BaseModel):
    profile: Dict
    goal: str

# === Routes ===
@router.post("/chat")
async def chat_with_bot(request: ChatRequest):
    try:
        user_message = request.message.strip()
        profile = request.profile or {}
        session_id = request.session_id or ""
        
        if not user_message:
            raise HTTPException(status_code=400, detail="Empty message")
        
        # Create cache key
        cache_key = hashlib.sha256(
            f"{user_message}-{json.dumps(profile, sort_keys=True)}".encode()
        ).hexdigest()
        
        # Return cached response if available
        if cache_key in RESPONSE_CACHE:
            return JSONResponse(content={
                "output": RESPONSE_CACHE[cache_key],
                "session_id": session_id
            })
        
        # Get or create session history
        async with SESSION_LOCK:
            if session_id and session_id in SESSION_HISTORY:
                conversation_history = SESSION_HISTORY[session_id]
                # Maintain only last 3 exchanges (6 messages)
                if len(conversation_history) > 6:
                    conversation_history = conversation_history[-6:]
            else:
                # Generate new session ID
                session_id = hashlib.sha256(f"{time.time()}{user_message}".encode()).hexdigest()[:16]
                conversation_history = []
            
            # Update session history
            conversation_history.append({"role": "user", "content": user_message})
            SESSION_HISTORY[session_id] = conversation_history
        
        lower_msg = user_message.lower()
        
        # Check FAQs
        if lower_msg in FAQS:
            response_text = FAQS[lower_msg]
            RESPONSE_CACHE[cache_key] = response_text
            return JSONResponse(content={
                "output": response_text,
                "session_id": session_id
            })
        
        # Analyze sentiment
        sentiment = analyze_sentiment(user_message)
        response_parts = [f"🔍 **Sentiment**: {sentiment}"]
        
        # Handle asset-specific recommendations
        asset_query = None
        if "real estate" in lower_msg or "property" in lower_msg:
            asset_query = "real_estate"
        elif "gold" in lower_msg or "precious metal" in lower_msg:
            asset_query = "gold"
        
        if asset_query:
            response_text = await generate_asset_recommendation(asset_query, profile)
            RESPONSE_CACHE[cache_key] = response_text
            return JSONResponse(content={
                "output": response_text,
                "session_id": session_id
            })
        
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
            response_text = await generate_stock_recommendation(profile)
            RESPONSE_CACHE[cache_key] = response_text
            return JSONResponse(content={
                "output": response_text,
                "session_id": session_id
            })
        
        # Check for specific data requests
        if "crypto" in lower_msg or "bitcoin" in lower_msg or "btc" in lower_msg:
            crypto = "BTC" if "btc" in lower_msg else "ETH"
            response_parts.append(await fetch_crypto_price(crypto))
        
        if "stock" in lower_msg:
            symbol = "AAPL"  # Default
            if "microsoft" in lower_msg or "msft" in lower_msg:
                symbol = "MSFT"
            elif "google" in lower_msg or "googl" in lower_msg:
                symbol = "GOOGL"
            elif "amazon" in lower_msg or "amzn" in lower_msg:
                symbol = "AMZN"
            response_parts.append(await fetch_stock_price(symbol))
        
        if "currency" in lower_msg or "exchange" in lower_msg:
            response_parts.append(await fetch_currency_rates())
        
        if "gold" in lower_msg or "silver" in lower_msg:
            metal = "gold" if "gold" in lower_msg else "silver"
            response_parts.append(await fetch_metal_prices(metal))
        
        if "news" in lower_msg:
            response_parts.append(await fetch_finance_news())
        
        # If we have API responses, use them
        if len(response_parts) > 1:
            response_text = "\n\n".join(response_parts)
            RESPONSE_CACHE[cache_key] = response_text
            return JSONResponse(content={
                "output": response_text,
                "session_id": session_id
            })
        
        # Default financial advice using Gemini
        context_prompt = """
You are an expert financial advisor. Follow these rules:

1. Scope:
   - Only respond to personal finance questions
   - For non-finance topics: "I specialize in financial advice."

2. Personalization:
   - Consider user profile: {profile}
   - Maintain conversational context

3. Response Style:
   - Be engaging and concise (2-3 sentences)
   - Use natural language with occasional emojis

CONVERSATION HISTORY:
{history}

USER QUESTION:
"{current_message}"
"""
        # Format history for prompt
        history_text = "\n".join(
            [f"{msg['role'].capitalize()}: {msg['content']}" 
            for msg in conversation_history[:-1]]
        )
        
        formatted_prompt = context_prompt.format(
            profile=json.dumps(profile, indent=2),
            history=history_text,
            current_message=user_message
        )
        
        # Generate response
        response = model.generate_content(formatted_prompt)
        response_text = response.text.strip()
        
        # Add financial tip if response is short
        if len(response_text.split()) < 30:
            tip = FINANCIAL_TIPS[int(time.time()) % len(FINANCIAL_TIPS)]
            response_text += f"\n\n💡 **Financial Tip**: {tip}"
        
        # Update conversation history and cache
        async with SESSION_LOCK:
            conversation_history.append({"role": "assistant", "content": response_text})
            SESSION_HISTORY[session_id] = conversation_history
        
        RESPONSE_CACHE[cache_key] = response_text
        return JSONResponse(content={
            "output": response_text,
            "session_id": session_id
        })
        
    except Exception as e:
        logger.exception(f"Chat error: {str(e)}")
        return JSONResponse(
            content={"error": "Financial advice service unavailable. Please try again later."},
            status_code=500
        )

@router.post("/generate-advice")
async def generate_advice(request: AdviceRequest):
    try:
        profile = request.profile
        goal = request.goal
        
        # Create cache key
        cache_key = hashlib.sha256(
            f"{goal}-{json.dumps(profile, sort_keys=True)}".encode()
        ).hexdigest()
        
        # Return cached response if available
        if cache_key in RESPONSE_CACHE:
            return JSONResponse(content={"advice": RESPONSE_CACHE[cache_key]})
        
        # Load AI insights
        ai_insights = load_ai_insights()
        predicted_returns = ai_insights["predicted_returns"]
        market_volatility = ai_insights["market_volatility"]
        
        # Build dynamic prompt
        prompt = f"""
You are a professional financial advisor. Your task:

1. Analyze user's financial health:
   - Income vs expenses
   - Investment capacity
   - Risk tolerance

2. Provide goal-specific advice:
"""
        if goal == "investment":
            prompt += (
                "- Recommend ONE primary investment strategy\n"
                "- Justify using income, lifestyle, risk tolerance\n"
                "- Use engaging language with 1-2 relevant emojis\n"
            )
        else:  # life_management
            prompt += (
                "- Provide 3 personalized financial wellness tips\n"
                "- Focus on budgeting, saving, and cost-cutting\n"
                "- Use bullet points with simple language\n"
            )
            
        prompt += f"""
3. Use natural language without markdown

USER PROFILE:
"""
        # Add profile details
        profile_fields = [
            ("Monthly Salary", profile.get("salary", "Not provided"), "EGP"),
            ("Home Ownership", profile.get("homeOwnership", "Not specified")),
            ("Utilities", profile.get("utilities", 0), "EGP"),
            ("Transport", profile.get("transportCost", 0), "EGP"),
            ("Recurring Expenses", profile.get("otherRecurring", "None")),
            ("Lifestyle", profile.get("lifestyle", "Standard")),
            ("Risk Tolerance", f"{profile.get('riskTolerance', 5)}/10"),
            ("Dependents", profile.get("dependents", 0)),
            ("Financial Goals", profile.get("financialGoals", "Not specified"))
        ]
        
        for field in profile_fields:
            prompt += f"- {field[0]}: {field[1]}"
            if len(field) > 2:
                prompt += f" {field[2]}"
            prompt += "\n"
        
        # Add market insights
        prompt += "\nMARKET INSIGHTS:\n"
        for asset in ["real_estate", "stocks", "gold"]:
            asset_return = predicted_returns.get(asset, "N/A")
            asset_volatility = market_volatility.get(asset, "N/A")
            prompt += (
                f"- {asset.replace('_', ' ').title()}: "
                f"Predicted Return = {asset_return}, "
                f"Volatility = {asset_volatility}\n"
            )
        
        prompt += "\nADVICE:"
        
        # Generate and cache response
        response = model.generate_content(prompt)
        response_text = response.text.strip()
        RESPONSE_CACHE[cache_key] = response_text
        
        return JSONResponse(content={"advice": response_text})
        
    except Exception as e:
        logger.exception(f"Advice generation error: {str(e)}")
        return JSONResponse(
            content={"error": "Financial planning service unavailable."},
            status_code=500
        )

# === Health Check ===
@router.get("/health")
async def health_check():
    # Test API connectivity
    api_status = {
        "alpha_vantage": bool(APIS["alpha_vantage"]),
        "finnhub": bool(APIS["finnhub"]),
        "marketstack": bool(APIS["marketstack"]),
        "twelve_data": bool(APIS["twelve_data"]),
        "binance": bool(APIS["binance"]["key"]),
        "mediastack": bool(APIS["mediastack"])
    }
    
    # Test stock data loading
    test_stock = await fetch_stock_data('AAPL', 1)
    
    return {
        "status": "operational",
        "version": "2.0.0",
        "model": "gemini-1.5-flash",
        "services": {
            "stock_data": bool(test_stock),
            "ai_insights": os.path.exists(AI_INSIGHTS_PATH),
            "api_connectivity": api_status
        },
        "cache_stats": {
            "response_cache": len(RESPONSE_CACHE),
            "stock_data_cache": len(STOCK_DATA_CACHE),
            "fundamentals_cache": len(FUNDAMENTALS_CACHE),
            "active_sessions": len(SESSION_HISTORY)
        }
    }

