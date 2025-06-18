from fastapi import APIRouter, Request, HTTPException, FastAPI, Depends
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
import yfinance as yf
import requests
import glob
import time
from cachetools import TTLCache
from dotenv import load_dotenv
from textblob import TextBlob
from scipy.stats import linregress

# Load environment variables
load_dotenv()

# === Setup ===
app = FastAPI()
print("✅ FastAPI application initialized")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
print("🔒 CORS middleware configured")

# Initialize Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-1.5-flash")
print("🧠 Gemini model initialized")

# Configure logging
logging.basicConfig(level=logging.INFO)
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

# Financial data APIs
APIS = {
    "alpha_vantage": os.getenv("ALPHA_VANTAGE_API_KEY"),
    "finnhub": os.getenv("FINNHUB_API_KEY"),
    "marketstack": os.getenv("MARKETSTACK_API_KEY"),
    "twelve_data": os.getenv("TWELVE_DATA_API_KEY"),  # Added Twelve Data
    "binance": {
        "key": os.getenv("BINANCE_API_KEY"),
        "secret": os.getenv("BINANCE_SECRET_KEY")
    },
    "mediastack": os.getenv("MEDIASTACK_API_KEY"),
}
print("🔌 Financial APIs configured")

# Risk categories and stock recommendations
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

# Financial tips and FAQs
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
def get_stock_data(symbol: str) -> dict:
    """Fetch stock data with caching"""
    if symbol in STOCK_DATA_CACHE:
        return STOCK_DATA_CACHE[symbol]
    
    try:
        print(f"📊 Fetching stock data for {symbol}")
        stock = yf.Ticker(symbol)
        info = stock.info
        company_name = info.get('shortName', symbol)
        df = yf.download(symbol, period="30d")
        if df.empty:
            print(f"⚠️ No data found for {symbol}")
            return {}
        
        result = {
            "Name": company_name,
            "Current": df["Close"][-1],
            "Close": df["Close"].tolist(),
            "Volume": df["Volume"].tolist(),
            "High": df["High"].tolist(),
            "Low": df["Low"].tolist()
        }
        
        STOCK_DATA_CACHE[symbol] = result
        return result
    except Exception as e:
        print(f"❌ Error fetching stock data: {str(e)}")
        return {}

def analyze_stock_fundamentals(symbol: str) -> dict:
    """Fetch fundamentals with caching"""
    if symbol in FUNDAMENTALS_CACHE:
        return FUNDAMENTALS_CACHE[symbol]
    
    try:
        stock = yf.Ticker(symbol)
        info = stock.info
        
        # Get key metrics
        pe_ratio = info.get('trailingPE', 0)
        peg_ratio = info.get('pegRatio', 0)
        debt_equity = info.get('debtToEquity', 0)
        roe = info.get('returnOnEquity', 0)
        dividend = info.get('dividendYield', 0) * 100 if info.get('dividendYield') else 0
        beta = info.get('beta', 1)
        profit_margins = info.get('profitMargins', 0) * 100 if info.get('profitMargins') else 0
        
        # Get analyst recommendations
        rec = info.get('recommendationKey', 'none').title()
        target_price = info.get('targetMeanPrice', 0)
        
        result = {
            "pe_ratio": pe_ratio,
            "peg_ratio": peg_ratio,
            "debt_equity": debt_equity,
            "roe": roe,
            "dividend": dividend,
            "beta": beta,
            "profit_margins": profit_margins,
            "recommendation": rec,
            "target_price": target_price
        }
        
        FUNDAMENTALS_CACHE[symbol] = result
        return result
    except Exception as e:
        print(f"❌ Error analyzing fundamentals: {str(e)}")
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
        response = requests.get(
            f"https://api.binance.com/api/v3/ticker/price?symbol={clean_symbol}"
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
    """Fetch stock price using fallback logic: Marketstack → Alpha Vantage → Twelve Data → Yahoo Finance"""
    # === Primary: Marketstack ===
    marketstack_key = APIS.get("marketstack")
    if marketstack_key:
        try:
            print(f"📈 Fetching stock price for {symbol} (Marketstack)")
            response = requests.get(
                "http://api.marketstack.com/v1/eod",
                params={
                    "access_key": marketstack_key,
                    "symbols": symbol,
                    "limit": 1
                },
                timeout=5
            )
            stock_json = response.json()
            if "data" in stock_json and stock_json["data"]:
                stock_data = stock_json["data"][0]
                print(f"✅ Marketstack success: ${stock_data['close']}")
                return f"📈 **{symbol} Price**: **${stock_data['close']}** (as of {stock_data['date']})"
            else:
                print(f"⚠️ Marketstack fallback: {stock_json.get('error', 'No data')}")
        except Exception as e:
            print(f"❌ Marketstack error: {str(e)}")
    else:
        print("⚠️ Marketstack API key not configured")

    # === Fallback 1: Alpha Vantage ===
    alpha_vantage_key = APIS.get("alpha_vantage")
    if alpha_vantage_key:
        try:
            print(f"📉 Trying Alpha Vantage fallback for {symbol}")
            alpha_response = requests.get(
                "https://www.alphavantage.co/query",
                params={
                    "function": "TIME_SERIES_DAILY_ADJUSTED",
                    "symbol": symbol,
                    "apikey": alpha_vantage_key
                },
                timeout=5
            )
            data = alpha_response.json()
            if "Time Series (Daily)" in data:
                latest_date = sorted(data["Time Series (Daily)"].keys())[-1]
                close_price = data["Time Series (Daily)"][latest_date]["4. close"]
                print(f"✅ Alpha Vantage success: ${close_price}")
                return f"📈 **{symbol} Price**: **${close_price}** (as of {latest_date})"
            else:
                error_msg = data.get("Error Message", "No time series data")
                print(f"⚠️ Alpha Vantage fallback: {error_msg}")
        except Exception as e:
            print(f"❌ Alpha Vantage error: {str(e)}")
    else:
        print("⚠️ Alpha Vantage API key not configured")

    # === Fallback 2: Twelve Data ===
    twelve_data_key = APIS.get("twelve_data")
    if twelve_data_key:
        try:
            print(f"📉 Trying Twelve Data fallback for {symbol}")
            twelve_response = requests.get(
                "https://api.twelvedata.com/time_series",
                params={
                    "symbol": symbol,
                    "interval": "1day",
                    "outputsize": 1,
                    "apikey": twelve_data_key
                },
                timeout=5
            )
            data = twelve_response.json()
            if "values" in data and data["values"]:
                latest = data["values"][0]
                print(f"✅ Twelve Data success: ${latest['close']}")
                return f"📈 **{symbol} Price**: **${latest['close']}** (as of {latest['datetime']})"
            else:
                error_msg = data.get("message", "No values data")
                print(f"⚠️ Twelve Data fallback: {error_msg}")
        except Exception as e:
            print(f"❌ Twelve Data error: {str(e)}")
    else:
        print("⚠️ Twelve Data API key not configured")

    # === Fallback 3: Yahoo Finance direct API ===
    try:
        print(f"📉 Trying Yahoo Finance direct API for {symbol}")
        response = requests.get(
            f"https://query1.finance.yahoo.com/v8/finance/chart/{symbol}",
            params={"range": "1d", "interval": "1d"},
            timeout=5
        )
        data = response.json()
        if "chart" in data and "result" in data["chart"]:
            result = data["chart"]["result"][0]
            meta = result["meta"]
            regular_price = meta["regularMarketPrice"]
            print(f"✅ Yahoo Finance direct success: ${regular_price}")
            return f"📈 **{symbol} Price**: **${regular_price}** (latest)"
    except Exception as e:
        print(f"❌ Yahoo Finance direct error: {str(e)}")

    # === Final Fallback: yfinance library with multiple periods ===
    try:
        print(f"📉 Trying yfinance library fallback for {symbol}")
        stock = yf.Ticker(symbol)
        # Try different periods to get data
        for period in ["1d", "5d", "1mo"]:
            hist = stock.history(period=period)
            if not hist.empty:
                close_price = hist["Close"].iloc[-1]
                last_date = hist.index[-1].strftime('%Y-%m-%d')
                print(f"✅ yfinance success: ${close_price} for period {period}")
                return f"📈 **{symbol} Price**: **${close_price:.2f}** (as of {last_date})"
        print(f"⚠️ yfinance returned no data for {symbol} after multiple periods")
    except Exception as e:
        print(f"❌ yfinance error: {str(e)}")

    # If all fail
    return f"⚠️ Sorry, couldn't fetch price for **{symbol}** right now."

async def fetch_currency_rates(base: str = "USD", target: str = "EUR") -> str:
    """Fetch currency exchange rates from Alpha Vantage"""
    alpha_vantage_key = APIS.get("alpha_vantage")
    if not alpha_vantage_key:
        return "⚠️ Alpha Vantage API key not configured"
    
    try:
        print(f"💱 Fetching currency rates: {base} to {target}")
        response = requests.get(
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
        response = requests.get(
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
        response = requests.get(
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
def generate_stock_recommendation(user_profile: dict) -> str:
    """Generate personalized stock recommendation"""
    risk = user_profile.get("riskTolerance", 5)
    investment_horizon = user_profile.get("investmentHorizon", 5)
    print(f"📈 Generating stock recommendation for risk level: {risk}/10")
    
    # Determine risk category
    risk_category = get_risk_category(risk)
    category_data = RISK_CATEGORIES[risk_category]
    
    # Evaluate candidates
    candidates = []
    for symbol in category_data["stocks"]:
        data = get_stock_data(symbol)
        if not data or not data.get('Close'):
            continue
            
        # Calculate metrics
        prices = data['Close']
        current_price = data['Current']
        trend = ((prices[-1] - prices[0]) / prices[0]) * 100
        volatility = pd.Series(prices).pct_change().std() * 100
        fundamentals = analyze_stock_fundamentals(symbol)
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
        
        # === 1. Check for investment advice request (moved up) ===
        if ("invest" in lower_msg or 
            "stock" in lower_msg or 
            "long term" in lower_msg or 
            "what should i buy" in lower_msg or
            "recommend" in lower_msg):
            print("🔍 Investment advice request detected")
            response_text = generate_stock_recommendation(profile)
            print(f"📤 Sending investment recommendation: {response_text[:100]}...")
            return JSONResponse(content={
                "output": response_text,
                "session_id": session_id
            })
        
        # === 2. Check for specific data requests ===
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
        
        # Generate stock recommendation for investment queries
        if ("portfolio" in lower_msg or 
            "diversify" in lower_msg or 
            "buy stock" in lower_msg):
            print("🔍 Investment advice request detected")
            response_text = generate_stock_recommendation(profile)
            print(f"📤 Sending investment recommendation: {response_text[:100]}...")
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
        "model": "gemini-1.5-flash",
        "services": {
            "alpha_vantage": bool(APIS["alpha_vantage"]),
            "finnhub": bool(APIS["finnhub"]),
            "marketstack": bool(APIS["marketstack"]),
            "binance": bool(APIS["binance"]["key"]),
            "mediastack": bool(APIS["mediastack"])
        },
        "cache_stats": {
            "response_cache": len(RESPONSE_CACHE),
            "stock_data_cache": len(STOCK_DATA_CACHE),
            "fundamentals_cache": len(FUNDAMENTALS_CACHE)
        }
    }

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)