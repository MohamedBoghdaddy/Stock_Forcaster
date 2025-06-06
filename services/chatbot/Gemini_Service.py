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
from textblob import TextBlob  # For sentiment analysis

# Load environment variables
load_dotenv()

# === Setup ===
app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
# Initialize Gemini
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))
model = genai.GenerativeModel("gemini-1.5-flash")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create response cache
RESPONSE_CACHE = TTLCache(maxsize=1000, ttl=300)  # 5-minute cache

# Session management for conversation history
SESSION_HISTORY = TTLCache(maxsize=1000, ttl=3600)  # 1-hour session cache
SESSION_LOCK = asyncio.Lock()

# Financial data APIs
APIS = {
    "alpha_vantage": os.getenv("ALPHA_VANTAGE_API_KEY"),
    "finnhub": os.getenv("FINNHUB_API_KEY"),
    "marketstack": os.getenv("MARKETSTACK_API_KEY"),
    "binance": {
        "key": os.getenv("BINANCE_API_KEY"),
        "secret": os.getenv("BINANCE_SECRET_KEY")
    },
    "mediastack": os.getenv("MEDIASTACK_API_KEY")
}

# Popular stock symbols to track
POPULAR_STOCKS = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META',
    'TSLA', 'NVDA', 'JPM', 'V', 'WMT',
    'DIS', 'NFLX', 'PYPL', 'ADBE', 'INTC'
]

# Financial tips and FAQs
FINANCIAL_TIPS = [
    "💰 Save at least 20% of your income each month.",
    "📉 Avoid impulse buying by waiting 24 hours before making a purchase.",
    "📊 Invest in diversified assets to reduce risk.",
    "🏦 Use high-yield savings accounts for emergency funds.",
    "💳 Pay off high-interest debt as soon as possible to avoid extra fees.",
]

FAQS = {
    "how to save money": "💰 Save at least 20% of your income each month and avoid impulse purchases.",
    "best way to invest": "📊 Diversify your investments and consider low-cost index funds.",
    "how to improve credit score": "✅ Pay bills on time and keep credit utilization below 30%.",
    "how to start budgeting": "📋 Track your expenses and allocate your income into savings, needs, and wants.",
}

# === API Helper Functions ===
async def fetch_crypto_price(symbol: str) -> str:
    """Fetch cryptocurrency price from Binance"""
    clean_symbol = symbol.replace(r"\W+", "", symbol).upper() + "USDT"
    try:
        response = requests.get(
            f"https://api.binance.com/api/v3/ticker/price?symbol={clean_symbol}"
        )
        data = response.json()
        return f"🚀 **{symbol.upper()} Price**: **${data['price']}**" if 'price' in data else ""
    except Exception as e:
        logger.error(f"Crypto price error: {str(e)}")
        return ""

async def fetch_stock_price(symbol: str) -> str:
    """Fetch stock price from MarketStack"""
    try:
        response = requests.get(
            "http://api.marketstack.com/v1/eod",
            params={
                "access_key": APIS["marketstack"],
                "symbols": symbol,
                "limit": 1
            }
        )
        stock_data = response.json()["data"][0]
        return (
            f"📈 **Stock Price for {symbol}**: "
            f"**${stock_data['close']}** (as of {stock_data['date']})"
        )
    except Exception as e:
        logger.error(f"Stock price error: {str(e)}")
        return ""

async def fetch_currency_rates(base: str = "USD", target: str = "EUR") -> str:
    """Fetch currency exchange rates from Alpha Vantage"""
    try:
        response = requests.get(
            "https://www.alphavantage.co/query",
            params={
                "function": "CURRENCY_EXCHANGE_RATE",
                "from_currency": base,
                "to_currency": target,
                "apikey": APIS["alpha_vantage"]
            }
        )
        rate = response.json()["Realtime Currency Exchange Rate"]["5. Exchange Rate"]
        return f"💱 **Exchange Rate**: **1 {base} = {rate} {target}**"
    except Exception as e:
        logger.error(f"Currency error: {str(e)}")
        return ""

async def fetch_metal_prices(metal: str) -> str:
    """Fetch metal prices from Finnhub"""
    metal_symbols = {"GOLD": "GC1!", "SILVER": "SI1!"}
    try:
        response = requests.get(
            "https://finnhub.io/api/v1/quote",
            params={
                "symbol": metal_symbols.get(metal.upper()),
                "token": APIS["finnhub"]
            }
        )
        data = response.json()
        return f"🥇 **{metal.capitalize()} Price**: **${data['c']} per ounce**"
    except Exception as e:
        logger.error(f"Metal price error: {str(e)}")
        return ""

async def fetch_finance_news() -> str:
    """Fetch financial news from Mediastack"""
    try:
        response = requests.get(
            "http://api.mediastack.com/v1/news",
            params={
                "access_key": APIS["mediastack"],
                "categories": "business",
                "languages": "en",
                "limit": 5
            }
        )
        articles = response.json()["data"]
        news_items = [
            f"📰 **{art['title']}**\n{art['description']}\n🔗 [Read more]({art['url']})"
            for art in articles
        ]
        return "📢 **Latest Financial News**:\n\n" + "\n\n".join(news_items)
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

# === Investment Recommendation Engine ===
def generate_stock_recommendation(user_profile: dict) -> str:
    """Generate stock recommendation based on user profile"""
    risk = user_profile.get("riskTolerance", 5)
    
    # Get recommended stock based on risk profile
    low_risk = ["MSFT", "JPM", "V", "WMT"]
    med_risk = ["AAPL", "GOOGL", "DIS", "PYPL"]
    high_risk = ["TSLA", "NVDA", "NFLX", "META"]
    
    if risk < 4:
        symbol = low_risk[0]
        reason = "stable blue-chip companies with consistent dividends"
    elif risk < 7:
        symbol = med_risk[0]
        reason = "balanced growth stocks with moderate volatility"
    else:
        symbol = high_risk[0]
        reason = "high-growth potential stocks"
    
    data = get_stock_data(symbol)
    if not data:
        return "I recommend Microsoft (MSFT) for its strong cloud business and consistent growth."
    
    return (
        f"📈 Based on your risk profile (level {risk}/10), "
        f"I recommend **{data['Name']} ({symbol})**.\n\n"
        f"**Why?**\n"
        f"- Current price: ${data['Current']:.2f}\n"
        f"- 30-day trend: {((data['Close'][-1] - data['Close'][0])/data['Close'][0]*100):.1f}%\n"
        f"- Ideal for investors seeking {reason}"
    )

# === Core Chat Endpoint ===
class ChatRequest(BaseModel):
    message: str
    profile: Optional[Dict] = {}
    session_id: Optional[str] = ""

@app.post("/chat")
async def chat_with_bot(request: ChatRequest):
    try:
        user_message = request.message.strip()
        profile = request.profile or {}
        session_id = request.session_id or ""
        
        if not user_message:
            raise HTTPException(status_code=400, detail="Empty message")
        
        # Check FAQs
        lower_msg = user_message.lower()
        if lower_msg in FAQS:
            return JSONResponse(content={
                "output": FAQS[lower_msg],
                "session_id": session_id
            })
        
        # Analyze sentiment
        sentiment = analyze_sentiment(user_message)
        
        # Check for specific data requests
        response_parts = [f"🔍 **Sentiment**: {sentiment}"]
        
        if "crypto" in lower_msg or "bitcoin" in lower_msg or "btc" in lower_msg:
            crypto = "BTC" if "btc" in lower_msg else "ETH"
            response_parts.append(await fetch_crypto_price(crypto))
        
        elif "stock" in lower_msg:
            symbol = "AAPL"  # Default to Apple
            if "microsoft" in lower_msg or "msft" in lower_msg:
                symbol = "MSFT"
            response_parts.append(await fetch_stock_price(symbol))
        
        elif "currency" in lower_msg or "exchange" in lower_msg:
            response_parts.append(await fetch_currency_rates())
        
        elif "gold" in lower_msg or "silver" in lower_msg:
            metal = "gold" if "gold" in lower_msg else "silver"
            response_parts.append(await fetch_metal_prices(metal))
        
        elif "news" in lower_msg:
            response_parts.append(await fetch_finance_news())
        
        # If we have API responses, use them
        if len(response_parts) > 1:
            response_text = "\n\n".join(response_parts)
            return JSONResponse(content={
                "output": response_text,
                "session_id": session_id
            })
        
        # Generate stock recommendation for investment queries
        if "invest" in lower_msg or "stock" in lower_msg or "long term" in lower_msg:
            response_text = generate_stock_recommendation(profile)
            return JSONResponse(content={
                "output": response_text,
                "session_id": session_id
            })
        
        # Default financial advice using Gemini
        prompt = (
            "You are a certified financial advisor. "
            "Provide concise investment advice (1-2 paragraphs). "
            "Focus on stocks, ETFs, or long-term investments. "
            f"User profile: {profile}\n\n"
            f"Question: {user_message}"
        )
        
        response = model.generate_content(prompt)
        response_text = response.text.strip()
        
        # Add financial tip if response is short
        if len(response_text.split()) < 30:
            tip = FINANCIAL_TIPS[time.time() % len(FINANCIAL_TIPS)]
            response_text += f"\n\n💡 **Financial Tip**: {tip}"
        
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

# === Health Check ===
# Routes
@app.get("/health")
async def health_check():
    return {
        "status": "operational",
        "model": "gemini-1.5-flash",
        "services": {
            "alpha_vantage": bool(APIS["alpha_vantage"]),
            "finnhub": bool(APIS["finnhub"]),
            "marketstack": bool(APIS["marketstack"]),
            "binance": bool(APIS["binance"]["key"]),
            "mediastack": bool(APIS["mediastack"])
        }
    }
@app.post("/chat")
async def chat_with_bot(request: ChatRequest):
    try:
        prompt = (
            "You are a certified financial advisor. "
            f"User profile: {request.profile or {}}\n\n"
            f"Question: {request.message.strip()}"
        )
        response = model.generate_content(prompt)
        reply = response.text.strip()

        if len(reply.split()) < 30:
            reply += "\n\n💡 Financial Tip: Save before you spend."

        return JSONResponse(content={"output": reply, "session_id": request.session_id})

    except Exception as e:
        logger.exception("Chat error")
        return JSONResponse(
            status_code=500,
            content={"error": "Service unavailable"}
        )

