import os 
import sys
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# === Initialize FastAPI app ===
app = FastAPI(
    title="Unified Financial AI API",
    version="1.0.0",
    description="Combines Stock Forecasting and AI Chatbot for financial insights",
    docs_url="/api/docs"
)
# === Add project root to path ===
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(PROJECT_ROOT)

# === Import routers ===
try:
    from stock.stock_router import router as stock_router
    from routes.chatbot_router import router as chatbot_router
 
except ImportError as e:
    print(f"❌ Import Error: {str(e)}")
    sys.exit(1)



# === Setup CORS ===
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Your React frontend origin
    allow_credentials=True,
    allow_methods=["*"],  # Explicitly allows OPTIONS, POST, etc.
    allow_headers=["*"],  # Must include Authorization
    expose_headers=["*"]
)

# === Register routes ===
app.include_router(stock_router)
app.include_router(chatbot_router)
print("✅ Routers mounted: /stock, /chatbot")

# === Health check endpoint ===
@app.get("/")
def root():
    return {
        "message": "🚀 Unified Stock Forecaster + AI Chatbot API is live",
        "routes": [
            "/stock/predict",
            "/stock/historical",
            "/chatbot/chat",
            "/chatbot/health"
        ]
    }

# === Start server ===
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
