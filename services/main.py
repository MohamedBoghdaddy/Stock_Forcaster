from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from Stock_Forecaster import app as stock_app
from chatbot.Gemini_Service import app as gemini_app

app = FastAPI(title="Unified Financial AI API")

# Add CORS to main app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount sub-apps
app.mount("/stock", stock_app)
app.mount("/chatbot", gemini_app)

print("Sub-apps loaded:", app.routes)


@app.get("/")
def root():
    return {
        "message": "🚀 Unified Stock Forecaster + AI Chatbot API is live",
        "routes": ["/stock/predict", "/stock/historical", "/chatbot/chat", "/chatbot/health"]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
