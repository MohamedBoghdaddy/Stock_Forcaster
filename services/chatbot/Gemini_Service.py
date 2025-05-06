from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import requests
from dotenv import load_dotenv
from typing import Dict, Any, Optional
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)

# Enhanced CORS configuration
CORS(app, resources={
    r"/api/*": {
        "origins": "*",
        "methods": ["GET", "POST", "OPTIONS", "PUT", "DELETE"],
        "allow_headers": ["Content-Type", "Authorization", "X-Requested-With"],
        "expose_headers": ["Content-Type", "X-Total-Count"],
        "supports_credentials": True,
        "max_age": 3600
    }
})

# Global constants
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent"
MAX_RETRIES = 3
REQUEST_TIMEOUT = 30  # seconds

API_KEY = os.getenv('GEMINI_API_KEY')
if not API_KEY:
    logger.warning("GEMINI_API_KEY not found in environment variables")

@app.after_request
def after_request(response):
    """Add CORS headers to all responses"""
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type, Authorization')
    response.headers.add('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE, OPTIONS')
    response.headers.add('Access-Control-Max-Age', '3600')
    return response

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "Gemini Financial Advisor API",
        "version": "1.0.0"
    })

@app.route('/api/chat', methods=['POST', 'OPTIONS'])
def chat():
    """Handle chat requests with Gemini API"""
    if request.method == 'OPTIONS':
        return _build_cors_preflight_response()
    
    try:
        data: Dict[str, Any] = request.get_json()
        if not data:
            logger.error("No data received in chat request")
            return jsonify({"error": "No data provided"}), 400
            
        message: str = data.get('message', '').strip()
        if not message:
            logger.error("Empty message in chat request")
            return jsonify({"error": "Message is required"}), 400
        
        # Get context from request if available
        context: str = data.get('context', '')
        
        logger.info(f"Processing chat request: {message[:50]}...")
        response = get_gemini_response(message, context)
        
        return jsonify({
            "reply": response,
            "timestamp": datetime.utcnow().isoformat(),
            "status": "success"
        })

    except Exception as e:
        logger.error(f"Chat error: {str(e)}", exc_info=True)
        return jsonify({
            "error": "Internal server error",
            "details": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }), 500

@app.route('/api/translate', methods=['POST', 'OPTIONS'])
def translate():
    """Translation endpoint (placeholder implementation)"""
    if request.method == 'OPTIONS':
        return _build_cors_preflight_response()
    
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400
            
        text = data.get('text', '').strip()
        lang_code = data.get('langCode', 'en').lower()
        
        if not text:
            return jsonify({"error": "Text is required"}), 400
            
        # Here you would integrate with a real translation API
        # This is just a placeholder implementation
        translated_text = f"[TRANSLATED TO {lang_code.upper()}] {text}"
        
        return jsonify({
            "translatedText": translated_text,
            "sourceLanguage": "en",
            "targetLanguage": lang_code,
            "timestamp": datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Translation error: {str(e)}", exc_info=True)
        return jsonify({
            "error": "Translation service unavailable",
            "details": str(e)
        }), 500

def _build_cors_preflight_response():
    """Handle CORS preflight requests"""
    response = jsonify({"message": "CORS preflight approved"})
    response.headers.add("Access-Control-Allow-Origin", "*")
    response.headers.add("Access-Control-Allow-Headers", "*")
    response.headers.add("Access-Control-Allow-Methods", "*")
    return response

def get_gemini_response(prompt: str, context: str = "") -> str:
    """
    Get response from Gemini API with enhanced error handling and retries
    """
    if not API_KEY:
        error_msg = "Gemini API key not configured"
        logger.error(error_msg)
        return error_msg
    
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json"
    }
    
    # Enhanced prompt engineering with context
    system_prompt = (
        "You are a professional financial advisor with 20 years of experience. "
        "Provide concise, accurate advice suitable for the user's context. "
        "If discussing investments, always mention risks. "
        "Format responses with clear paragraphs and bullet points when appropriate."
    )
    
    if context:
        system_prompt += f"\n\nContext from previous conversation: {context}"
    
    payload = {
        "contents": [{
            "parts": [
                {"text": system_prompt},
                {"text": f"User question: {prompt}"}
            ]
        }],
        "safetySettings": {
            "category": "HARM_CATEGORY_FINANCIAL_ADVICE",
            "threshold": "BLOCK_ONLY_HIGH"
        },
        "generationConfig": {
            "temperature": 0.7,
            "topP": 0.9,
            "maxOutputTokens": 1000
        }
    }
    
    params = {"key": API_KEY}
    
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.post(
                GEMINI_API_URL,
                headers=headers,
                json=payload,
                params=params,
                timeout=REQUEST_TIMEOUT
            )
            response.raise_for_status()
            
            data = response.json()
            
            if not data.get('candidates'):
                logger.warning("No candidates in Gemini response")
                return "I couldn't generate a response. Please try again."
                
            candidate = data['candidates'][0]
            if 'content' not in candidate or 'parts' not in candidate['content']:
                logger.error("Malformed Gemini response structure")
                return "There was an issue processing the response."
                
            return candidate['content']['parts'][0]['text']
            
        except requests.exceptions.RequestException as e:
            logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
            if attempt == MAX_RETRIES - 1:
                logger.error("All retries exhausted for Gemini API")
                return f"Sorry, I'm having trouble connecting to the knowledge base. Please try again later. Error: {str(e)}"
    
    return "The service is currently unavailable. Please try again later."

if __name__ == "__main__":
    app.run(
        host='0.0.0.0',
        port=int(os.getenv('PORT', 8000)),
        debug=os.getenv('FLASK_DEBUG', 'false').lower() == 'true'
    )from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import os
import requests
from dotenv import load_dotenv
import logging
from datetime import datetime
from typing import Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="Gemini Financial Advisor API",
    description="API for financial advice using Google's Gemini AI",
    version="1.0.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    max_age=3600
)

# Global constants
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent"
MAX_RETRIES = 3
REQUEST_TIMEOUT = 30  # seconds

API_KEY = os.getenv('GEMINI_API_KEY')
if not API_KEY:
    logger.error("GEMINI_API_KEY not found in environment variables")

# Pydantic models
class HealthCheck(BaseModel):
    status: str
    timestamp: str
    service: str
    version: str

class ChatRequest(BaseModel):
    message: str
    user_id: Optional[str] = None

class ChatResponse(BaseModel):
    reply: str
    timestamp: str
    status: str

class ErrorResponse(BaseModel):
    error: str
    details: Optional[str] = None
    timestamp: str

@app.get("/api/health", response_model=HealthCheck)
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "service": "Gemini Financial Advisor API",
        "version": "1.0.0"
    }

@app.post("/api/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """Handle chat requests with Gemini API"""
    try:
        if not request.message.strip():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Message is required"
            )
        
        logger.info(f"Processing chat request: {request.message[:50]}...")
        
        # Get response from Gemini API
        response = get_gemini_response(request.message)
        
        return {
            "reply": response,
            "timestamp": datetime.utcnow().isoformat(),
            "status": "success"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Chat error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=str(e)
        )

def get_gemini_response(prompt: str) -> str:
    """Get response from Gemini API with enhanced error handling"""
    if not API_KEY:
        return "Error: Gemini API key not configured"
    
    headers = {"Content-Type": "application/json"}
    payload = {
        "contents": [{
            "parts": [{
                "text": f"You are a financial advisor. Provide concise, professional answers to: {prompt}"
            }]
        }],
        "safetySettings": {
            "category": "HARM_CATEGORY_FINANCIAL_ADVICE",
            "threshold": "BLOCK_ONLY_HIGH"
        },
        "generationConfig": {
            "temperature": 0.7,
            "topP": 0.9,
            "maxOutputTokens": 1000
        }
    }
    
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.post(
                f"{GEMINI_API_URL}?key={API_KEY}",
                headers=headers,
                json=payload,
                timeout=REQUEST_TIMEOUT
            )
            response.raise_for_status()
            
            data = response.json()
            if 'candidates' in data and data['candidates']:
                return data['candidates'][0]['content']['parts'][0]['text']
            return "No response generated"
            
        except requests.exceptions.RequestException as e:
            logger.warning(f"Attempt {attempt + 1} failed: {str(e)}")
            if attempt == MAX_RETRIES - 1:
                logger.error("All retries exhausted for Gemini API")
                return f"Error communicating with Gemini API: {str(e)}"
    
    return "Service temporarily unavailable"

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "timestamp": datetime.utcnow().isoformat()
        }
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "gemini_service:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )