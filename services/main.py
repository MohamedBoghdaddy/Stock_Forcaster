import os
import threading
from subprocess import Popen
import time

def run_gemini_service():
    """Start the Gemini chatbot service (FastAPI)"""
    Popen(["uvicorn", "gemini_service:app", "--host", "0.0.0.0", "--port", "8000"])

def run_stock_forecaster():
    """Start the Stock Forecaster service (FastAPI)"""
    Popen(["uvicorn", "Stock_Forecaster:app", "--host", "0.0.0.0", "--port", "5001"])

if __name__ == "__main__":
    # Run both services in separate threads
    thread1 = threading.Thread(target=run_gemini_service)
    thread2 = threading.Thread(target=run_stock_forecaster)
    
    # Start both threads
    thread1.start()
    thread2.start()

    # Give services time to start
    time.sleep(2)
    print("Services should be running now:")
    print("- Gemini Service: http://localhost:8000")
    print("- Stock Forecaster: http://localhost:5001")
    print("\nGemini API Docs: http://localhost:8000/docs")

    # Keep main thread alive
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down services...")