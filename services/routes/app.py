from flask import Flask
from flask_cors import CORS
from dotenv import load_dotenv
from chatbot import chatbot_bp  # ✅ Import your blueprint

# ✅ Load environment variables from .env (including GEMINI_API_KEY)
load_dotenv()

app = Flask(__name__)

# ✅ Enable CORS for all origins (adjust if needed later)
CORS(app, resources={r"/*": {"origins": "*"}})

# ✅ Register your Gemini chatbot blueprint
app.register_blueprint(chatbot_bp, url_prefix="/chatbot")

# ✅ Main entry point
if __name__ == "__main__":
    app.run(debug=True, port=5001)
