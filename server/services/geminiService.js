require("dotenv").config();
const express = require("express");
const cors = require("cors");
const axios = require("axios");

const app = express();
const PORT = process.env.PORT || 4000;

// Constants
const GEMINI_API_KEY = process.env.GEMINI_API_KEY;
const GEMINI_API_URL =
  "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent";
const MAX_RETRIES = 3;
const REQUEST_TIMEOUT = 30000;

// Middleware
app.use(cors());
app.use(express.json());

// Health check route
app.get("/api/health", (req, res) => {
  res.json({
    status: "healthy",
    service: "Gemini Financial Advisor",
    timestamp: new Date().toISOString(),
  });
});

// Chat route
app.post("/api/chat", async (req, res) => {
  const { message, context = "" } = req.body;

  if (!message || !message.trim()) {
    return res.status(400).json({ error: "Message is required" });
  }

  try {
    const reply = await getGeminiResponse(message, context);
    res.json({
      reply,
      timestamp: new Date().toISOString(),
      status: "success",
    });
  } catch (error) {
    res.status(500).json({ error: error.message });
  }
});

// Dummy Translate route
app.post("/api/translate", (req, res) => {
  const { text, langCode } = req.body;
  res.json({
    translatedText: `[${langCode}] ${text}`,
  });
});

// Gemini API interaction
async function getGeminiResponse(userPrompt, context = "") {
  if (!GEMINI_API_KEY) return "❌ Gemini API key not configured.";

  const headers = { "Content-Type": "application/json" };

  let systemPrompt =
    "You are a financial advisor with 20+ years of experience.\n" +
    "Provide accurate, clear responses. Mention risks in investments.";

  if (context) {
    systemPrompt += `\n\nContext: ${context}`;
  }

  const payload = {
    contents: [
      {
        parts: [{ text: systemPrompt }, { text: `User: ${userPrompt}` }],
      },
    ],
    safetySettings: [
      {
        category: "HARM_CATEGORY_FINANCIAL_ADVICE",
        threshold: "BLOCK_ONLY_HIGH",
      },
    ],
    generationConfig: {
      temperature: 0.7,
      topP: 0.9,
      maxOutputTokens: 1000,
    },
  };

  for (let attempt = 1; attempt <= MAX_RETRIES; attempt++) {
    try {
      const response = await axios.post(
        `${GEMINI_API_URL}?key=${GEMINI_API_KEY}`,
        payload,
        { headers, timeout: REQUEST_TIMEOUT }
      );

      return (
        response.data?.candidates?.[0]?.content?.parts?.[0]?.text ||
        "No response."
      );
    } catch (err) {
      console.warn(`❗ Attempt ${attempt} failed:`, err.message);
      if (attempt === MAX_RETRIES) {
        throw new Error(`Gemini API error: ${err.message}`);
      }
    }
  }

  return "❌ Service temporarily unavailable.";
}

// Start server
app.listen(PORT, () =>
  console.log(`🚀 Server running on http://localhost:${PORT}`)
);
