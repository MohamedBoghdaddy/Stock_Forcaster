import fetch from "node-fetch";

const geminiUrl = "http://localhost:8000/api/chat";
const healthUrl = "http://localhost:8000/api/health";
const translateUrl = "http://localhost:8000/api/translate";

// Public Gemini Chat Endpoint
export const askGemini = async (req, res) => {
  try {
    const { message } = req.body;
    const response = await fetch(geminiUrl, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message }),
    });

    const data = await response.json();
    res.json({ reply: data.reply });
  } catch (e) {
    console.error("Chat error:", e);
    res.status(500).send("Failed to contact Gemini API");
  }
};

// Public Translate Endpoint
export const translateText = async (req, res) => {
  try {
    const { text, langCode } = req.body;
    const response = await fetch(translateUrl, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text, langCode }),
    });

    const data = await response.json();
    res.json({ translatedText: data.translatedText });
  } catch (e) {
    console.error("Translation error:", e);
    res.status(500).send("Translation failed");
  }
};

// Health check
export const healthCheck = async (req, res) => {
  try {
    const response = await fetch(healthUrl);
    const data = await response.json();
    res.json(data);
  } catch (e) {
    console.error("Health check error:", e);
    res.status(500).send("Health check failed");
  }
};
