import fetch from "node-fetch"; // Use node-fetch for making requests
import jwt from "jsonwebtoken";

// API URL pointing to the Python backend that interacts with Gemini AI
const apiUrl = "http://localhost:5001/chat"; // Python API for Gemini response

// Chat Controller
export const sendChatMessage = async (req, res) => {
  const { message } = req.body;

  // Verify JWT
  const token = req.headers["authorization"];
  if (!token) return res.status(401).send("No token provided");

  try {
    const decoded = jwt.verify(token, process.env.JWT_SECRET);
    console.log("User authenticated: ", decoded.username);

    // Get Gemini response from Python API
    const response = await getGeminiResponse(message);
    return res.json({ reply: response });
  } catch (e) {
    console.error("Error during authentication or processing:", e);
    return res.status(500).send("Authentication failed or error in processing");
  }
};

// Function to get Gemini response from the Python backend
async function getGeminiResponse(prompt) {
  try {
    // Make a POST request to the Python backend
    const response = await fetch(apiUrl, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({ message: prompt }), // Send user message to Gemini service
    });

    if (!response.ok) {
      throw new Error("Failed to get a response from the Gemini service");
    }

    const data = await response.json();
    return data.reply; // Return the reply from Gemini API
  } catch (e) {
    console.error("Error communicating with the Gemini service:", e);
    return "Error communicating with the Gemini service"; // Return a fallback error message
  }
}
