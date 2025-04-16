import express from "express";
import {
  handleChatRequest, // ✅ this is the full chatbot
  getStockPredictionByDate, // used for direct call (not in chatbot)
  getStockPriceByDate, // used for direct call (not in chatbot)
} from "../controller/chatbotController.js"; // ✅ fix the path if needed

import { auth } from "../Middleware/authMiddleware.js";

const router = express.Router();

// ✅ Use this for chatbot API
router.post("/chat", handleChatRequest);

// ✅ Optional: For directly calling individual endpoints
router.get("/stock-price", auth, getStockPriceByDate);
router.get("/predict", auth, getStockPredictionByDate);

export default router;
