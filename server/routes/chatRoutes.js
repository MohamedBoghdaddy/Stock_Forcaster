import express from "express";
import {
  getStockPredictionByDate,
  getStockPriceByDate,
} from "../controller/chatbotController.js";
import { auth } from "../Middleware/authMiddleware.js"; 

const router = express.Router();

// Route for handling chatbot requests
router.post("/chat", getStockPredictionByDate);
router.get("/chat/history", auth, getStockPriceByDate);

export default router;
