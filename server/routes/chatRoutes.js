import express from "express";
import {
  askGemini,
  translateText,
  healthCheck,
} from "../controller/chatbotController.js";

const router = express.Router();

router.get("/health", healthCheck); // GET /api/health
router.post("/chat", askGemini); // POST /api/chat
router.post("/translate", translateText); // POST /api/translate

export default router;
