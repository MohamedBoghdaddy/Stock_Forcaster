import express from "express";
import {
  handleChatRequest,
  getChatByID,
} from "../controller/chatbotController.js";
import { auth } from "../Middleware/authMiddleware.js"; 

const router = express.Router();

// Route for handling chatbot requests
router.post("/chat", handleChatRequest);
router.get("/chat/history", auth, getChatByID);

export default router;
