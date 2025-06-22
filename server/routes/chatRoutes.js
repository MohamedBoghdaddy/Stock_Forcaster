import express from "express";
import {
  handleChatRequest,
  getChatByID,
} from "../controller/chatbotController.js";
import { auth } from "../Middleware/authMiddleware.js"; 
import { getChatsByUser } from "../controller/chatbotController.js";

const router = express.Router();

// Route for handling chatbot requests
router.post("/chat", handleChatRequest);
router.get("/chat/history", auth, getChatByID);

router.get("/user/:userId", getChatsByUser);

export default router;
