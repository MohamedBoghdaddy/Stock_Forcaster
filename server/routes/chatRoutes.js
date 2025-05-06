// routes/chatRoutes.js
import express from "express";
import { sendChatMessage } from "../controllers/chatController.js";

const router = express.Router();

// Route for sending a chat message
router.post("/chat", sendChatMessage);

export default router;
