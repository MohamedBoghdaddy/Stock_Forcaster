import express from "express";
import {
  submitQuestionnaire,
  getUserQuestionnaire,
  getQuestionnairesByUser,
} from "../controller/questionnaireController.js";
import { auth } from "../Middleware/authMiddleware.js";

const router = express.Router();

/**
 * @route   POST /api/questionnaire/submit
 * @desc    Submit a new questionnaire (authenticated user)
 * @access  Private
 */
router.post("/submit", auth, submitQuestionnaire);

/**
 * @route   GET /api/questionnaire/latest
 * @desc    Get the latest questionnaire for the logged-in user
 * @access  Private
 */
router.get("/latest", auth, getUserQuestionnaire);

/**
 * @route   GET /api/questionnaire/user/:userId
 * @desc    Admin: Get all questionnaires submitted by a specific user
 * @access  Admin
 */
router.get("/user/:userId", auth, getQuestionnairesByUser);

export default router;
