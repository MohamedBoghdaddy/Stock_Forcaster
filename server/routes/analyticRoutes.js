import express from "express";
import {
  getUserAnalytics,
  getLifestyleAnalytics,
  getRiskToleranceAnalytics,
  getUserStatistics,
} from "../controller/analyticsController.js";
import { auth } from "../Middleware/authMiddleware.js";

const router = express.Router();

// Route to get analytics for a specific user's questionnaire data
router.get("/user", auth, getUserAnalytics);

// Route to get overall lifestyle analytics across all users
router.get("/lifestyle", getLifestyleAnalytics);

// Route to get overall risk tolerance distribution analytics
router.get("/risk-tolerance", getRiskToleranceAnalytics);

// Route to get statistics
router.get("/statistics", getUserStatistics);

export default router;
