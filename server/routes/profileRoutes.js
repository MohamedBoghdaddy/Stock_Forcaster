// 📁 routes/profileRoutes.js
import express from "express";
import rateLimit from "express-rate-limit";

import {
  getLatestProfile,
  createOrUpdateProfile,
  deleteProfile,
  getProfileById,
  getAllProfiles,
} from "../controller/profileController.js";

import { auth, authorizeRoles } from "../Middleware/authMiddleware.js";
import { profileValidationRules } from "../validators/profileValidator.js";

// Rate limiter for profile routes
const profileLimiter = rateLimit({
  windowMs: 15 * 60 * 1000, // 15 minutes
  max: 100, // max 100 requests per IP per window
  message: "Too many profile requests from this IP, please try again later",
});

const router = express.Router();

// Apply authentication middleware to all profile routes
router.use(auth);

// GET current user's profile (Private)
router.get("/me", async (req, res) => {
  console.log("Route /me hit"); // Debug log
  return getLatestProfile(req, res);
});
// POST create or update profile (Private)
router.post("/", profileLimiter, profileValidationRules, createOrUpdateProfile);

// DELETE profile (Private)
router.delete("/", profileLimiter, deleteProfile);

// GET profile by ID (Admin only)
router.get("/:id", profileLimiter, authorizeRoles("admin"), getProfileById);

// GET all profiles (Admin only)
router.get("/", profileLimiter, authorizeRoles("admin"), getAllProfiles);

export default router;
