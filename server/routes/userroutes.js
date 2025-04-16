import express from "express";
import {
  registerUser,
  loginUser,
  logoutUser,
  getAllUsers,
  getUserById,
  checkAuth,
  updateUserProfile,
  deleteUser,
  upload,
} from "../controller/usercontroller.js";

import { auth, authorizeRoles } from "../Middleware/authMiddleware.js";

const router = express.Router();

/**
 * ✅ PUBLIC ROUTES (No authentication required)
 */
router.post("/signup", registerUser); // Register a new user
router.post("/login", loginUser); // Login and get JWT token
router.post("/logout", logoutUser); // Logout user

/**
 * ✅ PROTECTED ROUTES (Requires authentication)
 */
router.get("/users", auth, getAllUsers); // Get all users (Admin only)
router.get("/users/:userId", auth, getUserById); // Get a specific user by ID
router.get("/checkAuth", checkAuth); // Check authentication status

// ✅ Update user profile (Profile photo upload supported)
router.put(
  "/profile/:userId",
  auth,
  upload.single("profilePhoto"),
  updateUserProfile
);

// ✅ Delete user (Only authorized users can delete)
router.delete("/users/:userId", auth, authorizeRoles("admin"), deleteUser);

/**
 * ✅ ROLE-BASED ROUTES (Restricted Access)
 */
router.get("/admin", auth, authorizeRoles("admin"), (req, res) => {
  res.status(200).json({ message: "Welcome, Admin!" });
});

router.get("/dashboard", auth, authorizeRoles("admin", "user"), (req, res) => {
  res.status(200).json({ message: "Welcome to the Dashboard" });
});

export default router;
