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
  toggleBlockStatus,
  updateLoginMeta,
  updateUserRoleOrPassword,
} from "../controller/usercontroller.js";

import { auth, authorizeRoles } from "../Middleware/authMiddleware.js";
import User from "../models/UserModel.js";
import { updateUserRole } from "../controller/usercontroller.js";

const router = express.Router();

/**
 * ✅ PUBLIC ROUTES (No authentication required)
 */
router.post("/signup", registerUser);
router.post("/login", loginUser);
router.post("/logout", logoutUser);

/**
 * ✅ PROTECTED ROUTES (Require user authentication)
 */
router.get("/checkAuth", checkAuth);
router.get("/users", auth, getAllUsers); // Only admin can use this typically
router.get("/users/:userId", auth, getUserById);

router.put(
  "/profile/:userId",
  auth,
  upload.single("profilePhoto"),
  updateUserProfile
);

router.delete("/users/:userId", auth, authorizeRoles("admin"), deleteUser);

/**
 * ✅ ROLE-BASED ACCESS ROUTES
 */
router.get("/admin", auth, authorizeRoles("admin"), (req, res) => {
  res.status(200).json({ message: "Welcome, Admin!" });
});

router.get("/dashboard", auth, authorizeRoles("admin", "user"), (req, res) => {
  res.status(200).json({ message: "Welcome to the Dashboard" });
});

/**
 * ✅ ADMIN DASHBOARD ROUTES (No auth middleware for testing; add it later)
 */
router.get("/all", async (req, res) => {
  try {
    const users = await User.find().sort({ createdAt: -1 });
    res.json(users);
  } catch (err) {
    res.status(500).json({ message: "Error fetching users" });
  }
});

router.delete("/delete/:id", async (req, res) => {
  try {
    const user = await User.findByIdAndDelete(req.params.id);
    if (!user) return res.status(404).json({ message: "User not found" });
    res.json({ success: true });
  } catch (err) {
    res.status(500).json({ message: "Error deleting user" });
  }
});

router.put("/toggle-block/:id", async (req, res) => {
  try {
    const user = await User.findById(req.params.id);
    if (!user) return res.status(404).json({ message: "User not found" });

    user.blocked = !user.blocked;
    await user.save();

    res.json({ success: true, blocked: user.blocked });
  } catch (err) {
    res.status(500).json({ message: "Error toggling block status" });
  }
});

router.put("/meta/login", updateLoginMeta); // Called on login

router.put("/admin/update-user/:id", updateUserRoleOrPassword);

router.put(
  "/admin/update-role/:id",
  auth,
  authorizeRoles("admin"),
  updateUserRole
);

// ✅ Fetch profile of the currently authenticated user
router.get("/profile", auth, async (req, res) => {
  try {
    const user = await User.findById(req.user.id).select("-password");
    if (!user) return res.status(404).json({ message: "User not found" });
    res.status(200).json({ user });
  } catch (err) {
    res.status(500).json({ message: "Error fetching profile" });
  }
});

/**
 * ⚠️ OPTIONAL (if you're using a controller function)
 */
// router.patch("/users/:id/toggle-block", toggleBlockStatus);

export default router;
