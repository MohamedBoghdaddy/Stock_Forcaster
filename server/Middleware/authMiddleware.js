// middleware/authMiddleware.js
import jwt from "jsonwebtoken";
import User from "../models/UserModel.js";
import dotenv from "dotenv";

dotenv.config();

const JWT_SECRET = process.env.JWT_SECRET || "secure_dev_token";

/**
 * ✅ Enhanced Authentication Middleware
 * - Supports both Bearer token and cookie-based authentication
 * - Verifies token against database user (using correct id field)
 * - Detailed logging for debugging
 * - Role-based authorization support
 */
export const auth = async (req, res, next) => {
  try {
    // Get token from header or cookie
    const token =
      req.header("Authorization")?.replace("Bearer ", "") || req.cookies?.token;

    if (!token) {
      console.log("No authentication token provided");
      return res.status(401).json({
        success: false,
        message: "Authentication required",
      });
    }

    // Verify JWT
    const decoded = jwt.verify(token, JWT_SECRET);
    console.log("Decoded token payload:", decoded);

    // Determine user ID from token payload
    const userId = decoded.id || decoded._id;
    if (!userId) {
      console.error("Token payload missing user ID");
      return res.status(401).json({
        success: false,
        message: "Invalid token payload",
      });
    }

    // Build user query
    const userQuery = { _id: userId };

    // Check if tokens array exists in schema, add token check if yes
    if (User.schema.path("tokens")) {
      userQuery["tokens.token"] = token;
    }

    // Find user by ID (and token if applicable)
    const user = await User.findOne(userQuery).select("-password");

    if (!user) {
      console.log("User not found for token", { userId, token });
      return res.status(401).json({
        success: false,
        message: "User not found",
      });
    }

    // Attach user and token to request
    req.token = token;
    req.user = user;
    console.log(`Authenticated user: ${user._id} (${user.role})`);
    next();
  } catch (err) {
    console.error("Authentication error:", {
      message: err.message,
      stack: err.stack,
    });

    let message = "Please authenticate";
    if (err.name === "TokenExpiredError") {
      message = "Session expired. Please login again.";
    } else if (err.name === "JsonWebTokenError") {
      message = "Invalid token";
    }

    res.status(401).json({
      success: false,
      message,
      ...(process.env.NODE_ENV === "development" && { error: err.message }),
    });
  }
};

/**
 * ✅ Role-based Authorization Middleware
 * @param {...string} roles - Allowed roles
 * Example: authorizeRoles("admin", "employee")
 */
export const authorizeRoles = (...roles) => {
  return (req, res, next) => {
    if (!req.user) {
      console.warn("Role check failed - no user in request");
      return res.status(401).json({
        success: false,
        message: "Not authenticated",
      });
    }

    if (!roles.includes(req.user.role)) {
      console.warn(
        `User ${req.user._id} with role ${req.user.role} attempted unauthorized access`
      );
      return res.status(403).json({
        success: false,
        message: `Access denied. Required role(s): ${roles.join(", ")}`,
        yourRole: req.user.role,
      });
    }

    console.log(`Role access granted to ${req.user.role}`);
    next();
  };
};

// Specific role variants
export const verifyAdmin = authorizeRoles("admin");
export const verifyAdminOrEmployee = authorizeRoles("admin", "employee");
export const verifyClient = authorizeRoles("client");
