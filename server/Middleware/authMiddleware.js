import jwt from "jsonwebtoken";
import User from "../models/UserModel.js";
import dotenv from "dotenv";

dotenv.config();
const JWT_SECRET = process.env.JWT_SECRET;

/**
 * ✅ Middleware: Ensure User is Authenticated
 * - Checks for a valid JWT token in headers or cookies.
 * - If valid, attaches the user object to `req.user`.
 */
export const auth = async (req, res, next) => {
  try {
    let token = req.headers.authorization;

    // ✅ Extract token from Authorization Header
    if (token?.startsWith("Bearer ")) {
      token = token.split(" ")[1];
    } else if (req.cookies?.token) {
      // ✅ Ensure the cookie-based token is used
      token = req.cookies.token;
    }

    if (!token) {
      return res
        .status(401)
        .json({ message: "Access denied. No token provided." });
    }

    // ✅ Verify Token
    const decoded = jwt.verify(token, JWT_SECRET);
    console.log("✅ Decoded Token:", decoded);

    // ✅ Fetch User from DB
    req.user = await User.findById(decoded.id).select("-password");

    if (!req.user) {
      return res.status(401).json({ message: "Invalid authentication." });
    }

    console.log("✅ Authenticated User:", req.user);
    next();
  } catch (error) {
    console.error("❌ Authentication Error:", error);

    let errorMessage = "Unauthorized access.";
    if (error.name === "TokenExpiredError") {
      errorMessage = "Session expired, please log in again.";
    }

    res.status(401).json({ message: errorMessage });
  }
};


/**
 * ✅ Middleware: Ensure User has Required Roles
 * - Only allows users with specified roles.
 * - Example usage: `authorizeRoles("admin", "employee")`
 */
export const authorizeRoles = (...roles) => {
  return (req, res, next) => {
    if (!req.user) {
      return res.status(401).json({ message: "Unauthorized access." });
    }

    if (!roles.includes(req.user.role)) {
      return res.status(403).json({
        message: `Access denied: Role '${req.user.role}' is not authorized.`,
      });
    }

    console.log(`✅ Authorized Role: ${req.user.role}`);
    next();
  };
};

/**
 * ✅ Middleware: Ensure User is Admin
 * - Allows only users with the role "admin".
 */
export const verifyAdmin = (req, res, next) => {
  return authorizeRoles("admin")(req, res, next);
};

/**
 * ✅ Middleware: Ensure User is Admin or Employee
 * - Allows only users with roles "admin" or "employee".
 */
export const verifyAdminOrEmployee = (req, res, next) => {
  return authorizeRoles("admin", "employee")(req, res, next);
};
