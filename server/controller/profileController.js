import { body, validationResult } from "express-validator";
import mongoose from "mongoose";
import Profile from "../models/ProfileModel.js";
import User from "../models/UserModel.js";

/**
 * Validation rules for creating/updating profile
 */
export const profileValidationRules = [
  body("age")
    .optional()
    .isInt({ min: 18, max: 120 })
    .withMessage("Age must be between 18 and 120"),
  body("employmentStatus")
    .optional()
    .isIn(["Employed", "Self-employed", "Unemployed", "Student", "Retired"])
    .withMessage("Invalid employment status"),
  body("salary").optional().isNumeric().withMessage("Salary must be a number"),
  body("financialGoals")
    .optional()
    .isString()
    .trim()
    .escape()
    .isLength({ max: 500 })
    .withMessage("Financial goals cannot exceed 500 characters"),
  body("customExpenses.*.name")
    .optional()
    .isString()
    .trim()
    .isLength({ max: 50 })
    .withMessage("Expense name cannot exceed 50 characters"),
  body("customExpenses.*.amount")
    .optional()
    .isNumeric()
    .withMessage("Expense amount must be a number")
    .custom((value) => value >= 0)
    .withMessage("Expense amount cannot be negative"),
];

/**
 * Helper: Log messages only in development
 */
const devLog = (...args) => {
  if (process.env.NODE_ENV === "development") {
    console.log(...args);
  }
};

/**
 * Get latest profile for authenticated user
 */
export const getLatestProfile = async (req, res) => {
  devLog("Received headers:", req.headers);
  devLog("Authenticated user:", req.user);

  try {
    const profile = await Profile.findOne({ userId: req.user._id })
      .sort({ createdAt: -1 })
      .lean()
      .select("+totalMonthlyExpenses"); // Include virtual if exists

    devLog("Found profile:", profile);

    if (!profile) {
      devLog("No profile found for user:", req.user._id);
      return res.status(404).json({
        success: false,
        message: "Profile not found",
      });
    }

    // Manual total expenses calculation as fallback or for accuracy
    profile.totalMonthlyExpenses =
      profile.customExpenses?.reduce(
        (total, expense) => total + (expense.amount || 0),
        0
      ) || 0;

    devLog("Profile with expenses calculated:", {
      totalMonthlyExpenses: profile.totalMonthlyExpenses,
      customExpensesCount: profile.customExpenses?.length || 0,
    });

    res.status(200).json({
      success: true,
      data: profile,
    });
  } catch (error) {
    console.error("Profile fetch error:", {
      message: error.message,
      stack: error.stack,
      userId: req.user?._id,
    });

    res.status(500).json({
      success: false,
      message: "Internal server error",
      ...(process.env.NODE_ENV === "development" && { error: error.message }),
    });
  }
};

/**
 * Create or update profile for authenticated user
 */
export const createOrUpdateProfile = async (req, res) => {
  // Step 1: Validate incoming request
  const errors = validationResult(req);
  if (!errors.isEmpty()) {
    return res.status(400).json({
      success: false,
      errors: errors.array(),
    });
  }

  try {
    const { _id: userId } = req.user;

    // Step 2: Sanitize input
    const updateData = {
      ...req.body,
      userId,
      lastUpdated: new Date(),
      updatedBy: userId,
    };

    if (Array.isArray(updateData.customExpenses)) {
      updateData.customExpenses = updateData.customExpenses
        .map((exp) => ({
          name: (exp.name || "").trim(),
          amount: Number(exp.amount),
        }))
        .filter((exp) => exp.name && !isNaN(exp.amount));
    }

    // Step 3: Check if it's a creation or update
    const existingProfile = await Profile.findOne({ userId });

    const profile = await Profile.findOneAndUpdate({ userId }, updateData, {
      new: true,
      upsert: true,
      runValidators: true,
      setDefaultsOnInsert: true,
    });

    // Step 4: Calculate total expenses (optional for frontend UI)
    const responseData = profile.toObject();
    responseData.totalMonthlyExpenses = (
      responseData.customExpenses || []
    ).reduce((sum, exp) => sum + (exp.amount || 0), 0);

    // Step 5: Send response
    res.status(200).json({
      success: true,
      message: existingProfile ? "Profile updated" : "Profile created",
      data: responseData,
    });
  } catch (error) {
    console.error("[Profile Controller] Save error:", error);

    let status = 500;
    let message = "Internal server error";

    if (error.name === "ValidationError") {
      status = 400;
      message = "Validation failed: " + error.message;
    } else if (error.code === 11000) {
      status = 409;
      message = "Duplicate key error";
    }

    res.status(status).json({
      success: false,
      message,
      error: process.env.NODE_ENV === "development" ? error.message : undefined,
    });
  }
};

/**
 * Delete profile for authenticated user
 */
export const deleteProfile = async (req, res) => {
  try {
    const { deletedCount } = await Profile.deleteOne({ userId: req.user._id });

    if (deletedCount === 0) {
      return res.status(404).json({
        success: false,
        message: "Profile not found or already deleted",
      });
    }

    res.status(200).json({
      success: true,
      message: "Profile deleted successfully",
    });
  } catch (error) {
    console.error("[Profile Controller] Delete error:", error);
    res.status(500).json({
      success: false,
      message: "Internal server error",
    });
  }
};

/**
 * Get paginated list of all profiles (Admin only)
 */
export const getAllProfiles = async (req, res) => {
  try {
    // Parse pagination query params
    const page = parseInt(req.query.page, 10) || 1;
    const limit = parseInt(req.query.limit, 10) || 10;

    const profiles = await Profile.find()
      .lean()
      .limit(limit)
      .skip((page - 1) * limit)
      .select("+totalMonthlyExpenses")
      .sort({ createdAt: -1 }); // newest first

    const count = await Profile.countDocuments();

    res.status(200).json({
      success: true,
      data: profiles,
      totalPages: Math.ceil(count / limit),
      currentPage: page,
      totalProfiles: count,
    });
  } catch (error) {
    console.error("[Profile Controller] Get all error:", error);
    res.status(500).json({
      success: false,
      message: "Internal server error",
    });
  }
};

/**
 * Get profile by ID with role-based access
 */
export const getProfileById = async (req, res) => {
  try {
    const { id } = req.params;

    if (!mongoose.Types.ObjectId.isValid(id)) {
      return res.status(400).json({
        success: false,
        message: "Invalid profile ID format",
      });
    }

    // Admin can get any profile, others only their own
    const filter = { _id: id };
    if (req.user.role !== "admin") {
      filter.userId = req.user._id;
    }

    const profile = await Profile.findOne(filter)
      .lean()
      .select("+totalMonthlyExpenses");

    if (!profile) {
      return res.status(404).json({
        success: false,
        message: "Profile not found or unauthorized access",
      });
    }

    res.status(200).json({
      success: true,
      data: profile,
    });
  } catch (error) {
    console.error("[Profile Controller] Get by ID error:", error);
    res.status(500).json({
      success: false,
      message: "Internal server error",
    });
  }
};
