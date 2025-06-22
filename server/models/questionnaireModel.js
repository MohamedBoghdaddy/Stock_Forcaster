import mongoose from "mongoose";

const { Schema, model } = mongoose;

// 🔐 Enumerations for select-type fields
const employmentOptions = [
  "Employed",
  "Self-employed",
  "Unemployed",
  "Student",
  "Retired",
];
const homeOwnershipOptions = ["Own", "Rent", "Other"];
const debtOptions = ["Yes", "No"];
const lifestyleOptions = ["Minimalist", "Balanced", "Spender"];
const dependentsOptions = ["Yes", "No"];

// 🧠 Questionnaire Schema
const questionnaireSchema = new Schema(
  {
    userId: {
      type: Schema.Types.ObjectId,
      ref: "User",
      required: [true, "User ID is required"],
      index: true,
    },

    // 🔢 Basic Demographics
    age: {
      type: Number,
      min: [18, "Age must be at least 18"],
      max: [120, "Age seems unrealistic"],
      required: true,
    },
    employmentStatus: {
      type: String,
      enum: employmentOptions,
      required: true,
    },
    salary: {
      type: Number,
      min: [0, "Salary cannot be negative"],
      required: true,
    },
    homeOwnership: {
      type: String,
      enum: homeOwnershipOptions,
      required: true,
    },
    hasDebt: {
      type: String,
      enum: debtOptions,
      required: true,
    },
    dependents: {
      type: String,
      enum: dependentsOptions,
      required: true,
    },

    // 🧬 Lifestyle & Goals
    lifestyle: {
      type: String,
      enum: lifestyleOptions,
      required: true,
    },
    financialGoals: {
      type: String,
      maxlength: 1000,
      required: true,
    },

    // 📊 Slider-based Ratings (1-10)
    riskTolerance: {
      type: Number,
      min: 1,
      max: 10,
      required: true,
    },
    investmentApproach: {
      type: Number,
      min: 1,
      max: 10,
      required: true,
    },
    emergencyPreparedness: {
      type: Number,
      min: 1,
      max: 10,
      required: true,
    },
    financialTracking: {
      type: Number,
      min: 1,
      max: 10,
      required: true,
    },
    futureSecurity: {
      type: Number,
      min: 1,
      max: 10,
      required: true,
    },
    spendingDiscipline: {
      type: Number,
      min: 1,
      max: 10,
      required: true,
    },
    assetAllocation: {
      type: Number,
      min: 1,
      max: 10,
      required: true,
    },
    riskTaking: {
      type: Number,
      min: 1,
      max: 10,
      required: true,
    },

    // 🕒 Timestamp
    createdAt: {
      type: Date,
      default: Date.now,
    },
  },
  {
    timestamps: true, // adds createdAt & updatedAt
  }
);

// ✅ Model Export
const Questionnaire = model("Questionnaire", questionnaireSchema);

export default Questionnaire;
