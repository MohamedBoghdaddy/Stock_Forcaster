import mongoose from "mongoose";

const CustomExpenseSchema = new mongoose.Schema(
  {
    name: {
      type: String,
      required: [true, "Expense name is required"],
      trim: true,
      maxlength: [50, "Expense name cannot exceed 50 characters"],
    },
    amount: {
      type: Number,
      required: [true, "Expense amount is required"],
      min: [0, "Expense amount cannot be negative"],
    },
  },
  { _id: false }
); // Prevent automatic _id creation for subdocuments

const ProfileSchema = new mongoose.Schema(
  {
    userId: {
      type: mongoose.Schema.Types.ObjectId,
      ref: "User",
      required: [true, "User ID is required"],
      index: true,
    },
    name: {
      type: String,
      trim: true,
    },
    email: {
      type: String,
      trim: true,
      lowercase: true,
    },
    income: {
      type: String,
      trim: true,
    },
    financialGoals: {
      type: String,
      trim: true,
      maxlength: [500, "Financial goals cannot exceed 500 characters"],
    },
    age: {
      type: Number,
      min: [18, "Age must be at least 18"],
      max: [120, "Age must be less than 120"],
    },
    occupation: {
      type: String,
      trim: true,
    },
    rent: {
      type: String,
      trim: true,
    },
    utilities: {
      type: String,
      trim: true,
    },
    dietPlan: {
      type: String,
      trim: true,
    },
    transportCost: {
      type: String,
      trim: true,
    },
    otherRecurring: {
      type: String,
      trim: true,
    },
    savingAmount: {
      type: String,
      trim: true,
    },
    customExpenses: {
      type: [CustomExpenseSchema],
      default: [],
    },
    employmentStatus: {
      type: String,
      enum: [
        "Employed",
        "Self-employed",
        "Unemployed",
        "Student",
        "Retired",
        null,
      ],
      trim: true,
    },
    salary: {
      type: Number,
      min: [0, "Salary cannot be negative"],
    },
    homeOwnership: {
      type: String,
      enum: ["Own", "Rent", "Other", null],
      trim: true,
    },
    hasDebt: {
      type: String,
      enum: ["Yes", "No", null],
      trim: true,
    },
    lifestyle: {
      type: String,
      trim: true,
    },
    riskTolerance: {
      type: Number,
      min: 1,
      max: 10,
    },
    investmentApproach: {
      type: Number,
      min: 1,
      max: 10,
    },
    emergencyPreparedness: {
      type: Number,
      min: 1,
      max: 10,
    },
    financialTracking: {
      type: Number,
      min: 1,
      max: 10,
    },
    futureSecurity: {
      type: Number,
      min: 1,
      max: 10,
    },
    spendingDiscipline: {
      type: Number,
      min: 1,
      max: 10,
    },
    assetAllocation: {
      type: Number,
      min: 1,
      max: 10,
    },
    riskTaking: {
      type: Number,
      min: 1,
      max: 10,
    },
    dependents: {
      type: String,
      trim: true,
    },
  },
  {
    timestamps: true,
    toJSON: { virtuals: true },
    toObject: { virtuals: true },
  }
);

// Add index for better query performance
ProfileSchema.index({ userId: 1, createdAt: -1 });

// Virtual for total monthly expenses
ProfileSchema.virtual("totalMonthlyExpenses").get(function () {
  if (!this.customExpenses || this.customExpenses.length === 0) return 0;
  return this.customExpenses.reduce(
    (total, expense) => total + (expense.amount || 0),
    0
  );
});

const Profile = mongoose.model("Profile", ProfileSchema);
export default Profile;
