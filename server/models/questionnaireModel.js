import mongoose from "mongoose";

const questionnaireSchema = new mongoose.Schema({
  userId: { type: mongoose.Schema.Types.ObjectId, ref: "User", required: true }, // Reference to the user
  age: { type: Number, required: true },
  employmentStatus: { type: String, required: true },
  salary: { type: Number, required: true },
  homeOwnership: { type: String, required: true },
  hasDebt: { type: String, required: true },
  lifestyle: { type: String, required: true },
  dependents: { type: String, required: true },
  financialGoals: { type: String, required: true },

  // Slider-based answers (stored as Numbers for easy calculations)
  riskTolerance: { type: Number, required: true }, // 1-10
  investmentApproach: { type: Number, required: true }, // 1-10
  emergencyPreparedness: { type: Number, required: true }, // 1-10
  financialTracking: { type: Number, required: true }, // 1-10
  futureSecurity: { type: Number, required: true }, // 1-10
  spendingDiscipline: { type: Number, required: true }, // 1-10
  assetAllocation: { type: Number, required: true }, // 1-10
  riskTaking: { type: Number, required: true }, // 1-10

  createdAt: { type: Date, default: Date.now },
});

const Questionnaire = mongoose.model("Questionnaire", questionnaireSchema);

export default Questionnaire;
