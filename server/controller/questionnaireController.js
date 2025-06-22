import Questionnaire from "../models/questionnaireModel.js";

// === [1] Submit New Questionnaire (Once per Day) ===
export const submitQuestionnaire = async (req, res) => {
  try {
    const userId = req.user._id;

    // ⏱️ Restrict submissions to once per calendar day
    const startOfDay = new Date();
    startOfDay.setHours(0, 0, 0, 0);

    const existing = await Questionnaire.findOne({
      userId,
      createdAt: { $gte: startOfDay },
    });

    if (existing) {
      return res.status(400).json({
        message: "You have already submitted a questionnaire today.",
      });
    }

    // 📝 Validate required fields
    const requiredFields = [
      "age",
      "employmentStatus",
      "salary",
      "homeOwnership",
      "hasDebt",
      "lifestyle",
      "dependents",
      "financialGoals",
      "riskTolerance",
      "investmentApproach",
      "emergencyPreparedness",
      "financialTracking",
      "futureSecurity",
      "spendingDiscipline",
      "assetAllocation",
      "riskTaking",
    ];

    const missing = requiredFields.filter((field) => !req.body[field]);
    if (missing.length > 0) {
      return res.status(400).json({
        message: "Missing required fields.",
        missingFields: missing,
      });
    }

    // 💾 Save new questionnaire
    const newQuestionnaire = new Questionnaire({
      ...req.body,
      userId,
    });

    const saved = await newQuestionnaire.save();

    res.status(201).json({
      message: "✅ Questionnaire submitted successfully.",
      data: saved,
    });
  } catch (error) {
    console.error("❌ Submit Error:", error.message);
    res.status(500).json({
      message: "Error submitting questionnaire.",
      error: error.message,
    });
  }
};

// === [2] Get Most Recent Questionnaire for Authenticated User ===
// IN controller
export const getUserQuestionnaire = async (req, res) => {
  try {
    const userId = req.query.userId; // ✅ Change this line

    const latest = await Questionnaire.findOne({ userId })
      .sort({ createdAt: -1 })
      .lean();

    if (!latest) {
      return res.status(404).json({ message: "No questionnaire found." });
    }

    res.status(200).json(latest);
  } catch (error) {
    console.error("❌ Get Latest Error:", error.message);
    res.status(500).json({
      message: "Error retrieving questionnaire.",
      error: error.message,
    });
  }
};

// === [3] Admin: Get All Questionnaires by User ID ===
export const getQuestionnairesByUser = async (req, res) => {
  try {
    const { userId } = req.params;

    if (!userId) {
      return res
        .status(400)
        .json({ message: "User ID is required in params." });
    }

    const forms = await Questionnaire.find({ userId }).sort({ createdAt: -1 });

    res.status(200).json(forms);
  } catch (err) {
    console.error("❌ Admin Log Fetch Error:", err.message);
    res.status(500).json({ message: "Error fetching questionnaire logs" });
  }
};
