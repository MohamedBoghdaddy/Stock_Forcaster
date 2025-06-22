import User from "../models/UserModel.js";
import Questionnaire from "../models/questionnaireModel.js";

// Analytics for user's financial and lifestyle questionnaires
export const getUserAnalytics = async (req, res) => {
  const userId = req.user.id;

  try {
    const user = await User.findById(userId);
    if (!user) return res.status(404).json({ message: "User not found" });

    const totalQuestionnaires = await Questionnaire.countDocuments({ userId });

    const questionnairesByRiskTolerance = await Questionnaire.aggregate([
      { $match: { userId } },
      { $group: { _id: "$riskTolerance", count: { $sum: 1 } } },
    ]);

    const analytics = {
      totalQuestionnaires,
      questionnairesByRiskTolerance,
    };

    res.status(200).json(analytics);
  } catch (error) {
    res.status(500).json({ message: error.message });
  }
};

// Analytics for lifestyle and financial preferences across all users
export const getLifestyleAnalytics = async (req, res) => {
  try {
    const analyticsData = await Questionnaire.aggregate([
      { $group: { _id: "$lifestyle", totalUsers: { $sum: 1 } } },
      { $sort: { totalUsers: -1 } },
    ]);

    res.status(200).json(analyticsData);
  } catch (error) {
    console.error("Failed to fetch lifestyle analytics:", error);
    res
      .status(500)
      .json({ message: "Failed to fetch lifestyle analytics", error });
  }
};

// Analytics for risk tolerance distribution
export const getRiskToleranceAnalytics = async (req, res) => {
  try {
    const analyticsData = await Questionnaire.aggregate([
      { $group: { _id: "$riskTolerance", totalUsers: { $sum: 1 } } },
      { $sort: { totalUsers: -1 } },
    ]);

    res.status(200).json(analyticsData);
  } catch (error) {
    console.error("Failed to fetch risk tolerance analytics:", error);
    res
      .status(500)
      .json({ message: "Failed to fetch risk tolerance analytics", error });
  }
};

//Statistics
export const getUserStatistics = async (req, res) => {
  try {
    const totalUsers = await User.countDocuments();
    const totalAdmins = await User.countDocuments({ role: "admin" });
    const totalRegularUsers = await User.countDocuments({ role: "user" });

    res.json({
      totalUsers,
      totalAdmins,
      totalRegularUsers,
    });
  } catch (err) {
    console.error("Error in getUserStatistics:", err.message);
    res.status(500).json({ error: "Failed to load statistics" });
  }
};
