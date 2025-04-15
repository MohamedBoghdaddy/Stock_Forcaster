import Questionnaire from "../models/questionnaireModel.js";

// Submit a new questionnaire response associated with the logged-in user
export const submitQuestionnaire = async (req, res) => {
  try {
    const userId = req.user._id;

    // Check if the user has already submitted a questionnaire today
    const startOfDay = new Date();
    startOfDay.setHours(0, 0, 0, 0);

    const existingQuestionnaire = await Questionnaire.findOne({
      userId,
      createdAt: { $gte: startOfDay },
    });

    if (existingQuestionnaire) {
      return res.status(400).json({
        message: "You can only submit the questionnaire once per day.",
      });
    }

    // Save new questionnaire
    const questionnaire = new Questionnaire({
      ...req.body,
      userId,
    });

    await questionnaire.save();

    res.status(201).json({
      message: "Questionnaire submitted successfully.",
      data: questionnaire,
    });
  } catch (error) {
    res.status(400).json({
      message: "Error submitting questionnaire.",
      error: error.message,
    });
  }
};

// Get the latest questionnaire submitted by the logged-in user
export const getUserQuestionnaire = async (req, res) => {
  try {
    const userId = req.user._id;

    const latestQuestionnaire = await Questionnaire.findOne({ userId })
      .sort({ createdAt: -1 }) // Get the most recent one
      .limit(1);

    if (!latestQuestionnaire) {
      return res.status(404).json({ message: "No questionnaire found." });
    }

    res.status(200).json(latestQuestionnaire);
  } catch (error) {
    res.status(500).json({
      message: "Error retrieving questionnaire.",
      error: error.message,
    });
  }
};
