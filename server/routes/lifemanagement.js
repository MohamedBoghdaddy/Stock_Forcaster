import express from "express";
import LifeManagement from "../models/lifemanagementModel.js";
import axios from "axios";

const router = express.Router();

router.post("/", async (req, res) => {
  try {
    const { userId, ...formData } = req.body;

    // Convert survey to string for model input
    const instruction = JSON.stringify(formData, null, 2);

    // Call AI model (Flask)
    const response = await axios.post(
      "http://127.0.0.1:5001/generate",
      { instruction },
      { headers: { "Content-Type": "application/json" } }
    );

    // ✅ Log what Flask actually returned
    console.log("✅ Flask model raw response:", response.data);

    // Extract model output (assuming Flask returns { output: "..." })
    const modelOutput = response.data.output;

    // Don't save if no userId
    if (!userId) {
      return res.status(200).json({
        lifeGoals: modelOutput,
        note: "userId not provided, response not saved to database.",
      });
    }

    // Save to MongoDB
    const newEntry = new LifeManagement({
      userId,
      formData,
      modelOutput,
    });

    await newEntry.save();

    // ✅ Send model output to frontend
    res.status(200).json({ lifeGoals: modelOutput });
  } catch (error) {
    console.error("❌ LifeManagement Error:", error.message);
    res.status(500).json({
      error: "Internal server error. Make sure the model server is running.",
    });
  }
});

export default router;
