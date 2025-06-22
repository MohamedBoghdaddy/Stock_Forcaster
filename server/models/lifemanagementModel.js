// 📁 models/LifeManagement.js
import mongoose from "mongoose";

const LifeManagementSchema = new mongoose.Schema({
  userId: { type: mongoose.Schema.Types.ObjectId, ref: "User", required: true },
  formData: { type: Object, required: true },
  modelOutput: { type: Array }, // optional: e.g., [{ goal: "...", details: "...", priority: "High" }]
  createdAt: { type: Date, default: Date.now }
});

export default mongoose.model("LifeManagement", LifeManagementSchema);
