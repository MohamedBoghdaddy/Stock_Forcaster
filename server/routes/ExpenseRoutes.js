import express from "express";
import Expense from "../models/ExpenseModel.js";

const router = express.Router();

// GET all expenses (use req.query.userId if no auth)
router.get("/", async (req, res) => {
  const { userId } = req.query;
  console.log("🔍 Incoming userId:", userId);

  try {
    const expenses = await Expense.find({ userId }).sort({ date: -1 });
    console.log("📦 Expenses found:", expenses.length);
    res.json(expenses);
  } catch (err) {
    res.status(500).json({ error: "Failed to fetch expenses." });
  }
});

// POST a new expense
router.post("/", async (req, res) => {
  const { userId, description, amount, date } = req.body;

  try {
    const newExpense = new Expense({ userId, description, amount, date });
    await newExpense.save();
    res.status(201).json(newExpense);
  } catch (err) {
    res.status(400).json({ error: "Failed to add expense." });
  }
});
// DELETE a specific expense
router.delete("/:id", async (req, res) => {
  try {
    const { id } = req.params;
    const deleted = await Expense.findByIdAndDelete(id);
    if (!deleted) {
      return res.status(404).json({ error: "Expense not found" });
    }
    res.json({ message: "Expense deleted successfully" });
  } catch (err) {
    res.status(500).json({ error: "Failed to delete expense" });
  }
});

export default router;
