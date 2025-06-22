// ✅ FILE: /server/routes/currencyRoutes.js

import express from "express";
import axios from "axios";
import dotenv from "dotenv";
dotenv.config();

const router = express.Router();

// ✅ Static list of currencies for dropdowns
router.get("/symbols", (req, res) => {
  const symbols = [
    { code: "USD", description: "United States Dollar" },
    { code: "EGP", description: "Egyptian Pound" },
    { code: "EUR", description: "Euro" },
    { code: "GBP", description: "British Pound" },
    { code: "JPY", description: "Japanese Yen" },
    { code: "SAR", description: "Saudi Riyal" },
    { code: "AED", description: "UAE Dirham" },
  ];

  res.json({ success: true, symbols });
});
router.get("/convert", async (req, res) => {
  const { from, to, amount } = req.query;

  if (!from || !to || !amount) {
    return res.status(400).json({
      success: false,
      message: "Missing query parameters: from, to, amount",
    });
  }

  try {
    const response = await axios.get(`https://open.er-api.com/v6/latest/${from}`);
    const rate = response.data?.rates?.[to];
    if (rate) {
      const result = (rate * parseFloat(amount)).toFixed(2);
      return res.json({ success: true, result });
    } else {
      return res.status(500).json({ success: false, message: "Currency not found" });
    }
  } catch (err) {
    console.error("💥 Currency conversion error:", err.message);
    return res.status(500).json({ success: false, message: "API request failed" });
  }
});


export default router;