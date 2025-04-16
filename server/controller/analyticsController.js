// controllers/analyticsController.js

import { fetchStockData } from "../models/analyticModel.js";

export const getAnalyticsData = async (req, res) => {
  try {
    const { symbol, period } = req.query;

    if (!symbol || !period) {
      return res.status(400).json({ error: "Symbol and period are required." });
    }

    const data = await fetchStockData(symbol, period);
    return res.json({ symbol, period, data });
  } catch (err) {
    return res.status(500).json({ error: err.message });
  }
};
