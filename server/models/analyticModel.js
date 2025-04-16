// models/analyticModel.js

import yahooFinance from "yahoo-finance2";

export async function fetchStockData(symbol, period) {
  const periods = {
    "1d": { period1: "1d" },
    "1w": { period1: "7d" },
    "1m": { period1: "1mo" },
    "3m": { period1: "3mo" },
    "6m": { period1: "6mo" },
    "1y": { period1: "1y" },
    "5y": { period1: "5y" },
  };

  if (!periods[period]) throw new Error("Invalid period");

  const queryOptions = { period: periods[period].period1 };

  const result = await yahooFinance.historical(symbol, {
    period1: getStartDate(period),
    interval: "1d",
  });

  return result;
}

function getStartDate(period) {
  const now = new Date();
  const date = new Date(now);

  switch (period) {
    case "1d":
      date.setDate(now.getDate() - 1);
      break;
    case "1w":
      date.setDate(now.getDate() - 7);
      break;
    case "1m":
      date.setMonth(now.getMonth() - 1);
      break;
    case "3m":
      date.setMonth(now.getMonth() - 3);
      break;
    case "6m":
      date.setMonth(now.getMonth() - 6);
      break;
    case "1y":
      date.setFullYear(now.getFullYear() - 1);
      break;
    case "5y":
      date.setFullYear(now.getFullYear() - 5);
      break;
    default:
      throw new Error("Invalid period");
  }

  return date;
}
