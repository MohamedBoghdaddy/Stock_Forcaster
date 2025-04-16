import axios from "axios";
import ChatModel from "../models/ChatModel.js";
import dotenv from "dotenv";
import Sentiment from "sentiment";
import fs from "fs";
import path from "path";
import csv from "csv-parser";

dotenv.config();
const sentiment = new Sentiment();
const checkpointsDir = path.resolve("checkpoints");

// FAQ responses
const faqs = {
  "how to save money": "💰 Save 20% of your income and avoid impulse buying.",
  "how to invest": "📊 Diversify your portfolio using index funds.",
};

// Financial tips
const tips = [
  "📈 Invest for the long-term.",
  "💳 Avoid unnecessary credit card debt.",
  "📉 Track your expenses weekly.",
  "📊 Rebalance your portfolio yearly.",
];

// Helper: Get stock price on a specific date
export const getStockPriceByDate = async (symbol, date) => {
  const file = path.join(checkpointsDir, `${symbol}_future_predictions.csv`);
  if (!fs.existsSync(file)) return `❌ No data found for ${symbol}`;

  return new Promise((resolve, reject) => {
    const results = [];
    fs.createReadStream(file)
      .pipe(csv())
      .on("data", (row) => {
        if (row.Date === date) results.push(row);
      })
      .on("end", () => {
        if (results.length === 0)
          return resolve(`❌ No prediction for ${symbol} on ${date}`);
        resolve(
          `📅 Price for ${symbol} on ${date}: **$${parseFloat(
            results[0]["Predicted Close"]
          ).toFixed(2)}**`
        );
      })
      .on("error", () => resolve("⚠️ Error reading prediction file."));
  });
};

// Helper: Predict stock price in N days
export const getStockPredictionByDate = async (symbol, days) => {
  const file = path.join(checkpointsDir, `${symbol}_future_predictions.csv`);
  if (!fs.existsSync(file)) return `❌ No data found for ${symbol}`;

  return new Promise((resolve, reject) => {
    const rows = [];
    fs.createReadStream(file)
      .pipe(csv())
      .on("data", (row) => rows.push(row))
      .on("end", () => {
        const index = days - 1;
        if (index >= rows.length)
          return resolve(`❌ No prediction ${days} days ahead for ${symbol}`);
        const prediction = rows[index];
        resolve(
          `🔮 Predicted price for ${symbol} in ${days} days (${
            prediction.Date
          }): **$${parseFloat(prediction["Predicted Close"]).toFixed(2)}**`
        );
      })
      .on("error", () => resolve("⚠️ Error reading prediction file."));
  });
};

// ✅ Main chatbot handler
export const handleChatRequest = async (req, res) => {
  const { message } = req.body;
  const userId = req.user?._id || null;

  if (!message)
    return res.status(400).json({ response: "Message is required." });

  let responseText = "";
  const lowerMessage = message.toLowerCase();

  // FAQs
  if (faqs[lowerMessage]) return res.json({ response: faqs[lowerMessage] });

  // Sentiment
  const analysis = sentiment.analyze(message);
  const mood =
    analysis.score > 0
      ? "😊 Positive"
      : analysis.score < 0
      ? "😞 Negative"
      : "😐 Neutral";
  responseText += `🧠 Sentiment: ${mood}`;

  let matched = false;

  // Match: "stock price of AAPL on 2024-04-25"
  const priceRegex = /stock price of (\w+) on (\d{4}-\d{2}-\d{2})/;
  const priceMatch = lowerMessage.match(priceRegex);
  if (priceMatch) {
    const [, symbol, date] = priceMatch;
    const price = await getStockPriceByDate(symbol.toUpperCase(), date);
    responseText += `\n${price}`;
    matched = true;
  }

  // Match: "predict AAPL in 10 days"
  const predictionRegex = /predict (\w+) in (\d{1,2}) days?/;
  const predMatch = lowerMessage.match(predictionRegex);
  if (predMatch) {
    const [, symbol, days] = predMatch;
    const pred = await getStockPredictionByDate(
      symbol.toUpperCase(),
      parseInt(days)
    );
    responseText += `\n${pred}`;
    matched = true;
  }

  // If no match, give a tip
  if (!matched) {
    const tip = tips[Math.floor(Math.random() * tips.length)];
    responseText += `\n💡 Financial Tip: ${tip}`;
  }

  // Save chat
  await new ChatModel({
    userId,
    message,
    response: responseText,
  }).save();

  res.json({ response: responseText });
};
