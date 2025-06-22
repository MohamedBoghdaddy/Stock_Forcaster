import axios from "axios";
import Binance from "node-binance-api";
import ChatModel from "../models/ChatModel.js";
import dotenv from "dotenv";
import Sentiment from "sentiment";

dotenv.config();
const sentiment = new Sentiment();

// **Initialize Binance API**
const binance = new Binance().options({
  APIKEY: process.env.BINANCE_API_KEY,
  APISECRET: process.env.BINANCE_SECRET_KEY,
});

// **Financial Tips**
const financialTips = [
  "💰 Save at least 20% of your income each month.",
  "📉 Avoid impulse buying by waiting 24 hours before making a purchase.",
  "📊 Invest in diversified assets to reduce risk.",
  "🏦 Use high-yield savings accounts for emergency funds.",
  "💳 Pay off high-interest debt as soon as possible to avoid extra fees.",
];

// **FAQs**
const faqs = {
  "how to save money":
    "💰 Save at least 20% of your income each month and avoid impulse purchases.",
  "best way to invest":
    "📊 Diversify your investments and consider low-cost index funds.",
  "how to improve credit score":
    "✅ Pay bills on time and keep credit utilization below 30%.",
  "how to start budgeting":
    "📋 Track your expenses and allocate your income into savings, needs, and wants.",
};

// **Fetch Crypto Price from Binance**
const fetchCryptoPrice = async (symbol) => {
  try {
    const cleanSymbol = symbol.replace(/\W/g, "").toUpperCase() + "USDT";
    console.log(`Fetching Binance price for: ${cleanSymbol}`);

    const { data } = await axios.get(
      `https://api.binance.com/api/v3/ticker/price?symbol=${cleanSymbol}`
    );

    return data.price
      ? `🚀 **${symbol.toUpperCase()} Price**: **$${data.price}**`
      : `❌ No price data found for ${symbol}`;
  } catch (error) {
    console.error(`Error fetching ${symbol} price:`, error.message);
    return "❌ Unable to fetch crypto price.";
  }
};

// **Fetch Stock Price from MarketStack**
const fetchStockPrice = async (symbol) => {
  try {
    const response = await axios.get(`http://api.marketstack.com/v1/eod`, {
      params: {
        access_key: process.env.MARKETSTACK_API_KEY,
        symbols: symbol,
        limit: 1,
      },
    });

    const stockData = response.data.data[0];
    if (!stockData) return `❌ No stock data found for ${symbol}`;

    return `📈 **Stock Price for ${symbol}**: **$${stockData.close}** (as of ${stockData.date})`;
  } catch (error) {
    console.error(`Error fetching stock price for ${symbol}:`, error.message);
    return "❌ Unable to fetch stock market data.";
  }
};

// **Fetch Currency Exchange Rates**
const fetchCurrencyRates = async (base = "USD", target = "EUR") => {
  try {
    const { data } = await axios.get(
      `https://www.alphavantage.co/query?function=CURRENCY_EXCHANGE_RATE&from_currency=${base}&to_currency=${target}&apikey=${process.env.ALPHA_VANTAGE_API_KEY}`
    );

    const rate = data["Realtime Currency Exchange Rate"]["5. Exchange Rate"];
    return `💱 **Exchange Rate**: **1 ${base} = ${rate} ${target}**`;
  } catch (error) {
    console.error("Error fetching currency rates:", error);
    return "❌ Unable to fetch currency exchange rates.";
  }
};

// **Fetch Metal Prices (Gold & Silver)**
const fetchMetalPrices = async (metal) => {
  const metalSymbols = { GOLD: "GC1!", SILVER: "SI1!" };

  if (!metalSymbols[metal]) return `❌ No data available for ${metal}`;

  try {
    const { data } = await axios.get(`https://finnhub.io/api/v1/quote`, {
      params: {
        symbol: metalSymbols[metal],
        token: process.env.FINNHUB_API_KEY,
      },
    });

    return data.c
      ? `🥇 **${metal} Price**: **$${data.c} per ounce**`
      : `❌ No price data found for ${metal}`;
  } catch (error) {
    console.error(`Error fetching ${metal} price:`, error.message);
    return "❌ Unable to fetch metal prices.";
  }
};

// **Fetch Global Finance News from Mediastack**
const fetchFinanceNews = async () => {
  try {
    const response = await axios.get(`http://api.mediastack.com/v1/news`, {
      params: {
        access_key: process.env.MEDIASTACK_API_KEY,
        categories: "business",
        languages: "en",
        limit: 20,
      },
    });

    const financeKeywords = [
      "finance",
      "investment",
      "stock",
      "market",
      "economy",
      "cryptocurrency",
      "banking",
      "trading",
      "money",
      "debt",
      "recession",
      "inflation",
      "interest rates",
      "exchange rates",
      "bonds",
      "financial crisis",
      "Wall Street",
      "Federal Reserve",
      "gold prices",
      "forex",
    ];

    const filteredArticles = response.data.data.filter((article) =>
      financeKeywords.some(
        (keyword) =>
          (article.title && article.title.toLowerCase().includes(keyword)) ||
          (article.description &&
            article.description.toLowerCase().includes(keyword))
      )
    );

    if (filteredArticles.length === 0)
      return "❌ No relevant finance news found.";

    return (
      `📢 **Latest Finance News:**\n\n` +
      filteredArticles
        .map(
          (article) =>
            `🔹 **${article.title}**\n📰 ${
              article.description || "No description available"
            }\n🔗 [Read more](${article.url})`
        )
        .join("\n\n")
    );
  } catch (error) {
    console.error("Error fetching finance news:", error.message);
    return "❌ Unable to fetch finance news.";
  }
};

// **Chatbot Handler**
export const handleChatRequest = async (req, res) => {
  const { message } = req.body;
  const userId = req.user ? req.user._id : null;

  if (!message) {
    return res.status(400).json({ message: "Message cannot be empty." });
  }

  const lowerMessage = message.toLowerCase().trim();

  if (faqs[lowerMessage]) return res.json({ response: faqs[lowerMessage] });

  try {
    const sentimentResult = sentiment.analyze(message);
    const sentimentLabel =
      sentimentResult.score > 0
        ? "😊 Positive"
        : sentimentResult.score < 0
        ? "😞 Negative"
        : "😐 Neutral";

    let responseText = `🔍 **Sentiment Analysis**: ${sentimentLabel}\n🧐 **Analyzing financial data for**: ${message}`;
    let foundRelevantData = false;

    if (/btc|eth|crypto price/i.test(lowerMessage)) {
      const cryptoSymbol =
        lowerMessage.match(/(?:btc|eth|crypto price of )(\w+)/)?.[1] || "BTC";
      responseText += `\n${await fetchCryptoPrice(cryptoSymbol)}`;
      foundRelevantData = true;
    }

    if (/stock price/i.test(lowerMessage)) {
      const stockSymbol =
        lowerMessage.match(/stock price of (\w+)/)?.[1]?.toUpperCase() ||
        "AAPL";
      responseText += `\n${await fetchStockPrice(stockSymbol)}`;
      foundRelevantData = true;
    }

    if (/exchange rate|currency/i.test(lowerMessage)) {
      responseText += `\n${await fetchCurrencyRates()}`;
      foundRelevantData = true;
    }

    if (/gold|silver|metal prices/i.test(lowerMessage)) {
      responseText += `\n${await fetchMetalPrices(lowerMessage.toUpperCase())}`;
      foundRelevantData = true;
    }

    if (/finance news|business news/i.test(lowerMessage)) {
      responseText += `\n${await fetchFinanceNews()}`;
      foundRelevantData = true;
    }

    if (!foundRelevantData)
      responseText += `\n💡 **Financial Tip**: ${
        financialTips[Math.floor(Math.random() * financialTips.length)]
      }`;

    // **Save chat to MongoDB, including userId if found**
    const chatEntry = new ChatModel({
      userId: userId, // Store user ID if available, otherwise set to null
      message,
      response: responseText,
    });

    await chatEntry.save();
    res.json({ response: responseText });
  } catch (error) {
    console.error("Error processing request:", error);
    res.status(500).json({ response: "⚠️ Error fetching financial data." });
  }
};

// **✅ Get Chat History by User ID**
export const getChatByID = async (req, res) => {
  try {
    const userId = req.user._id; // Extract user ID from token

    if (!userId) {
      return res.status(400).json({ message: "User ID is required." });
    }

    // **Find chats by user ID and sort by timestamp (latest first)**
    const chats = await ChatModel.find({ userId }).sort({ timestamp: -1 });

    if (chats.length === 0) {
      return res
        .status(404)
        .json({ message: "No chat history found for this user." });
    }

    res.status(200).json(chats);
  } catch (error) {
    console.error("Error fetching chat history:", error);
    res.status(500).json({ message: "Error retrieving chat history." });
  }
};

export const getChatsByUser = async (req, res) => {
  try {
    const chats = await AIChat.find({ userId: req.params.userId }).sort({
      createdAt: -1,
    });
    res.json(chats);
  } catch (err) {
    res.status(500).json({ message: "Error fetching chat logs" });
  }
};
