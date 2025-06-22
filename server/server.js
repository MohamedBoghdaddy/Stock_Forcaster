// 📦 Core Imports
import express from "express";
import dotenv from "dotenv";
import cookieParser from "cookie-parser";
import cors from "cors";
import mongoose from "mongoose";
import multer from "multer";
import session from "express-session";
import connectMongoDBSession from "connect-mongodb-session";
import path from "path";
import { fileURLToPath } from "url";
import axios from "axios";
import helmet from "helmet";
import rateLimit from "express-rate-limit";
import loanRoutes from './routes/loanRoutes.js';
import expenseRoutes from "./routes/ExpenseRoutes.js";

// 🌍 Route Imports
import userRoutes from "./routes/userroutes.js";
import chatRoutes from "./routes/chatRoutes.js";
import analyticsRoutes from "./routes/analyticRoutes.js";
import profileRoutes from "./routes/profileRoutes.js";
import currencyRoutes from "./routes/currencyRoutes.js";

// 📁 Path & Env Setup
const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);
dotenv.config();

// 🔐 Configuration
const PORT = process.env.PORT || 4000;
const CLIENT_URL = process.env.CLIENT_URL || "http://localhost:3000";
const FLASK_API_BASE_URL = process.env.FLASK_API_URL || "http://localhost:8000";
const JWT_SECRET = process.env.JWT_SECRET || "secure_dev_token";
const MONGO_URL = process.env.MONGO_URL || "mongodb://localhost:27017/financialAI";

// 🚨 Verify Config
if (!MONGO_URL) {
  console.error("❌ MongoDB URI missing.");
  process.exit(1);
}

// 🍃 MongoDB Connection
const connectDB = async () => {
  try {
    await mongoose.connect(MONGO_URL, {
      useNewUrlParser: true,
      useUnifiedTopology: true,
    });
    console.log("✅ MongoDB connected.");
  } catch (error) {
    console.error("❌ MongoDB connection failed:", error);
    process.exit(1);
  }
};
await connectDB();

// 🚀 Express App Init (⚠️ Must come BEFORE routes)
const app = express();

// 🧠 Session Store
const MongoDBStore = connectMongoDBSession(session);
const store = new MongoDBStore({
  uri: MONGO_URL,
  collection: "sessions",
});
store.on("error", (err) => console.error("❌ Session store error:", err));

// 🛡️ Security Middleware
app.use(helmet());
app.use(
  cors({
    origin: [
      CLIENT_URL,
      FLASK_API_BASE_URL,
      "http://localhost:8000",
      "http://127.0.0.1:8000",
    ],
    methods: ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allowedHeaders: ["Content-Type", "Authorization"],
    credentials: true,
  })
);

// 🔐 Rate Limiting
const apiLimiter = rateLimit({
  windowMs: 15 * 60 * 1000, // 15 minutes
  max: 100,
  message: "Too many requests from this IP, please try again later",
});

// 📂 Uploads Configuration
const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    cb(null, "uploads/");
  },
  filename: (req, file, cb) => {
    cb(null, `${Date.now()}-${file.originalname}`);
  },
});
const upload = multer({ storage });

// 🛠️ General Middleware
app.use(express.json());
app.use(express.urlencoded({ extended: true }));
app.use(cookieParser());
app.use(
  session({
    secret: JWT_SECRET,
    resave: false,
    saveUninitialized: false,
    store: store,
    cookie: {
      maxAge: 1000 * 60 * 60 * 24, // 1 day
      httpOnly: true,
      secure: process.env.NODE_ENV === "production",
    },
  })
);

// 🧩 API Routes
app.use("/api/users", userRoutes);
app.use("/api/chat", apiLimiter, chatRoutes);
app.use("/api/analytics", analyticsRoutes);
app.use("/api/profile", profileRoutes);
app.use("/api/currency", currencyRoutes); // ✅ moved after app initialization
app.use('/api/loan', loanRoutes);
app.use("/api/expenses", expenseRoutes);
// 🔄 FastAPI Proxy Configuration
const forwardRequest = async (req, res, endpoint) => {
  try {
    const token = req.headers.authorization || req.cookies.token;
    const response = await axios({
      method: req.method,
      url: `${FLASK_API_BASE_URL}${endpoint}`,
      data: req.body,
      headers: {
        "Content-Type": "application/json",
        ...(token && { Authorization: token }),
      },
      withCredentials: true,
    });
    res.status(response.status).json(response.data);
  } catch (error) {
    console.error(`❌ Proxy Error [${endpoint}]:`, error.message);
    res.status(error?.response?.status || 500).json({
      error: error?.response?.data?.error || "Internal Server Error",
    });
  }
};

// 🔄 Proxy Routes
app.post("/api/chat", (req, res) => forwardRequest(req, res, "/chat"));
app.post("/api/phi/infer", (req, res) => forwardRequest(req, res, "/infer"));
app.post("/api/phi/analyze", (req, res) => forwardRequest(req, res, "/analyze"));
app.post("/api/phi/generate", (req, res) => forwardRequest(req, res, "/generate"));

// ⚛️ Serve React Frontend in Production
if (process.env.NODE_ENV === "production") {
  app.use(express.static(path.join(__dirname, "../client/build")));
  app.get("*", (req, res) => {
    res.sendFile(path.join(__dirname, "../client/build/index.html"));
  });
}

// 🚨 Error Handling Middleware
app.use((err, req, res, next) => {
  console.error("❌ Server Error:", err.stack);
  res.status(500).json({
    error: "Internal Server Error",
    message: process.env.NODE_ENV === "development" ? err.message : undefined,
  });
});

// 🚀 Start Server
app.listen(PORT, () => {
  console.log(`
  🚀 Server is running!
  🔗 Local: http://localhost:${PORT}
  🌐 Client: ${CLIENT_URL}
  🧠 AI API: ${FLASK_API_BASE_URL}
  `);
});
