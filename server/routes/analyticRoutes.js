// routes/analyticRoutes.js

import express from "express";
import { getAnalyticsData } from "../controller/analyticsController.js";

const router = express.Router();

router.get("/analytics", getAnalyticsData);

export default router;
