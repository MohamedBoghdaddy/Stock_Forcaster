import React, { useContext } from "react";
import { ProgressBar } from "react-bootstrap";
import { DashboardContext } from "../../../../../context/DashboardContext";
import { useAuthContext } from "../../../../../context/AuthContext";
import "../../../styles/dashboard.css";

const SavingsProgress = () => {
  const { profileData } = useContext(DashboardContext);
  const { user } = useAuthContext;

  const current = profileData?.currentSavings || 0;
  const goal = profileData?.savingsGoal || 10000;
  const progress = goal ? (current / goal) * 100 : 0;

  return (
    <div className="bg-white dark:bg-gray-800 p-5 rounded-xl shadow-md">
      <h5 className="font-bold text-gray-800 dark:text-white mb-3">
        🎯 {user?.username}'s Savings Goal Progress
      </h5>

      <ProgressBar
        now={progress}
        label={`${progress.toFixed(0)}%`}
        className="mb-3"
        variant="success"
      />
      <div className="d-flex justify-content-between text-sm text-gray-600 dark:text-gray-300 mb-4">
        <span>💵 Saved: ${current.toLocaleString()}</span>
        <span>🏁 Goal: ${goal.toLocaleString()}</span>
      </div>

      {/* 🔟 Feature Highlights */}
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 text-sm text-gray-700 dark:text-gray-200">
        <FeatureItem emoji="🔔" label="Smart Alerts" />
        <FeatureItem emoji="📅" label="Budget Timeline" />
        <FeatureItem emoji="🧾" label="Receipt Tracker" />
        <FeatureItem emoji="🎯" label="Goal Tracker" />
        <FeatureItem emoji="💬" label="AI Coaching" />
        <FeatureItem emoji="📤" label="Report Export" />
        <FeatureItem emoji="🧠" label="Auto Categorization" />
        <FeatureItem emoji="📊" label="Monthly Insights" />
        <FeatureItem emoji="🏦" label="Bank Sync" />
        <FeatureItem emoji="🧮" label="Loan Forecast" />
      </div>
    </div>
  );
};

// Reusable feature display
const FeatureItem = ({ emoji, label }) => (
  <div className="flex items-center gap-2 bg-gray-100 dark:bg-gray-700 px-3 py-2 rounded-md shadow-sm">
    <span className="text-lg">{emoji}</span>
    <span className="font-medium">{label}</span>
  </div>
);

export default SavingsProgress;
