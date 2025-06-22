import React, { useContext } from "react";
import { useLocation } from "react-router-dom";
import { DashboardContext } from "../../../../context/DashboardContext";
import { useAuthContext } from "../../../../context/AuthContext";

const FinancialReport = () => {
  const location = useLocation();
  const { profileData } = useContext(DashboardContext);
  const { user } = useAuthContext;

  const aiOutput = location.state?.output;

  if (!aiOutput) {
    return (
      <div className="p-6 text-red-600 text-lg font-semibold">
        ❌ No report data found. Please fill the Life Management form again.
      </div>
    );
  }

  let parsedOutput;

  try {
    const cleaned =
      typeof aiOutput === "string"
        ? aiOutput
            .replace(/^```json/, "")
            .replace(/```$/, "")
            .replace(/\\n/g, "\n")
            .trim()
        : aiOutput;

    parsedOutput = typeof cleaned === "string" ? JSON.parse(cleaned) : cleaned;
  } catch (err) {
    console.error("❌ Failed to parse AI response:", err);
    return (
      <p className="text-red-600 text-lg">Invalid response format from AI.</p>
    );
  }

  const { summary, advice } = parsedOutput;

  const salary = profileData?.salary ?? 0;
  const expenses = profileData?.expenses ?? 0;
  const savings = salary - expenses;

  return (
    <div className="bg-gradient-to-br from-white to-blue-50 shadow-2xl p-10 mt-12 rounded-3xl border border-gray-300 max-w-5xl mx-auto space-y-10 animate-fade-in">
      {/* Header */}
      <div className="text-center">
        <h1 className="text-4xl font-bold text-blue-800 mb-2 flex items-center justify-center gap-2">
          📊 AI Financial Report
        </h1>
        <p className="text-gray-500 text-md">
          Hello {user?.username}, here are your personalized insights powered by
          AI
        </p>
      </div>

      {/* Profile Overview */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 text-center">
        <div className="bg-white p-4 rounded-lg shadow">
          <h3 className="text-sm font-semibold text-gray-500">Salary</h3>
          <p className="text-xl font-bold text-green-600">
            EGP {salary.toLocaleString()}
          </p>
        </div>
        <div className="bg-white p-4 rounded-lg shadow">
          <h3 className="text-sm font-semibold text-gray-500">Expenses</h3>
          <p className="text-xl font-bold text-red-600">
            EGP {expenses.toLocaleString()}
          </p>
        </div>
        <div className="bg-white p-4 rounded-lg shadow">
          <h3 className="text-sm font-semibold text-gray-500">Savings</h3>
          <p className="text-xl font-bold text-blue-600">
            EGP {savings.toLocaleString()}
          </p>
        </div>
      </div>

      {/* Summary */}
      <div className="bg-blue-50 border-l-4 border-blue-500 p-6 rounded-xl shadow-inner">
        <h2 className="text-2xl font-semibold text-blue-700 mb-3 flex items-center gap-2">
          💡 Summary
        </h2>
        <textarea
          className="w-full p-4 border border-blue-200 rounded-md resize-none bg-white text-gray-800 font-medium leading-relaxed shadow-sm"
          value={summary}
          readOnly
          rows="5"
        />
      </div>

      {/* Recommendations */}
      <div className="bg-green-50 border-l-4 border-green-500 p-6 rounded-xl shadow-inner">
        <h2 className="text-2xl font-semibold text-green-700 mb-4 flex items-center gap-2">
          ✅ Recommendations
        </h2>
        <div className="space-y-6">
          {Array.isArray(advice) && advice.length > 0 ? (
            advice.map((tip, idx) => {
              const cleanedTip = tip.replace(/^Tip\s*\d+\s*-\s*/i, "").trim();
              return (
                <div key={idx}>
                  <label className="block text-base font-semibold text-gray-700 mb-1">
                    Tip {idx + 1}
                  </label>
                  <textarea
                    className="w-full p-4 border border-green-200 rounded-md resize-none bg-white text-gray-700 shadow-sm"
                    value={cleanedTip}
                    readOnly
                    rows="3"
                  />
                </div>
              );
            })
          ) : (
            <p className="text-red-500 font-semibold">No advice available.</p>
          )}
        </div>
      </div>

      {/* Additional Feature Blocks */}
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-6 text-sm text-gray-700">
        <div className="bg-white p-4 rounded-lg shadow">
          🔔 <strong>Notifications Center</strong>: Real-time alerts for
          overspending.
        </div>
        <div className="bg-white p-4 rounded-lg shadow">
          📅 <strong>Budget Calendar</strong>: Your income/expense cycle
          visualized.
        </div>
        <div className="bg-white p-4 rounded-lg shadow">
          🧾 <strong>Receipt OCR</strong>: AI extracts data from your uploaded
          receipts.
        </div>
        <div className="bg-white p-4 rounded-lg shadow">
          🎯 <strong>Goals Tracker</strong>: You’re 45% toward your “Buy Laptop”
          goal.
        </div>
        <div className="bg-white p-4 rounded-lg shadow">
          💬 <strong>AI Advice Generator</strong>: Get more Phi-2 powered
          insights.
        </div>
        <div className="bg-white p-4 rounded-lg shadow">
          📤 <strong>Export to PDF/CSV</strong>: Export this report or your full
          history.
        </div>
        <div className="bg-white p-4 rounded-lg shadow">
          🧠 <strong>Smart Categorization</strong>: Your spending is 40% food,
          30% bills.
        </div>
        <div className="bg-white p-4 rounded-lg shadow">
          📊 <strong>Monthly Report</strong>: Auto-generated summaries every 30
          days.
        </div>
        <div className="bg-white p-4 rounded-lg shadow">
          🏦 <strong>Multi-Account Support</strong>: Linked to 2 bank accounts.
        </div>
        <div className="bg-white p-4 rounded-lg shadow">
          🧮 <strong>Loan/Investment Simulator</strong>: Forecast future
          outcomes.
        </div>
      </div>

      {/* Footer */}
      <div className="text-sm text-gray-400 italic text-center pt-4 border-t">
        Last updated by{" "}
        <span className="text-green-600 font-semibold">
          FinWise Assistant 🌿
        </span>
      </div>
    </div>
  );
};

export default FinancialReport;
