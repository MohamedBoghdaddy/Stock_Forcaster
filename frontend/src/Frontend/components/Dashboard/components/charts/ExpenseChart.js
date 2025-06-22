import React from "react";
import { Pie, Bar } from "react-chartjs-2";
import {
  Chart,
  ArcElement,
  BarElement,
  CategoryScale,
  LinearScale,
  Tooltip,
  Legend,
} from "chart.js";

// Register required Chart.js components
Chart.register(
  ArcElement,
  BarElement,
  CategoryScale,
  LinearScale,
  Tooltip,
  Legend
);

const ExpenseChart = ({ data, onAddExpense }) => {
  // Data for the Pie Chart (Expenses by Category)
  const pieData = {
    labels: data.labels,
    datasets: [
      {
        label: "Expenses",
        data: data.values,
        backgroundColor: [
          "#FF6384", // red
          "#36A2EB", // blue
          "#FFCE56", // yellow
          "#8DD3C7", // teal
          "#FDB462", // orange
          "#DDA0DD", // purple
          "#FFA07A", // light salmon
        ],
      },
    ],
  };

  // Data for the Bar Chart (Income vs. Savings vs. Expenses)
  const barData = {
    labels: ["Income", "Savings", "Total Expenses"],
    datasets: [
      {
        label: "EGP",
        data: [data.income, data.savings, data.totalExpenses],
        backgroundColor: "#42a5f5",
      },
    ],
  };

  return (
    <div className="bg-white dark:bg-gray-800 p-6 rounded-xl shadow mt-6">
      {/* Header + Add Button */}
      <div className="flex justify-between items-center mb-4">
        <h2 className="text-xl font-bold text-gray-800 dark:text-white">
          📊 Expense Breakdown
        </h2>
        <button
          onClick={onAddExpense}
          className="bg-blue-600 hover:bg-blue-700 text-white font-semibold px-4 py-2 rounded shadow"
        >
          ＋ Add Expense
        </button>
      </div>

      {/* Pie Chart */}
      <Pie data={pieData} />

      {/* Bar Chart */}
      <h3 className="text-lg font-semibold text-gray-700 dark:text-white mt-6 mb-3">
        📉 Income vs Savings vs Expenses
      </h3>
      <Bar data={barData} />
    </div>
  );
};

export default ExpenseChart;
