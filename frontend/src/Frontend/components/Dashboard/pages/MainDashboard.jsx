import { useContext } from "react";
import Sidebar from "../components/Sidebar";
import { DashboardContext } from "../../../../context/DashboardContext";
import { useAuthContext } from "../../../../context/AuthContext";
import "../../styles/dashboard.css";

const MainDashboard = () => {
  const { profileData, loading, error } = useContext(DashboardContext);
  const { user } = useAuthContext;

  const salary = profileData?.salary || 0;
  const expenses = profileData?.expenses || 0;
  const balance = salary - expenses;

  return (
    <div className="flex min-h-screen bg-gray-100 dark:bg-gray-950 text-gray-800 dark:text-white">
      <Sidebar />

      <div className="flex-1">

        <main className="p-4 sm:p-6 md:p-8">
          <div className="max-w-6xl mx-auto space-y-10">
            {loading ? (
              <div className="text-center text-sm text-gray-400">
                Loading data...
              </div>
            ) : error ? (
              <div className="text-center text-red-500">
                Error loading profile
              </div>
            ) : (
              <>
                {/* Header greeting */}
                <div className="text-center">
                  <h2 className="text-2xl font-bold mb-2">
                    Welcome back,{" "}
                    <span className="text-blue-600">{user?.username}</span> 👋
                  </h2>
                  <p className="text-gray-500 dark:text-gray-400 text-sm">
                    Here’s a quick overview of your financial health.
                  </p>
                </div>

                {/* Balance Cards */}
                <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4 sm:gap-6">
                  <div className="dashboard-card bg-white dark:bg-gray-800 p-6 rounded-xl text-center">
                    <h3 className="text-base font-semibold mb-2">
                      💰 Total Balance
                    </h3>
                    <p className="text-3xl font-bold text-blue-600">
                      ${balance.toLocaleString()}
                    </p>
                  </div>

                  <div className="dashboard-card bg-white dark:bg-gray-800 p-6 rounded-xl text-center">
                    <h3 className="text-base font-semibold mb-2">📈 Income</h3>
                    <p className="text-3xl font-bold text-green-600">
                      ${salary.toLocaleString()}
                    </p>
                  </div>

                  <div className="dashboard-card bg-white dark:bg-gray-800 p-6 rounded-xl text-center">
                    <h3 className="text-base font-semibold mb-2">
                      💸 Expenses
                    </h3>
                    <p className="text-3xl font-bold text-red-600">
                      ${expenses.toLocaleString()}
                    </p>
                  </div>
                </div>

                {/* Feature Grid */}
                <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4 mt-8">
                  <FeatureCard
                    emoji="🔔"
                    title="Notifications Center"
                    desc="Get alerts for overspending or goal deadlines."
                  />
                  <FeatureCard
                    emoji="📅"
                    title="Budget Calendar"
                    desc="Visualize your monthly income and bill dates."
                  />
                  <FeatureCard
                    emoji="🧾"
                    title="Receipt OCR"
                    desc="Upload and scan your receipts automatically."
                  />
                  <FeatureCard
                    emoji="🎯"
                    title="Financial Goals"
                    desc="Track your progress towards custom goals."
                  />
                  <FeatureCard
                    emoji="💬"
                    title="AI Advisor"
                    desc="Generate tailored insights from your Phi-2 assistant."
                  />
                  <FeatureCard
                    emoji="📤"
                    title="Export Reports"
                    desc="Download your dashboard summary as PDF or CSV."
                  />
                  <FeatureCard
                    emoji="🧠"
                    title="Smart Categorization"
                    desc="Automatically tag and classify your expenses."
                  />
                  <FeatureCard
                    emoji="📊"
                    title="Monthly Summary"
                    desc="View auto-generated financial reports every month."
                  />
                  <FeatureCard
                    emoji="🏦"
                    title="Multi-Accounts"
                    desc="Connect bank, wallet, and external finance apps."
                  />
                  <FeatureCard
                    emoji="🧮"
                    title="Loan Simulator"
                    desc="Predict EMI and savings impact for any loan."
                  />
                </div>
              </>
            )}
          </div>
        </main>
      </div>
    </div>
  );
};

// Sub-component for reusable feature cards
const FeatureCard = ({ emoji, title, desc }) => (
  <div className="bg-white dark:bg-gray-800 p-5 rounded-xl shadow transition hover:scale-[1.02] duration-300">
    <h4 className="text-lg font-bold text-gray-700 dark:text-white mb-2 flex items-center gap-2">
      {emoji} {title}
    </h4>
    <p className="text-sm text-gray-600 dark:text-gray-300">{desc}</p>
  </div>
);

export default MainDashboard;
