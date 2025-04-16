import PageBreadcrumb from "../../components/common/PageBreadCrumb";
import ComponentCard from "../../components/common/ComponentCard";
// import LineChartOne from "../../components/charts/line/LineChartOne";
import PageMeta from "../../components/common/PageMeta";
import { StockDataProvider } from "../../context/StockDataContext";
import { useStockData } from "../../context/StockDataContext";
import { Line } from "react-chartjs-2";
import { useMemo } from "react";

export default function LineChart() {
  return (
    <StockDataProvider>
      <PageMeta
        title="React.js Chart Dashboard | TailAdmin - React.js Admin Dashboard Template"
        description="This is React.js Chart Dashboard page for TailAdmin - React.js Tailwind CSS Admin Dashboard Template"
      />
      <PageBreadcrumb pageTitle="Line Chart" />
      <StatisticsDashboard />
    </StockDataProvider>
  );
}

function StatisticsDashboard() {
  const { historicalData, futurePredictions, loading, error } = useStockData();

  const chartData = useMemo(() => {
    return {
      labels: historicalData
        .map((item) => item.Date)
        .concat(futurePredictions.map((p) => p.Date)),
      datasets: [
        {
          label: "Actual Close",
          data: historicalData.map((item) => item.Close),
          borderColor: "#3b82f6",
          fill: false,
        },
        {
          label: "Predicted Close",
          data: [
            ...new Array(historicalData.length).fill(null),
            ...futurePredictions.map((item) => item["Predicted Close"]),
          ],
          borderColor: "#f97316",
          borderDash: [5, 5],
          fill: false,
        },
      ],
    };
  }, [historicalData, futurePredictions]);

  return (
    <div className="space-y-6">
      <ComponentCard title="Line Chart: Actual vs Predicted Close">
        {loading ? (
          <p>Loading...</p>
        ) : error ? (
          <p className="text-red-500">{error}</p>
        ) : (
          <Line
            data={chartData}
            options={{
              responsive: true,
              plugins: {
                legend: { position: "top" },
                title: { display: true, text: "AAPL Stock Price Forecast" },
              },
            }}
          />
        )}
      </ComponentCard>
    </div>
  );
}
