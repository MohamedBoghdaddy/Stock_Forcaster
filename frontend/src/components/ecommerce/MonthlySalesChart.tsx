import Chart from "react-apexcharts";
import { ApexOptions } from "apexcharts";
import { useState, useEffect } from "react";
import axios from "axios";

type StockEntry = {
  date?: string;
  timestamp?: string;
  close?: number;
};

const timeRanges = ["5y", "3y", "1y", "6m", "1w", "1d"];

export default function MonthlySalesChart() {
  const [salesData, setSalesData] = useState<number[]>([]);
  const [categories, setCategories] = useState<string[]>([]);
  const [range, setRange] = useState<string>("1y");

  const fetchData = async (period: string) => {
    try {
      // 1. Get historical
      const histRes = await axios.get("http://localhost:8000/historical", {
        params: { symbol: "AAPL", period },
      });

      const histData: StockEntry[] = histRes.data.data;

      const grouped = histData.reduce((acc: Record<string, number>, entry) => {
        const label = new Date(
          entry.date || entry.timestamp || ""
        ).toLocaleDateString("en-US");
        acc[label] = (acc[label] || 0) + (entry.close || 0);
        return acc;
      }, {});

      const histLabels = Object.keys(grouped);
      const histValues = Object.values(grouped);

      // 2. Get predictions
      const predRes = await axios.get("http://localhost:8000/predict", {
        params: { symbol: "AAPL" },
      });

      const pred = predRes.data;
      const predLabels: string[] = pred.dates || [];
      const predValues: number[] = pred.predicted || [];

      // ✅ Log only after data is ready
      console.log("HISTORICAL:", histLabels.length, histValues);
      console.log("PREDICTION:", predLabels.length, predValues);
      console.log("TOTAL CATEGORIES:", [...histLabels, ...predLabels]);
      console.log("TOTAL DATA:", [...histValues, ...predValues]);

      // 3. Merge data
      setCategories([...histLabels, ...predLabels]);
      setSalesData([...histValues, ...predValues]);
    } catch (err) {
      console.error("Data fetch error:", err);
    }
  };


  useEffect(() => {
    fetchData(range);
  }, [range]);

  const options: ApexOptions = {
    colors: ["#465fff"],
    chart: {
      fontFamily: "Outfit, sans-serif",
      type: "bar",
      height: 180,
      toolbar: { show: false },
    },
    plotOptions: {
      bar: {
        horizontal: false,
        columnWidth: "39%",
        borderRadius: 5,
        borderRadiusApplication: "end",
      },
    },
    dataLabels: { enabled: false },
    stroke: { show: true, width: 4, colors: ["transparent"] },
    xaxis: {
      categories: categories,
      labels: { rotate: -45 },
    },
    legend: {
      show: true,
      position: "top",
      horizontalAlign: "left",
      fontFamily: "Outfit",
    },
    yaxis: { title: { text: undefined } },
    grid: { yaxis: { lines: { show: true } } },
    fill: { opacity: 1 },
    tooltip: {
      y: {
        formatter: (val: number) => `${val.toFixed(2)}`,
      },
    },
  };

  const series = [
    {
      name: "Close Price",
      data: salesData,
    },
  ];

  return (
    <div className="overflow-hidden rounded-2xl border border-gray-200 bg-white px-5 pt-5 dark:border-gray-800 dark:bg-white/[0.03] sm:px-6 sm:pt-6">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold text-gray-800 dark:text-white/90">
          Stock Chart + 15-Day Prediction
        </h3>
      </div>

      <div className="flex gap-2 flex-wrap mb-4">
        {timeRanges.map((r) => (
          <button
            key={r}
            className={`text-sm px-3 py-1 rounded-full border ${
              r === range
                ? "bg-blue-600 text-white"
                : "text-gray-600 border-gray-300 hover:bg-gray-100"
            }`}
            onClick={() => setRange(r)}
          >
            {r.toUpperCase()}
          </button>
        ))}
      </div>

      <div className="max-w-full overflow-x-auto custom-scrollbar">
        <div className="-ml-5 min-w-[650px] xl:min-w-full pl-2">
          <Chart options={options} series={series} type="bar" height={180} />
        </div>
      </div>
    </div>
  );
}
