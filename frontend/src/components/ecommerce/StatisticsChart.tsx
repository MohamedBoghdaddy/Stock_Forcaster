import Chart from "react-apexcharts";
import { ApexOptions } from "apexcharts";
import { useState, useEffect, useRef } from "react";
import axios from "axios";

// Types
type StockEntry = {
  [key: string]: string | number | undefined;
  date?: string;
  timestamp?: string;
  Date?: string;
  close?: number;
  Close?: number;
};

type PredictionResponse = {
  symbol: string;
  dates: string[];
  predicted: number[];
};

// Stock tickers
const stockSymbols = [
  "AAPL",
  "META",
  "AMZN",
  "NFLX",
  "GOOGL",
  "MSFT",
  "TSLA",
  "NVDA",
  "BRK-B",
  "JPM",
  "V",
  "JNJ",
  "WMT",
  "UNH",
  "PG",
];

const timeRanges = ["5y", "3y", "1y", "6mo", "3mo", "1mo", "7d", "1d"];
const allButtons = [...timeRanges, "predict"];

export default function StatisticsChart() {
  const [range, setRange] = useState<string>("1y");
  const [selectedSymbol, setSelectedSymbol] = useState<string>("AAPL");
  const [categories, setCategories] = useState<string[]>([]);
  const [data, setData] = useState<number[]>([]);
  const scrollRef = useRef<HTMLDivElement>(null);

  const fetchData = async (period: string, symbol: string): Promise<void> => {
    try {
      if (period === "predict") {
        const predRes = await axios.get("http://localhost:8000/predict", {
          params: { symbol },
        });
        const predData: PredictionResponse = predRes.data;
        if (
          !Array.isArray(predData.dates) ||
          !Array.isArray(predData.predicted)
        )
          return;

        setCategories(predData.dates);
        setData(predData.predicted);
      } else {
        const histRes = await axios.get("http://localhost:8000/historical", {
          params: { symbol, period },
        });

        const histData: StockEntry[] = histRes.data.data;

        const grouped: { [key: string]: number } = {};
        for (const entry of histData) {
          const rawDate = entry.date ?? entry.timestamp ?? entry.Date;
          const close = entry.close ?? entry.Close;

          if (!rawDate || close === undefined) continue;

          const date = new Date(rawDate);
          if (isNaN(date.getTime())) continue;

          const label =
            period === "1d"
              ? date.toLocaleTimeString("en-US", {
                  hour: "2-digit",
                  minute: "2-digit",
                })
              : date.toLocaleDateString("en-US");

          grouped[label] = close;
        }

        setCategories(Object.keys(grouped));
        setData(Object.values(grouped));
      }
    } catch (err) {
      console.error("❌ Data fetch error:", err);
    }
  };

  useEffect(() => {
    fetchData(range, selectedSymbol);
  }, [range, selectedSymbol]);

  const options: ApexOptions = {
    colors: ["#465FFF"],
    chart: {
      fontFamily: "Outfit, sans-serif",
      type: "area",
      height: 310,
      toolbar: { show: false },
    },
    stroke: { curve: "smooth", width: 2 },
    fill: { type: "gradient", gradient: { opacityFrom: 0.55, opacityTo: 0 } },
    markers: { size: 0, hover: { size: 5 } },
    grid: {
      yaxis: { lines: { show: true } },
      xaxis: { lines: { show: false } },
    },
    dataLabels: { enabled: false },
    tooltip: {
      x: { format: "d M yy H:m" },
      y: { formatter: (val: number) => val.toFixed(2) },
    },
    xaxis: {
      type: "category",
      categories,
      labels: {
        style: { fontSize: "11px", colors: ["#6B7280"] },
        rotate: -45,
      },
      axisBorder: { show: false },
      axisTicks: { show: false },
    },
    yaxis: {
      labels: { style: { fontSize: "12px", colors: ["#6B7280"] } },
      title: { text: "" },
    },
  };

  const series = [{ name: "Close Price", data }];

  const scrollLeft = () => {
    scrollRef.current?.scrollBy({ left: -200, behavior: "smooth" });
  };

  const scrollRight = () => {
    scrollRef.current?.scrollBy({ left: 200, behavior: "smooth" });
  };

  return (
    <div className="rounded-2xl border border-gray-200 bg-white px-5 pb-5 pt-5 dark:border-gray-800 dark:bg-white/[0.03] sm:px-6 sm:pt-6">
      <div className="flex flex-col gap-5 mb-6 sm:flex-row sm:justify-between">
        <div className="w-full">
          <h3 className="text-lg font-semibold text-gray-800 dark:text-white/90">
            Stock Statistics ({selectedSymbol})
          </h3>
          <p className="mt-1 text-gray-500 text-theme-sm dark:text-gray-400">
            View historical data or forecasted 15-day predictions
          </p>
        </div>

        <div
          className="flex items-center gap-2 overflow-x-auto custom-scrollbar"
          ref={scrollRef}
        >
          <button onClick={scrollLeft} className="text-lg px-2">
            ←
          </button>
          {stockSymbols.map((sym) => (
            <button
              key={sym}
              onClick={() => setSelectedSymbol(sym)}
              className={`text-sm px-3 py-1 rounded-full border ${
                sym === selectedSymbol
                  ? "bg-green-600 text-white"
                  : "text-gray-600 border-gray-300 hover:bg-gray-100"
              }`}
            >
              {sym}
            </button>
          ))}
          <button onClick={scrollRight} className="text-lg px-2">
            →
          </button>
        </div>

        <div className="flex items-start w-full gap-2 flex-wrap sm:justify-end">
          {allButtons.map((r) => (
            <button
              key={r}
              className={`text-sm px-3 py-1 rounded-full border ${
                r === range
                  ? "bg-blue-600 text-white"
                  : "text-gray-600 border-gray-300 hover:bg-gray-100"
              }`}
              onClick={() => setRange(r)}
            >
              {r === "predict" ? "PREDICTIONS" : r.toUpperCase()}
            </button>
          ))}
        </div>
      </div>

      <div className="max-w-full overflow-x-auto custom-scrollbar">
        <div className="min-w-[1000px] xl:min-w-full">
          <Chart options={options} series={series} type="area" height={310} />
        </div>
      </div>
    </div>
  );
}
