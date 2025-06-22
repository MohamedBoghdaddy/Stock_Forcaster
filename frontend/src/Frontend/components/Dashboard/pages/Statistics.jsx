import { useState, useEffect, useRef } from "react";
import axios from "axios";
import Chart from "react-apexcharts";
// import { ArrowUpIcon, ArrowDownIcon } from "../../icons";
// import Badge from "../ui/badge/Badge";

import amazon from "../../../../assets/icons8-amazon-100.svg";
import apple from "../../../../assets/icons8-apple-inc-100.svg";
import google from "../../../../assets/icons8-google-96.svg";
import meta from "../../../../assets/icons8-meta-96.svg";
import tesla from "../../../../assets/icons8-tesla-96.svg";
import netflix from "../../../../assets/icons8-netflix-80.svg";

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
  "DIS",
  "BAC",
  "XOM",
  "HD",
  "INTC",
];

const timeRanges = ["1d", "7d", "1mo", "3mo", "6mo", "1y", "3y", "5y"];
const technicalIndicators = [
  "MACD",
  "RSI",
  "Bollinger",
  "Stochastic",
  "Volume",
];
const allButtons = [...timeRanges, "predict"];

const stockIcons = {
  AAPL: apple,
  AMZN: amazon,
  GOOGL: google,
  META: meta,
  TSLA: tesla,
  NFLX: netflix,
  MSFT: "",
  NVDA: "",
  "BRK-B": "",
  JPM: "",
  V: "",
  JNJ: "",
  WMT: "",
  UNH: "",
  PG: "",
  DIS: "",
  BAC: "",
  XOM: "",
  HD: "",
  INTC: "",
};

const API_BASE_URL = "http://localhost:8000";

export default function Statistics() {
  const [selectedSymbol, setSelectedSymbol] = useState("AAPL");
  const [selectedIndicator, setSelectedIndicator] = useState("MACD");
  const [range, setRange] = useState("1y");
  const [categories, setCategories] = useState([]);
  const [data, setData] = useState([]);
  const [metrics, setMetrics] = useState({});
  const [loadingMetrics, setLoadingMetrics] = useState(true);
  const [loadingChart, setLoadingChart] = useState(false);
  const [modelMetrics, setModelMetrics] = useState(null);
  const [predictions, setPredictions] = useState(null);
  const scrollRef = useRef(null);

  useEffect(() => {
    const fetchMetrics = async () => {
      try {
        const modelRes = await axios.get(`${API_BASE_URL}/stock/metrics`);
        setModelMetrics(modelRes.data);

        const results = {};
        for (const symbol of stockSymbols.slice(0, 7)) {
          try {
            const res = await axios.get(`${API_BASE_URL}/stock/historical`, {
              params: { symbol, period: "1d" },
            });
            const histData = res.data.data || res.data;
            const closePrices = histData
              .map((d) => d.Close ?? d.close)
              .filter((p) => typeof p === "number");
            if (closePrices.length >= 2) {
              const latest = closePrices[closePrices.length - 1];
              const previous = closePrices[closePrices.length - 2];
              const change = ((latest - previous) / previous) * 100;
              results[symbol] = {
                latest: Number(latest.toFixed(2)),
                change: Number(change.toFixed(2)),
              };
            }
          } catch (err) {
            console.error(`Failed to fetch ${symbol}`, err);
          }
        }
        setMetrics(results);
      } catch (error) {
        console.error("Error fetching metrics:", error);
      } finally {
        setLoadingMetrics(false);
      }
    };

    fetchMetrics();
    const interval = setInterval(fetchMetrics, 60000);
    return () => clearInterval(interval);
  }, []);

  useEffect(() => {
    const fetchChartData = async () => {
      setLoadingChart(true);
      try {
        if (range === "predict") {
          const predRes = await axios.get(`${API_BASE_URL}/stock/predict`, {
            params: { symbol: selectedSymbol },
          });
          const predData = predRes.data;

          const histRes = await axios.get(`${API_BASE_URL}/stock/historical`, {
            params: { symbol: selectedSymbol, period: "1mo" },
          });
          const histData = histRes.data.data || histRes.data;

          const historicalDates = histData
            .map((item) => item.date ?? item.timestamp ?? item.Date ?? "")
            .filter(Boolean);
          const historicalPrices = histData
            .map((item) => item.close ?? item.Close)
            .filter((p) => typeof p === "number");

          setCategories([...historicalDates, ...predData.dates]);
          setData([...historicalPrices, ...predData.predicted]);
          setPredictions(predData);
        } else {
          const response = await axios.get(`${API_BASE_URL}/stock/historical`, {
            params: { symbol: selectedSymbol, period: range },
          });
          const histData = response.data.data || response.data;

          const grouped = {};
          for (const entry of histData) {
            const rawDate = entry.date ?? entry.timestamp ?? entry.Date;
            const close = entry.close ?? entry.Close;
            if (!rawDate || close === undefined) continue;
            const date = new Date(rawDate);
            if (isNaN(date.getTime())) continue;
            const label =
              range === "1d"
                ? date.toLocaleTimeString("en-US", {
                    hour: "2-digit",
                    minute: "2-digit",
                  })
                : date.toLocaleDateString("en-US");
            grouped[label] = close;
          }
          setCategories(Object.keys(grouped));
          setData(Object.values(grouped));
          setPredictions(null);
        }
      } catch (error) {
        console.error("Error fetching chart data:", error);
      } finally {
        setLoadingChart(false);
      }
    };

    fetchChartData();
  }, [selectedSymbol, range]);

  const options = {
    colors: ["#465FFF"],
    chart: {
      type: "area",
      height: 350,
      toolbar: { show: false },
      zoom: { enabled: false },
    },
    stroke: { curve: "smooth", width: 3 },
    fill: {
      type: "gradient",
      gradient: {
        shadeIntensity: 1,
        opacityFrom: 0.7,
        opacityTo: 0.1,
        stops: [0, 100],
      },
    },
    xaxis: {
      categories,
      labels: {
        style: { colors: "#6B7280", fontSize: "12px" },
        rotate: -45,
      },
    },
    yaxis: {
      labels: { style: { colors: "#6B7280", fontSize: "12px" } },
      min: data.length > 0 ? Math.min(...data) - 5 : 0,
      max: data.length > 0 ? Math.max(...data) + 5 : 100,
    },
    tooltip: {
      x: { format: range === "1d" ? "H:m" : "dd MMM yyyy" },
      y: { formatter: (val) => `$${val.toFixed(2)}` },
    },
    dataLabels: { enabled: false },
  };

  const series = [
    { name: range === "predict" ? "Price/Prediction" : "Close Price", data },
  ];

  const scrollLeft = () =>
    scrollRef.current?.scrollBy({ left: -200, behavior: "smooth" });
  const scrollRight = () =>
    scrollRef.current?.scrollBy({ left: 200, behavior: "smooth" });

  return (
    <div className="p-4">
      <h2 className="text-2xl font-bold mb-4">
        📈 {selectedSymbol} Stock Statistics
      </h2>

      <div className="flex gap-2 mb-4">
        {stockSymbols.slice(0, 7).map((symbol) => (
          <button
            key={symbol}
            className={`px-3 py-1 rounded ${
              symbol === selectedSymbol
                ? "bg-blue-600 text-white"
                : "bg-gray-200"
            }`}
            onClick={() => setSelectedSymbol(symbol)}
          >
            {symbol}
          </button>
        ))}
      </div>

      <div className="flex gap-2 mb-4">
        {allButtons.map((r) => (
          <button
            key={r}
            className={`px-3 py-1 rounded-full ${
              r === range ? "bg-blue-500 text-white" : "bg-gray-100"
            }`}
            onClick={() => setRange(r)}
          >
            {r.toUpperCase()}
          </button>
        ))}
      </div>

      <div className="bg-white p-4 rounded-xl shadow-md">
        {loadingChart ? (
          <div className="text-center py-10">Loading Chart...</div>
        ) : (
          <Chart options={options} series={series} type="area" height={350} />
        )}
      </div>
    </div>
  );
}
