import { useEffect, useState } from "react";
import axios from "axios";
import { ArrowUpIcon, ArrowDownIcon } from "../../icons";
import Badge from "../ui/badge/Badge";

// Company logos
import amazon from "../../assets/icons8-amazon-100.svg";
import apple from "../../assets/icons8-apple-inc-100.svg";
import google from "../../assets/icons8-google-96.svg";
import meta from "../../assets/icons8-meta-96.svg";
import tesla from "../../assets/icons8-tesla-96.svg";
import netflix from "../../assets/icons8-netflix-80.svg"; // ✅ Make sure this file exists in /assets

// Companies to track
const trackedStocks: string[] = [
  "AAPL",
  "META",
  "AMZN",
  "NFLX",
  "GOOGL",
  "MSFT",
  "TSLA",
];

// Logo map
const stockIcons: Record<string, string> = {
  AAPL: apple,
  AMZN: amazon,
  GOOGL: google,
  META: meta,
  TSLA: tesla,
  NFLX: netflix,
};

interface StockMetric {
  latest: number;
  change: number;
}

export default function StockMetrics() {
  const [metrics, setMetrics] = useState<Record<string, StockMetric>>({});
  const [loading, setLoading] = useState<boolean>(true);

  useEffect(() => {
    const fetchStockData = async () => {
      const results: Record<string, StockMetric> = {};
      setLoading(true);

      for (const symbol of trackedStocks) {
        try {
          const res = await axios.get("http://localhost:8000/stock/historical", {
            params: { symbol, period: "1d" },
          });

          const data: { Close?: number; close?: number }[] = res.data.data;
          const closePrices = data
            .map((d) => d.Close ?? d.close)
            .filter((p): p is number => typeof p === "number");

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
          console.error(`❌ Failed to fetch data for ${symbol}:`, err);
        }
      }

      setMetrics(results);
      setLoading(false);
    };

    fetchStockData();

    // Set an interval to refresh the stock data every 60 seconds
    const interval = setInterval(fetchStockData, 60000);

    return () => clearInterval(interval); // Clean up the interval on component unmount
  }, []);

  if (loading) {
    return <div>Loading...</div>; // Loading state
  }

  return (
    <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5">
      {Object.entries(metrics).map(([symbol, { latest, change }]) => (
        <div
          key={symbol}
          className="rounded-2xl border border-gray-200 bg-white p-5 dark:border-gray-800 dark:bg-white/[0.03]"
        >
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              {stockIcons[symbol] ? (
                <img
                  src={stockIcons[symbol]}
                  alt={`${symbol} logo`}
                  className="w-6 h-6"
                />
              ) : (
                <div className="w-6 h-6 bg-gray-200 rounded-full" /> // Fallback for missing logos
              )}
              <h4 className="text-sm text-gray-500 dark:text-gray-400">
                {symbol}
              </h4>
            </div>
            <Badge color={change >= 0 ? "success" : "error"}>
              {change >= 0 ? <ArrowUpIcon /> : <ArrowDownIcon />}
              {Math.abs(change)}%
            </Badge>
          </div>
          <div className="mt-3 font-bold text-gray-800 text-xl dark:text-white/90">
            ${latest}
          </div>
        </div>
      ))}
    </div>
  );
}
