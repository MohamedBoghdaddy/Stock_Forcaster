import { useState, useEffect, useCallback } from "react";
import axios, { AxiosError } from "axios";

export interface StockRecord {
  Date: string;
  Close: number;
}

export interface PredictionRecord {
  Date: string;
  "Predicted Close": number;
  "Predicted SMA_50": number;
  "Predicted EMA_20": number;
}

export function useStockStatistics(period: string = "1y") {
  const [historicalData, setHistoricalData] = useState<StockRecord[]>([]);
  const [futurePredictions, setFuturePredictions] = useState<
    PredictionRecord[]
  >([]);
  const [loading, setLoading] = useState<boolean>(true);
  const [error, setError] = useState<string | null>(null);

  // ✅ useCallback to satisfy exhaustive-deps rule
  const fetchAll = useCallback(async () => {
    setLoading(true);
    try {
      const [histRes, predRes] = await Promise.all([
        axios.get<StockRecord[]>(
          `http://localhost:8000/historical?period=${period}`
        ),
        axios.get<PredictionRecord[]>("http://localhost:8000/predict"),
      ]);
      setHistoricalData(histRes.data);
      setFuturePredictions(predRes.data);
      setError(null);
    } catch (err) {
      const axiosError = err as AxiosError;
      setError(axiosError.message || "Failed to fetch stock statistics");
    } finally {
      setLoading(false);
    }
  }, [period]);

  useEffect(() => {
    fetchAll();
  }, [fetchAll]);

  return {
    loading,
    error,
    historicalData,
    futurePredictions,
  };
}
