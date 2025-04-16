import {
  createContext,
  useContext,
  useState,
  useEffect,
  ReactNode,
} from "react";
import axios from "axios";

// Define the types for the data
export interface HistoricalRecord {
  Date: string;
  Close: number;
}

export interface FuturePrediction {
  Date: string;
  "Predicted Close": number;
  "Predicted SMA_50": number;
  "Predicted EMA_20": number;
}

interface StockDataContextType {
  loading: boolean;
  error: string | null;
  historicalData: HistoricalRecord[];
  futurePredictions: FuturePrediction[];
  selectedPeriod: string;
  setSelectedPeriod: (period: string) => void;
  fetchHistoricalData: (period?: string) => Promise<void>;
  fetchFuturePredictions: () => Promise<void>;
}

const StockDataContext = createContext<StockDataContextType | undefined>(
  undefined
);

interface ProviderProps {
  children: ReactNode;
}

export const StockDataProvider = ({ children }: ProviderProps) => {
  const [loading, setLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);
  const [futurePredictions, setFuturePredictions] = useState<
    FuturePrediction[]
  >([]);
  const [historicalData, setHistoricalData] = useState<HistoricalRecord[]>([]);
  const [selectedPeriod, setSelectedPeriod] = useState<string>("1y");

  useEffect(() => {
    fetchHistoricalData(selectedPeriod);
    fetchFuturePredictions();
  }, [selectedPeriod]);

  const fetchHistoricalData = async (period: string = "1y") => {
    setLoading(true);
    try {
      const res = await axios.get<HistoricalRecord[]>(
        `http://localhost:8000/historical?period=${period}`
      );
      setHistoricalData(res.data);
      setError(null);
    } catch {
      setError("Failed to fetch historical data.");
    } finally {
      setLoading(false);
    }
  };

  const fetchFuturePredictions = async () => {
    try {
      const res = await axios.get<FuturePrediction[]>(
        "http://localhost:8000/predict"
      );
      setFuturePredictions(res.data);
    } catch {
      setError("Failed to fetch future predictions.");
    }
  };

  return (
    <StockDataContext.Provider
      value={{
        loading,
        error,
        historicalData,
        futurePredictions,
        selectedPeriod,
        setSelectedPeriod,
        fetchHistoricalData,
        fetchFuturePredictions,
      }}
    >
      {children}
    </StockDataContext.Provider>
  );
};

export const useStockData = (): StockDataContextType => {
  const context = useContext(StockDataContext);
  if (!context) {
    throw new Error("useStockData must be used within a StockDataProvider");
  }
  return context;
};
