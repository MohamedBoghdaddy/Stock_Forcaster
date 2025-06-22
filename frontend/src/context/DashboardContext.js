// 📁 context/DashboardContext.js
import React, {
  createContext,
  useReducer,
  useEffect,
  useCallback,
  useMemo,
  useState,
} from "react";
import PropTypes from "prop-types";
import { toast } from "react-toastify";
import axios from "axios";

const API_URL = process.env.REACT_APP_API_URL || "http://localhost:4000";

export const DashboardContext = createContext();

const createInitialState = () => ({
  profile: null,
  survey: null,
  analytics: { riskTolerance: [], lifestyle: [] },
  marketData: {
    stockHistory: null,
    goldHistory: null,
    realEstateHistory: null,
    stockPredictions: null,
    goldPredictions: null,
    realEstatePredictions: null,
  },
  loading: {
    profile: false,
    marketData: false,
    predictions: false,
    aiAdvice: false,
    goalPlan: false,
  },
  error: null,
  aiAdvice: null,
  goalPlan: null,
  lastUpdated: null,
});

const initialState = createInitialState();

const dashboardReducer = (state, action) => {
  switch (action.type) {
    case "FETCH_START":
      return {
        ...state,
        loading: { ...state.loading, [action.payload]: true },
        error: null,
      };
    case "FETCH_PROFILE_SUCCESS":
      return {
        ...state,
        profile: action.payload,
        lastUpdated: new Date().toISOString(),
        loading: { ...state.loading, profile: false },
      };
    case "FETCH_MARKET_DATA_SUCCESS":
      return {
        ...state,
        marketData: { ...state.marketData, ...action.payload },
        loading: { ...state.loading, marketData: false },
      };
    case "FETCH_MARKET_PREDICTIONS_SUCCESS":
      return {
        ...state,
        marketData: { ...state.marketData, ...action.payload },
        loading: { ...state.loading, predictions: false },
      };
    case "FETCH_AI_ADVICE_SUCCESS":
      return {
        ...state,
        aiAdvice: action.payload,
        loading: { ...state.loading, aiAdvice: false },
      };
    case "FETCH_GOAL_PLAN_SUCCESS":
      return {
        ...state,
        goalPlan: action.payload,
        loading: { ...state.loading, goalPlan: false },
      };
    case "UPDATE_PROFILE":
      return {
        ...state,
        profile: action.payload,
        lastUpdated: new Date().toISOString(),
        loading: { ...state.loading, profile: false },
      };
    case "FETCH_ERROR":
      return {
        ...state,
        loading: Object.keys(state.loading).reduce(
          (acc, key) => ({ ...acc, [key]: false }),
          {}
        ),
        error: action.payload,
      };
    case "RESET_STATE":
      return createInitialState();
    default:
      return state;
  }
};

const useAuthToken = () => {
  const [token, setToken] = useState(() => {
    const validateToken = (t) => t && t.split(".").length === 3;

    const localToken = localStorage.getItem("token");
    if (localToken && validateToken(localToken)) return localToken;

    const userString = localStorage.getItem("user");
    if (userString) {
      try {
        const user = JSON.parse(userString);
        if (user?.token && validateToken(user.token)) return user.token;
      } catch {
        return null;
      }
    }
    return null;
  });

  useEffect(() => {
    const handler = () => {
      const newToken =
        localStorage.getItem("token") ||
        JSON.parse(localStorage.getItem("user") || "{}")?.token;
      setToken(newToken);
    };
    window.addEventListener("storage", handler);
    return () => window.removeEventListener("storage", handler);
  }, []);

  return token;
};

export const DashboardProvider = ({ children }) => {
  const [state, dispatch] = useReducer(dashboardReducer, initialState);
  const token = useAuthToken();

  const handleError = useCallback((error, defaultMessage) => {
    const message = error.response?.data?.message || defaultMessage;
    console.error("API Error:", {
      message: error.message,
      code: error.code,
      config: error.config,
      response: error.response?.data,
    });
    toast.error(`❌ ${message}`);
    return message;
  }, []);

 const fetchProfile = useCallback(() => {
  if (!token) {
    dispatch({ type: "FETCH_ERROR", payload: "Authentication required" });
    return Promise.resolve();
  }

  dispatch({ type: "FETCH_START", payload: "profile" });
  const controller = new AbortController();

  const fetchData = async () => {
    try {
      const res = await axios.get(`${API_URL}/api/profile/me`, {
        headers: {
          Authorization: `Bearer ${token}`,
        },
        timeout: 10000,
        signal: controller.signal,
      });

      // ✅ Extract the actual profile object from the response
      const profile = res.data?.data;

      if (!profile) {
        throw new Error("No profile data returned from backend");
      }

      // ✅ Log the clean profile data (not the entire wrapper)
      console.log("✅ Clean profile fetched:", profile);

      // ✅ Dispatch the correct profile to state
      dispatch({ type: "FETCH_PROFILE_SUCCESS", payload: profile });
    } catch (err) {
      if (axios.isCancel(err)) return;

      let errorMessage = "Failed to load profile";
      if (err.code === "ERR_NETWORK") {
        errorMessage = "Network error. Please check your connection.";
      } else if (err.response?.status === 401) {
        localStorage.removeItem("token");
        localStorage.removeItem("user");
        errorMessage = "Session expired. Please login again.";
      }

      console.error("❌ Profile fetch error:", err);
      dispatch({ type: "FETCH_ERROR", payload: errorMessage });
    }
  };

  fetchData();
  return () => controller.abort();
}, [token]);

  const submitProfile = useCallback(
    async (profileData) => {
      if (!token) {
        toast.error("❌ Authentication required");
        window.location.href = "/login";
        return;
      }

      dispatch({ type: "FETCH_START", payload: "profile" });

      try {
        const res = await axios.post(`${API_URL}/api/profile`, profileData, {
          headers: {
            Authorization: `Bearer ${token}`,
            "Content-Type": "application/json",
          },
          timeout: 15000,
        });

        dispatch({ type: "UPDATE_PROFILE", payload: res.data.data });
        toast.success("✅ Profile saved successfully!");
        return res.data;
      } catch (err) {
        let errorMessage = "Profile submission failed";
        if (err.code === "ERR_NETWORK") {
          errorMessage = "Network error. Check your connection";
        } else if (err.response?.status === 401) {
          localStorage.removeItem("token");
          localStorage.removeItem("user");
          errorMessage = "Session expired. Please login again.";
          window.location.reload();
        } else if (err.response?.status === 400) {
          errorMessage = err.response.data.message || "Invalid data format";
        }

        toast.error(`❌ ${errorMessage}`);
        dispatch({ type: "FETCH_ERROR", payload: errorMessage });
        throw err;
      }
    },
    [token]
  );

  const contextValue = useMemo(
    () => ({
      ...state,
      actions: {
        fetchProfile,
        submitProfile,
      },
    }),
    [state, fetchProfile, submitProfile]
  );

  useEffect(() => {
    fetchProfile();
  }, [fetchProfile]);

  return (
    <DashboardContext.Provider value={contextValue}>
      {children}
    </DashboardContext.Provider>
  );
};

DashboardProvider.propTypes = {
  children: PropTypes.node.isRequired,
};

export default DashboardProvider;