// ✅ AuthContext.js (ensure token loads user before app renders)
import React, {
  createContext, useReducer, useEffect,
  useCallback, useContext, useMemo
} from "react";
import axios from "axios";
import Cookies from "js-cookie";

const API_URL = process.env.REACT_APP_API_URL || "http://localhost:4000";
const AuthContext = createContext();

const initialState = {
  user: null,
  isAuthenticated: false,
  loading: true,
};

const authReducer = (state, action) => {
  switch (action.type) {
    case "LOGIN_SUCCESS":
    case "USER_LOADED":
      return { ...state, user: action.payload, isAuthenticated: true, loading: false };
    case "LOGOUT_SUCCESS":
    case "AUTH_ERROR":
      return { ...state, user: null, isAuthenticated: false, loading: false };
    default:
      return state;
  }
};

const getToken = () => {
  return (
    Cookies.get("token") ||
    localStorage.getItem("token") ||
    JSON.parse(localStorage.getItem("user") || "null")?.token
  );
};

export const AuthProvider = ({ children }) => {
  const [state, dispatch] = useReducer(authReducer, initialState);

  const setAuthHeaders = (token) => {
    axios.defaults.headers.common["Authorization"] = `Bearer ${token}`;
  };

  const clearAuthStorage = () => {
    Cookies.remove("token");
    localStorage.removeItem("user");
    localStorage.removeItem("token");
    delete axios.defaults.headers.common["Authorization"];
  };

  const checkAuth = useCallback(async () => {
    try {
      const token = getToken();
      if (!token) return dispatch({ type: "AUTH_ERROR" });
      setAuthHeaders(token);

      const res = await axios.get(`${API_URL}/api/users/checkAuth`, {
        withCredentials: true,
      });

      if (res.data?.user) {
        const userData = { user: res.data.user, token };
        localStorage.setItem("user", JSON.stringify(userData));
        Cookies.set("token", token, { expires: 7 });
        dispatch({ type: "USER_LOADED", payload: res.data.user });
      } else {
        dispatch({ type: "AUTH_ERROR" });
      }
    } catch (err) {
      clearAuthStorage();
      dispatch({ type: "AUTH_ERROR" });
    }
  }, []);

  useEffect(() => {
    const storedUser = localStorage.getItem("user");
    const token = getToken();
    if (storedUser && token) {
      try {
        const parsedUser = JSON.parse(storedUser);
        setAuthHeaders(token);
        dispatch({ type: "LOGIN_SUCCESS", payload: parsedUser.user });
      } catch {
        clearAuthStorage();
      }
    }
    checkAuth();
  }, [checkAuth]);

  return (
    <AuthContext.Provider value={{ state, dispatch, checkAuth }}>
      {children}
    </AuthContext.Provider>
  );
};

export const useAuthContext = () => useContext(AuthContext);
