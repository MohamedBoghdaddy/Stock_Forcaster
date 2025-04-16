import React, {
  createContext,
  useReducer,
  useEffect,
  useCallback,
  useMemo,
} from "react";
import axios from "axios";
import Cookies from "js-cookie";
import {
  AuthContextType,
  AuthProviderProps,
  AuthState,
  AuthAction,
} from "./authTypes";

// ------------------
// Initial State
// ------------------

const initialState: AuthState = {
  user: null,
  isAuthenticated: false,
  loading: true,
};

// ------------------
// Reducer
// ------------------

const authReducer = (state: AuthState, action: AuthAction): AuthState => {
  switch (action.type) {
    case "LOGIN_SUCCESS":
    case "USER_LOADED":
      return {
        ...state,
        user: action.payload,
        isAuthenticated: true,
        loading: false,
      };
    case "LOGOUT_SUCCESS":
    case "AUTH_ERROR":
      return {
        ...state,
        user: null,
        isAuthenticated: false,
        loading: false,
      };
    default:
      return state;
  }
};

// ------------------
// Context
// ------------------

const AuthContext = createContext<AuthContextType | undefined>(undefined);

// ------------------
// Provider
// ------------------

export const AuthProvider: React.FC<AuthProviderProps> = ({ children }) => {
  const [state, dispatch] = useReducer(authReducer, initialState);

  const checkAuth = useCallback(async () => {
    try {
      const token = Cookies.get("token") || localStorage.getItem("token");
      if (!token) {
        dispatch({ type: "AUTH_ERROR" });
        return;
      }

      axios.defaults.headers.common["Authorization"] = `Bearer ${token}`;

      const response = await axios.get(
        `${
          process.env.REACT_APP_API_URL || "http://localhost:4000"
        }/api/users/checkAuth`,
        { withCredentials: true }
      );

      if (response.data?.user) {
        dispatch({ type: "USER_LOADED", payload: response.data.user });
      } else {
        throw new Error("User data not found.");
      }
    } catch (error) {
      const err = error as {
        response?: { data?: { message?: string } };
        message?: string;
      };
      console.error(
        "❌ Authentication check failed:",
        err.response?.data?.message || err.message
      );
      dispatch({ type: "AUTH_ERROR" });
      Cookies.remove("token");
      localStorage.removeItem("token");
    }
  }, []);

  useEffect(() => {
    const storedUser = localStorage.getItem("user");
    if (storedUser) {
      try {
        const parsed = JSON.parse(storedUser);
        dispatch({ type: "LOGIN_SUCCESS", payload: parsed.user });

        if (parsed.token) {
          axios.defaults.headers.common[
            "Authorization"
          ] = `Bearer ${parsed.token}`;
        }
      } catch {
        dispatch({ type: "AUTH_ERROR" });
      }
    } else {
      checkAuth();
    }
  }, [checkAuth]);

  const logout = () => {
    Cookies.remove("token");
    localStorage.removeItem("user");
    delete axios.defaults.headers.common["Authorization"];
    dispatch({ type: "LOGOUT_SUCCESS" });
  };

  const contextValue = useMemo(() => ({ state, dispatch, logout }), [state]);

  return (
    <AuthContext.Provider value={contextValue}>{children}</AuthContext.Provider>
  );
};

export default AuthContext;
