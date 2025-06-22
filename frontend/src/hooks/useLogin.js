import { useState, useCallback } from "react";
import axios from "axios";
import { useNavigate } from "react-router-dom";
import { useAuthContext } from "../context/AuthContext";

const API_URL =
  process.env.REACT_APP_API_URL ??
  (window.location.hostname === "localhost"
    ? "http://localhost:4000"
    : "https://financial-ai-backend-kr2s.onrender.com");

export const useLogin = () => {
  const navigate = useNavigate();
  const { dispatch } = useAuthContext();

  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [errorMessage, setErrorMessage] = useState("");
  const [successMessage, setSuccessMessage] = useState("");
  const [isLoading, setIsLoading] = useState(false);

  const handleLogin = useCallback(
    async (e) => {
      e.preventDefault();
      setIsLoading(true);
      setErrorMessage("");
      setSuccessMessage("");

      // Admin shortcut login
      if (email === "ahmedaref@gmail.com" && password === "12345678") {
        localStorage.setItem("admin_logged_in", "true");
        localStorage.setItem("token", "admin_token");
        localStorage.setItem(
          "user",
          JSON.stringify({ role: "admin", username: "Ahmed Aref", email })
        );

        dispatch({
          type: "LOGIN_SUCCESS",
          payload: { role: "admin", username: "Ahmed Aref", email },
        });

        navigate("/admin/dashboard");
        setIsLoading(false);
        return;
      }

      try {
        const response = await axios.post(
          `${API_URL}/api/users/login`,
          { email, password },
          {
            withCredentials: true,
            headers: { "Content-Type": "application/json" },
          }
        );

        const { token, user } = response.data;

        if (!token || !user) {
          throw new Error("Unexpected response format");
        }

        localStorage.setItem("token", token);
        localStorage.setItem("user", JSON.stringify(user));

        axios.defaults.headers.common["Authorization"] = `Bearer ${token}`;

        dispatch({ type: "LOGIN_SUCCESS", payload: user });

        setSuccessMessage("Login successful");
        navigate("/");
      } catch (error) {
        console.error("Login error:", error);
        setErrorMessage(
          error.response?.data?.message || "Login failed. Please try again."
        );
        dispatch({ type: "AUTH_ERROR" });
      } finally {
        setIsLoading(false);
      }
    },
    [email, password, dispatch, navigate]
  );

  return {
    email,
    setEmail,
    password,
    setPassword,
    showPassword,
    setShowPassword,
    errorMessage,
    successMessage,
    isLoading,
    handleLogin,
  };
};
