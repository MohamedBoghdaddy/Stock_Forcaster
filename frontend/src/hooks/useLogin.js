import { useState, useCallback } from "react";
import axios from "axios";
import { useAuthContext } from "../context/AuthContext";

const apiUrl = process.env.REACT_APP_API_URL;
const localUrl = "http://localhost:4000";

export const useLogin = () => {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [errorMessage, setErrorMessage] = useState("");
  const [successMessage, setSuccessMessage] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const { dispatch } = useAuthContext();

  const handleLogin = useCallback(
    async (e) => {
      e.preventDefault();
      setIsLoading(true);
      setErrorMessage("");
      setSuccessMessage("");

      try {
        const response = await axios.post(
          `${
            process.env.NODE_ENV === "production" ? apiUrl : localUrl
          }/api/users/login`,
          { email, password },
          { withCredentials: true }
        );

        const { token, user } = response.data;

        if (token && user) {
          // Store token and user in local storage
          localStorage.setItem("token", token);
          localStorage.setItem("user", JSON.stringify({ token, user }));

          // Set Authorization header
          axios.defaults.headers.common["Authorization"] = `Bearer ${token}`;

          // Dispatch login success
          dispatch({ type: "LOGIN_SUCCESS", payload: user });

          // Set success message
          setSuccessMessage("Login successful");
        } else {
          console.error("Unexpected response format:", response.data);
          throw new Error("Invalid response data");
        }
      } catch (error) {
        console.error("Login error:", error);
        setErrorMessage(error.response?.data?.message || "Login failed");
        dispatch({ type: "AUTH_ERROR" });
      } finally {
        setIsLoading(false);
      }
    },
    [email, password, dispatch]
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
