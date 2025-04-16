import { useState } from "react";
import axios from "axios";
import { useAuthContext } from "../context/AuthContext";

const API_URL =
  process.env.REACT_APP_API_URL ||
  (window.location.hostname === "localhost"
    ? "http://localhost:4000"
    : "https://hedj.onrender.com");

export const useSignup = () => {
  const [username, setUsername] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [confirmPassword, setConfirmPassword] = useState("");
  const [firstName, setFirstName] = useState("");
  const [lastName, setLastName] = useState("");
  const [gender, setGender] = useState("");
  const [errorMessage, setErrorMessage] = useState("");
  const [successMessage, setSuccessMessage] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const { dispatch } = useAuthContext();

  const handleSignup = async (e) => {
    e.preventDefault();
    setIsLoading(true);
    setErrorMessage("");
    setSuccessMessage("");

    // ✅ Password validation
    if (password !== confirmPassword) {
      setErrorMessage("Passwords do not match.");
      setIsLoading(false);
      return;
    }

    try {
      const response = await axios.post(
        `${API_URL}/api/users/signup`,
        {
          username,
          email,
          password,
          firstName,
          lastName,
          gender,
        },
        { withCredentials: true }
      );

      const { user, token } = response.data;

      // ✅ Store user data in localStorage
      localStorage.setItem("user", JSON.stringify({ user, token }));

      // ✅ Dispatch signup success
      dispatch({ type: "LOGIN_SUCCESS", payload: user });

      setSuccessMessage("Registration successful.");
    } catch (error) {
      console.error("Signup error:", error);
      setErrorMessage(
        error.response?.data?.message || "Signup failed. Please try again."
      );
    } finally {
      setIsLoading(false);
    }
  };

  return {
    username,
    setUsername,
    email,
    setEmail,
    password,
    setPassword,
    confirmPassword,
    setConfirmPassword,
    firstName,
    setFirstName,
    lastName,
    setLastName,
    gender,
    setGender,
    errorMessage,
    successMessage,
    isLoading,
    handleSignup,
  };
};
