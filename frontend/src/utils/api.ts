// utils/api.ts
import axios, { AxiosInstance } from "axios";

/**
 * Creates a configured Axios instance with optional Bearer token.
 * @param token Optional JWT or auth token.
 * @returns Configured Axios client.
 */
export const createApiClient = (token?: string): AxiosInstance => {
  const apiClient = axios.create({
    baseURL: import.meta.env.VITE_API_BASE_URL || "http://localhost:8000",
    headers: {
      "Content-Type": "application/json",
    },
    timeout: 10000,
  });

  // Inject token if available
  apiClient.interceptors.request.use((config) => {
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  });

  // Optional: Log response errors globally
  apiClient.interceptors.response.use(
    (response) => response,
    (error) => {
      console.error("❌ API Error:", error.response?.data || error.message);
      return Promise.reject(error);
    }
  );

  return apiClient;
};

/**
 * Hook version of API client using token from localStorage.
 * Ideal for use in components.
 */
export const useApiClient = (): AxiosInstance => {
  const token = localStorage.getItem("authToken") || "";
  return createApiClient(token);
};

// Optional: If using auth context
/*
import { useAuthContext } from "../hooks/useAuthContext";

export const useAuthApiClient = (): AxiosInstance => {
  const { state } = useAuthContext();
  const token = state.user?.token || "";
  return createApiClient(token);
};
*/
