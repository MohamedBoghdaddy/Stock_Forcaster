import { useCallback } from "react";
import axios from "axios";
import { useAuthContext } from "../context/AuthContext";

export const useLogout = () => {
  const { dispatch } = useAuthContext();

  const logout = useCallback(async () => {
    try {
      // Send the logout request to the server
      await axios.post(
        "http://localhost:4000/api/users/logout",
        {},
        { withCredentials: true } // Ensure cookies are sent if needed
      );

      // Clear localStorage (client-side only)
      localStorage.removeItem("user");
      localStorage.removeItem("token");

      // Dispatch the logout action to update the auth state
      dispatch({ type: "LOGOUT_SUCCESS" });
    } catch (error) {
      console.error("Logout error:", error);
    }
  }, [dispatch]);

  return { logout };
};
