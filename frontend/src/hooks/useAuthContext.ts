// src/context/useAuthContext.ts

import { useContext } from "react";
import AuthContext from "../context/AuthContext";
import { AuthContextType } from "../context/authTypes";

export const useAuthContext = (): AuthContextType => {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error("useAuthContext must be used within an AuthProvider");
  }
  return context;
};
