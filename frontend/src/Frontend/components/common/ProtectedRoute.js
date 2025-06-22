import React from "react";
import { Navigate, useLocation } from "react-router-dom";
import { useAuthContext } from "../../../context/AuthContext";

const ProtectedRoute = ({ children }) => {
  const { state } = useAuthContext();
  const { isAuthenticated, isLoading } = state;
  const location = useLocation();

  // Show loading state if auth is still being checked
  if (isLoading) {
    return (
      <div className="loading-container">
        <div className="spinner"></div>
        <p>Checking authentication...</p>
      </div>
    );
  }

  // Redirect to login if not authenticated
  if (!isAuthenticated) {
    return <Navigate to="/login" state={{ from: location }} replace />;
  }

  // Render the children if authenticated
  return children;
};

export default ProtectedRoute;
