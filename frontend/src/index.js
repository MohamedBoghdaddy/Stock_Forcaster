import React from "react";
import { createRoot } from "react-dom/client";
import App from "./App";
import reportWebVitals from "./reportWebVitals";
import { AuthProvider } from "./context/AuthContext";
import { ChatProvider } from "./context/ChatContext";
import { DashboardProvider } from "./context/DashboardContext";
import { ThemeProvider } from "./context/ThemeContext"; // ✅ Theme context

import "bootstrap/dist/css/bootstrap.min.css";

// Create root container
const container = document.getElementById("root");
const root = createRoot(container);

// Render the application
root.render(
  <React.StrictMode>
    <ThemeProvider>
      <AuthProvider>
        <ChatProvider>
          <DashboardProvider>
            <App />
          </DashboardProvider>
        </ChatProvider>
      </AuthProvider>
    </ThemeProvider>
  </React.StrictMode>
);

reportWebVitals();
