import React from "react";
import { BrowserRouter, Route, Routes } from "react-router-dom";
import { AuthProvider } from "./context/AuthContext";
import Chatbot from "./Frontend/components/Pages/Chatbot.js";

// Tools
import ProfileCardPage from "./Frontend/components/Dashboard/pages/ProfileCardPage";
import CalendarExpenseTracker from "./Frontend/components/Dashboard/pages/CalendarExpenseTracker";
import FinanceTools from "./Frontend/components/Dashboard/tools/FinanceTools";
import CurrencyConverter from "./Frontend/components/Dashboard/tools/CurrencyConverter";

// Frontend Components
import Home from "./Frontend/components/Home/home";
import NavBar from "./Frontend/components/Home/Navbar";
import Footer from "./Frontend/components/Home/Footer";
import MiniNavbar from "./Frontend/components/Home/Mininavbar";

import Login from "./Frontend/components/LOGIN&REGISTRATION/Login/Login";
import Signup from "./Frontend/components/LOGIN&REGISTRATION/Signup/Signup";
import Dashboard from "./Frontend/components/Dashboard/pages/MainDashboard";

import Sidebar from "./Frontend/components/Dashboard/components/Sidebar.jsx";
import Settings from "./Frontend/components/Dashboard/pages/Settings.jsx";
import Profile from "./Frontend/components/Dashboard/pages/Profile.jsx";
import AnalyticsReport from "./Frontend/components/Dashboard/analytics";
import FinancialReportPage from "./Frontend/components/Dashboard/pages/FinancialReport";
import AdminDashboard from "./Frontend/components/Dashboard/admin/AdminDashboard.js";
import UserDetails from "./Frontend/components/Dashboard/admin/UserDetails";

import AIChat from "./Frontend/components/chatbot/AIChat";
import Contact from "./Frontend/components/Contact/contact";
import TabsView from "./Frontend/components/Dashboard/tabs/Tabs.jsx";

import ProtectedRoute from "./Frontend/components/Auth/ProtectedRoute";
import AdminRoute from "./Frontend/components/Auth/AdminRoute";
import Statistics from "./Frontend/components/Dashboard/pages/Statistics.jsx";

function App() {
  return (
    <AuthProvider>
      <BrowserRouter>
        <Routes>
          {/* Public Routes */}
          <Route
            path="/"
            element={
              <>
                <NavBar />
                <Home />
                <Footer />
              </>
            }
          />
          <Route
            path="/login"
            element={
              <>
                <MiniNavbar />
                <Login />
              </>
            }
          />
          <Route
            path="/signup"
            element={
              <>
                <MiniNavbar />
                <Signup />
              </>
            }
          />

          {/* Protected Routes */}
          <Route
            path="/dashboard"
            element={
              <ProtectedRoute>
                <>
                  <MiniNavbar />
                  <Dashboard />
                  <Footer />
                </>
              </ProtectedRoute>
            }
          />
          <Route
            path="/settings"
            element={
              <ProtectedRoute>
                <>
                  <MiniNavbar />
                  <Sidebar />
                  <Settings />
                  <Footer />
                </>
              </ProtectedRoute>
            }
          />
          <Route
            path="/Statistics"
            element={
              <ProtectedRoute>
                <>
                  <MiniNavbar />
                  <Sidebar />
                  <Statistics />
                  <Footer />
                </>
              </ProtectedRoute>
            }
          />
          <Route
            path="/profile"
            element={
              <ProtectedRoute>
                <>
                  <MiniNavbar />
                  <Sidebar />
                  <Profile />
                  <Footer />
                </>
              </ProtectedRoute>
            }
          />
          <Route
            path="/analytics"
            element={
              <ProtectedRoute>
                <>
                  <MiniNavbar />
                  <Sidebar />
                  <AnalyticsReport />
                  <Footer />
                </>
              </ProtectedRoute>
            }
          />
          <Route
            path="/contact"
            element={
              <ProtectedRoute>
                <>
                  <MiniNavbar />
                  <Sidebar />
                  <Contact />
                  <Footer />
                </>
              </ProtectedRoute>
            }
          />
          <Route
            path="/aichat"
            element={
              <ProtectedRoute>
                <>
                  <MiniNavbar />
                  <Sidebar />
                  <AIChat />
                  <Footer />
                </>
              </ProtectedRoute>
            }
          />
          <Route
            path="/chatbot"
            element={
              <>
                <NavBar />
                <Sidebar />
                <Chatbot />
                <Footer />
              </>
            }
          />
          <Route
            path="/currency-converter"
            element={
              <ProtectedRoute>
                <>
                  <MiniNavbar />
                  <Sidebar />
                  <CurrencyConverter />
                  <Footer />
                </>
              </ProtectedRoute>
            }
          />
          <Route
            path="/dashboard/tabs"
            element={
              <ProtectedRoute>
                <>
                  <MiniNavbar />
                  <Sidebar />
                  <TabsView />
                  <Footer />
                </>
              </ProtectedRoute>
            }
          />
          <Route
            path="/financial-report"
            element={
              <ProtectedRoute>
                <>
                  <MiniNavbar />
                  <Sidebar />
                  <FinancialReportPage />
                  <Footer />
                </>
              </ProtectedRoute>
            }
          />
          <Route
            path="/profile-card"
            element={
              <ProtectedRoute>
                <>
                  <MiniNavbar />
                  <Sidebar />
                  <ProfileCardPage />
                  <Footer />
                </>
              </ProtectedRoute>
            }
          />
          <Route
            path="/expenses"
            element={
              <ProtectedRoute>
                <>
                  <MiniNavbar />
                  <Sidebar />
                  <CalendarExpenseTracker />
                </>
              </ProtectedRoute>
            }
          />
          <Route
            path="/finance"
            element={
              <ProtectedRoute>
                <>
                  <MiniNavbar />
                  <Sidebar />
                  <FinanceTools />
                  <Footer />
                </>
              </ProtectedRoute>
            }
          />
          <Route
            path="/FinanceTools"
            element={
              <ProtectedRoute>
                <>
                  <MiniNavbar />
                  <Sidebar />
                  <FinanceTools />
                  <Footer />
                </>
              </ProtectedRoute>
            }
          />

          {/* Admin Routes */}
          <Route
            path="/admin/dashboard"
            element={
              <AdminRoute>
                <AdminDashboard />
              </AdminRoute>
            }
          />
          <Route
            path="/admin/users/:id"
            element={
              <AdminRoute>
                <UserDetails />
              </AdminRoute>
            }
          />
        </Routes>
      </BrowserRouter>
    </AuthProvider>
  );
}

export default App;
