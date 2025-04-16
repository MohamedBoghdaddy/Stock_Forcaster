import { useState, useEffect, useRef, useCallback } from "react";
import { useAuthContext } from "../../../context/AuthContext";
import Cookies from "js-cookie";
import "../styles/AIChat.css";

const API_URLS = [
  "http://127.0.0.1:5000/api/chat", // Flask API
  "http://localhost:4000/api/chat/chat", // Express API
  "http://127.0.0.1:8000/api/chat", // FastAPI
];

const USER_ANALYSIS_API_URL = "http://127.0.0.1:5000/api/user";

const AIChat = () => {
  const { state } = useAuthContext();
  const { user, isAuthenticated } = state;
  const [userInput, setUserInput] = useState("");
  const [chatHistory, setChatHistory] = useState([]);
  const [questionnaire, setQuestionnaire] = useState(null);
  const [financialAnalysis, setFinancialAnalysis] = useState(null);
  const [loading, setLoading] = useState(false);
  const chatEndRef = useRef(null);

  // ✅ Fetch Financial Analysis
  const fetchFinancialAnalysis = useCallback(async () => {
    if (!isAuthenticated || !user) return;

    try {
      const token = Cookies.get("token") || localStorage.getItem("token");
      if (!token) throw new Error("No authentication token found.");

      const response = await fetch(USER_ANALYSIS_API_URL, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${token}`,
        },
        credentials: "include",
        body: JSON.stringify({ userId: user._id }),
      });

      if (!response.ok)
        throw new Error(`Failed to fetch analysis: ${response.status}`);

      const data = await response.json();
      setFinancialAnalysis(data);
    } catch (error) {
      console.error("❌ Financial Analysis Fetch Error:", error.message);
    }
  }, [isAuthenticated, user]);

  // ✅ Fetch Questionnaire
  const fetchQuestionnaire = useCallback(async () => {
    if (!isAuthenticated || !user) return;

    try {
      const token = Cookies.get("token") || localStorage.getItem("token");
      if (!token) throw new Error("No authentication token found.");

      const response = await fetch(
        "http://localhost:4000/api/questionnaire/latest",
        {
          method: "GET",
          headers: {
            Authorization: `Bearer ${token}`,
            "Content-Type": "application/json",
          },
          credentials: "include",
        }
      );

      if (!response.ok)
        throw new Error(`Failed to fetch questionnaire: ${response.status}`);

      const data = await response.json();
      setQuestionnaire(data);
    } catch (error) {
      console.error("❌ Questionnaire Fetch Error:", error.message);
    }
  }, [isAuthenticated, user]);

  useEffect(() => {
    fetchQuestionnaire();
    fetchFinancialAnalysis();
  }, [fetchQuestionnaire, fetchFinancialAnalysis]);

  // ✅ Scroll to latest message
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [chatHistory]);

  // ✅ Send Message to AI
  const sendMessageToAI = async (e) => {
    e.preventDefault();
    if (!user?._id || !user.salary || !userInput.trim()) return;

    const newUserMessage = { role: "user", text: userInput };
    setChatHistory((prev) => [...prev, newUserMessage]);
    setUserInput("");
    setLoading(true);

    try {
      let response;
      for (const url of API_URLS) {
        response = await fetch(url, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            userId: user._id,
            salary: user.salary,
            message: userInput,
          }),
        });

        if (response.ok) break;
      }

      if (!response.ok) throw new Error(`Server error: ${response.status}`);

      const data = await response.json();
      setChatHistory((prev) => [
        ...prev,
        { role: "ai", text: `🤖 AI Agent: Analyzing data...` },
        { role: "ai", text: data.response || "🤖 No response from AI." },
      ]);
    } catch (error) {
      console.error("❌ Error chatting with AI:", error);
      setChatHistory((prev) => [
        ...prev,
        { role: "ai", text: "❌ AI is unavailable." },
      ]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="ai-chat-container">
      <h2>💬 Financial AI Advisor</h2>

      {financialAnalysis ? (
        <div className="financial-analysis-box">
          <h3>📊 Financial Insights</h3>
          <p>
            <strong>Investment Recommendation:</strong>{" "}
            {financialAnalysis.investment_recommendation || "N/A"}
          </p>
          <p>
            <strong>Behavior Analysis:</strong>{" "}
            {financialAnalysis.survey_analysis?.financial_behavior.join(", ") ||
              "N/A"}
          </p>
        </div>
      ) : (
        <p className="no-financial-analysis">❌ No financial analysis found.</p>
      )}

      {questionnaire ? (
        <div className="questionnaire-box">
          <h3>📋 Your Financial Profile</h3>
          {Object.entries(questionnaire).map(([key, value]) => (
            <p key={key}>
              <strong>{key.replace(/_/g, " ").toUpperCase()}:</strong>{" "}
              {value || "N/A"}
            </p>
          ))}
        </div>
      ) : (
        <p className="no-questionnaire">❌ No questionnaire found.</p>
      )}

      <div className="chat-box">
        {chatHistory.map((msg, index) => (
          <div key={index} className={`message ${msg.role}-message`}>
            <span dangerouslySetInnerHTML={{ __html: msg.text }} />
          </div>
        ))}
        <div ref={chatEndRef} />
      </div>

      <form onSubmit={sendMessageToAI} className="chat-form">
        <input
          type="text"
          value={userInput}
          onChange={(e) => setUserInput(e.target.value)}
          className="chat-input"
          placeholder="Ask about investments..."
          disabled={loading}
        />
        <button type="submit" className="chat-button" disabled={loading}>
          {loading ? "⏳ Thinking..." : "Send"}
        </button>
      </form>
    </div>
  );
};

export default AIChat;
