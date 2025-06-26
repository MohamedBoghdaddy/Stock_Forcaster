import React, { useState, useContext, useEffect, useRef } from "react";
import axios from "axios";
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import {
  faPaperPlane,
  faChartLine,
  faLightbulb,
  faNewspaper,
  faQuestionCircle,
} from "@fortawesome/free-solid-svg-icons";
import { DashboardContext } from "../../../context/DashboardContext";
import "../styles/chatbot.css";

const API_URL = process.env.REACT_APP_API_URL || "http://localhost:8000";

const Chatbot = () => {
  const dashboardContext = useContext(DashboardContext);
  const profile = dashboardContext?.profile || {};
  const fetchProfile = dashboardContext?.actions?.fetchProfile;
  const messagesEndRef = useRef(null);

  const [messages, setMessages] = useState([
    {
      sender: "bot",
      text: "Hi! I'm your AI Stock Advisor. Ask me about stocks, portfolios, or market news!",
    },
  ]);
  const [loading, setLoading] = useState(false);
  const [inputText, setInputText] = useState("");
  const [isTyping, setIsTyping] = useState(false);
  const [typingContent, setTypingContent] = useState("");
  const [sessionId, setSessionId] = useState("");

  useEffect(() => {
    console.log("📦 useEffect: checking profile and initializing session...");
    if (!profile || Object.keys(profile).length === 0) {
      console.log("🔄 Fetching profile...");
      fetchProfile?.();
    }

    const newSessionId = generateSessionId();
    console.log("🆕 Session ID initialized:", newSessionId);
    setSessionId(newSessionId);
  }, [fetchProfile, profile]);

  const generateSessionId = () => {
    return `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, isTyping]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  const simulateTyping = (response, callback) => {
    console.log("⏳ Simulating typing...");
    setIsTyping(true);
    setTypingContent("");
    let index = 0;
    const typingInterval = setInterval(() => {
      if (index < response.length) {
        setTypingContent((prev) => prev + response.charAt(index));
        index++;
      } else {
        clearInterval(typingInterval);
        if (callback) callback(response);
        setTypingContent("");
        setIsTyping(false);
        console.log("✅ Typing simulation finished.");
      }
    }, 20);
  };

  const generateStockAdvice = async () => {
    console.log("📈 Requesting stock advice...");

    if (!profile || Object.keys(profile).length === 0) {
      console.warn("⚠️ Profile missing, cannot generate advice");
      setMessages((prev) => [
        ...prev,
        {
          sender: "bot",
          text: "⚠️ Please complete your profile first to get personalized stock recommendations.",
        },
      ]);
      return;
    }

    setLoading(true);
    setMessages((prev) => [
      ...prev,
      { sender: "user", text: "Generate stock investment advice" },
    ]);

    try {
      const token = localStorage.getItem("token");
      console.log("🔐 Token retrieved:", token);

      const res = await axios.post(
        `${API_URL}/chatbot/generate/investment`,
        {},
        {
          headers: {
            Authorization: `Bearer ${token}`,
          },
        }
      );

      console.log("✅ Stock advice API response:", res.data);

      let adviceText = "";
      if (res.data.investment_plan) {
        adviceText = res.data.investment_plan
          .map((item) => `• ${item}`)
          .join("\n");
      } else if (res.data.output) {
        adviceText = res.data.output;
      } else {
        adviceText = "Received advice in an unexpected format.";
      }

      simulateTyping(adviceText, (fullResponse) => {
        console.log("💬 Final bot message:", fullResponse);
        setMessages((prev) => [...prev, { sender: "bot", text: fullResponse }]);
      });
    } catch (err) {
      console.error("❌ Stock advice error:", err);
      setMessages((prev) => [
        ...prev,
        {
          sender: "bot",
          text: "❌ Couldn't generate stock advice. Please try again later.",
        },
      ]);
    } finally {
      setLoading(false);
    }
  };

  const sendMessage = async () => {
    if (!inputText.trim()) return;

    const userMessage = inputText;
    console.log("📤 Sending message:", userMessage);

    setMessages((prev) => [...prev, { sender: "user", text: userMessage }]);
    setLoading(true);
    setInputText("");

    try {
      const token = localStorage.getItem("token");
      console.log("🔐 Token used:", token);
      console.log("📡 Sending to:", `${API_URL}/chatbot/chat`);

      const res = await axios.post(
        `${API_URL}/chatbot/chat`,
        {
          message: userMessage,
          session_id: sessionId,
        },
        {
          headers: {
            Authorization: `Bearer ${token}`,
          },
        }
      );

      console.log("✅ Chat response:", res.data);

      if (res.data.session_id) {
        console.log("🔄 Updating session ID:", res.data.session_id);
        setSessionId(res.data.session_id);
      }

      simulateTyping(res.data.output, (fullResponse) => {
        console.log("🤖 Bot reply complete:", fullResponse);
        setMessages((prev) => [...prev, { sender: "bot", text: fullResponse }]);
      });
    } catch (err) {
      console.error("❌ Chat error:", err.response?.data || err.message);
      setMessages((prev) => [
        ...prev,
        {
          sender: "bot",
          text: "❌ Service unavailable. Please try again later.",
        },
      ]);
    } finally {
      setLoading(false);
    }
  };

  const quickAction = (action) => {
    console.log("⚡ Quick action triggered:", action);
    let message = "";
    switch (action) {
      case "price":
        message = "What's the current price of Apple stock?";
        break;
      case "recommend":
        message = "Recommend stocks based on my risk profile";
        break;
      case "news":
        message = "Show me the latest stock market news";
        break;
      case "faq":
        message = "How do I analyze stocks?";
        break;
      default:
        return;
    }

    setInputText(message);
    setTimeout(() => {
      sendMessage();
      scrollToBottom();
    }, 300);
  };

  const formatMessage = (text) => {
    return text.split("\n").map((line, i) => <p key={i}>{line}</p>);
  };

  return (
    <div className="chatbot-fullpage">
      <div className="chat-header">
        <div className="header-content">
          <div className="title-icon">
            <FontAwesomeIcon icon={faChartLine} />
          </div>
          <div>
            <h1>AI Stock Advisor</h1>
            <p className="subtitle">Expert stock market advice and analysis</p>
          </div>
        </div>
      </div>

      <div className="chat-container">
        <div className="chat-messages">
          {messages.map((msg, index) => (
            <div key={index} className={`message-bubble ${msg.sender}`}>
              <div className="bubble-content">{formatMessage(msg.text)}</div>
            </div>
          ))}

          {isTyping && (
            <div className="message-bubble bot">
              <div className="bubble-content">
                {formatMessage(typingContent)}
                <div className="typing-indicator">
                  <span></span>
                  <span></span>
                  <span></span>
                </div>
              </div>
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>

        <div className="quick-actions">
          <h3>
            <FontAwesomeIcon icon={faLightbulb} /> Quick Actions
          </h3>
          <div className="action-buttons">
            <button onClick={() => quickAction("price")} disabled={loading}>
              <FontAwesomeIcon icon={faChartLine} /> Stock Price
            </button>
            <button onClick={generateStockAdvice} disabled={loading}>
              <FontAwesomeIcon icon={faLightbulb} /> Get Recommendations
            </button>
            <button onClick={() => quickAction("news")} disabled={loading}>
              <FontAwesomeIcon icon={faNewspaper} /> Market News
            </button>
            <button onClick={() => quickAction("faq")} disabled={loading}>
              <FontAwesomeIcon icon={faQuestionCircle} /> Stock FAQs
            </button>
          </div>
        </div>

        <div className="chat-controls">
          <div className="input-group">
            <input
              type="text"
              placeholder="Ask about stocks, portfolios, or market trends..."
              value={inputText}
              onChange={(e) => setInputText(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && !loading && sendMessage()}
              disabled={loading}
            />
            <button
              onClick={sendMessage}
              disabled={loading || !inputText.trim()}
              className="send-button"
            >
              <FontAwesomeIcon icon={faPaperPlane} />
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Chatbot;
