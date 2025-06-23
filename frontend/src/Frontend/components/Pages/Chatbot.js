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

  // Fetch profile if not already available
  useEffect(() => {
    if (!profile || Object.keys(profile).length === 0) {
      fetchProfile?.();
    }

    // Initialize session ID
    setSessionId(generateSessionId());
  }, [fetchProfile, profile]);

  // Generate unique session ID
  const generateSessionId = () => {
    return `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  };

  // Auto-scroll to bottom of messages
  useEffect(() => {
    scrollToBottom();
  }, [messages, isTyping]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  const simulateTyping = (response, callback) => {
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
      }
    }, 20);
  };

  const generateStockAdvice = async () => {
    if (!profile || Object.keys(profile).length === 0) {
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
      // Get token from localStorage
      const token = localStorage.getItem("token");

      // Updated endpoint and headers
      const res = await axios.post(
        `${API_URL}/chatbot/generate/investment`,
        {},
        {
          headers: {
            Authorization: `Bearer ${token}`,
          },
        }
      );

      // Handle new response format
      let adviceText = "";
      if (res.data.investment_plan) {
        // Format array as bullet points
        adviceText = res.data.investment_plan
          .map((item) => `• ${item}`)
          .join("\n");
      } else if (res.data.output) {
        adviceText = res.data.output;
      } else {
        adviceText = "Received advice in an unexpected format.";
      }

      simulateTyping(adviceText, (fullResponse) => {
        setMessages((prev) => [...prev, { sender: "bot", text: fullResponse }]);
      });
    } catch (err) {
      console.error("Stock advice error:", err);
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
    setMessages((prev) => [...prev, { sender: "user", text: userMessage }]);
    setLoading(true);
    setInputText("");

    try {
      // Get token from localStorage
      const token = localStorage.getItem("token");

      // Updated request with token in headers
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

      // Store session ID if we got one back
      if (res.data.session_id) {
        setSessionId(res.data.session_id);
      }

      simulateTyping(res.data.output, (fullResponse) => {
        setMessages((prev) => [...prev, { sender: "bot", text: fullResponse }]);
      });
    } catch (err) {
      console.error("Chat error:", err.response?.data || err.message);
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
