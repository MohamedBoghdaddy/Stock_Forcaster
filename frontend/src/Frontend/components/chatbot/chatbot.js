import React, { useState, useContext, useEffect, useRef } from "react";
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import {
  faCommentDots,
  faPaperPlane,
  faTimes,
  faCircle,
  faUser,
  faRobot,
  faChartLine,
} from "@fortawesome/free-solid-svg-icons";
import axios from "axios";
import { DashboardContext } from "../../../context/DashboardContext";
import logo from "../../../assets/latest_logo.png";
import "../styles/chatbot.css";

const API_URL = process.env.REACT_APP_API_URL || "http://localhost:8000";

const Chatbot = () => {
  const dashboardContext = useContext(DashboardContext);
  const profile = dashboardContext?.profile || {};
  const fetchProfile = dashboardContext?.actions?.fetchProfile;

  const [isChatOpen, setIsChatOpen] = useState(false);
  const [messages, setMessages] = useState([]);
  const [inputText, setInputText] = useState("");
  const [loading, setLoading] = useState(false);
  const [goal, setGoal] = useState("investment");
  const [typingContent, setTypingContent] = useState("");
  const [isTyping, setIsTyping] = useState(false);
  const messagesEndRef = useRef(null);
  const inputRef = useRef(null);

  // Fetch profile if not available
  useEffect(() => {
    if (!profile || Object.keys(profile).length === 0) {
      fetchProfile?.();
    }
  }, [fetchProfile, profile]);

  // Scroll to bottom when messages change
  useEffect(() => {
    scrollToBottom();
  }, [messages, typingContent]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  const toggleChat = () => {
    setIsChatOpen(!isChatOpen);
    if (!isChatOpen) {
      setTimeout(() => {
        inputRef.current?.focus();
      }, 300);
    }
  };

  const simulateTyping = (response) => {
    setIsTyping(true);
    let index = 0;
    const typingInterval = setInterval(() => {
      if (index < response.length) {
        setTypingContent((prev) => prev + response.charAt(index));
        index++;
      } else {
        clearInterval(typingInterval);
        setMessages((prev) => [
          ...prev,
          {
            sender: "bot",
            text: response,
            timestamp: new Date().toISOString(),
          },
        ]);
        setTypingContent("");
        setIsTyping(false);
      }
    }, 20);
  };

  const sendProfile = async () => {
    if (!profile || Object.keys(profile).length === 0) {
      setMessages([
        ...messages,
        {
          sender: "user",
          text: `Generate ${goal.replace("_", " ")} plan`,
          timestamp: new Date().toISOString(),
        },
        {
          sender: "bot",
          text: "⚠️ Please complete your profile first.",
          timestamp: new Date().toISOString(),
        },
      ]);
      return;
    }

    setLoading(true);
    const userMessage = {
      sender: "user",
      text: `Generate ${goal.replace("_", " ")} plan`,
      timestamp: new Date().toISOString(),
    };

    setMessages((prev) => [...prev, userMessage]);

    try {
      const endpoint = `${API_URL}/chatbot/generate/${
        goal === "life_management" ? "life" : "investment"
      }`;

      const res = await axios.post(endpoint, profile);
      const responseText =
        typeof res.data === "object"
          ? JSON.stringify(res.data, null, 2)
          : res.data;

      simulateTyping(responseText);
    } catch (err) {
      console.error(err);
      setMessages((prev) => [
        ...prev,
        {
          sender: "bot",
          text: "❌ Failed to generate plan. Please try again.",
          timestamp: new Date().toISOString(),
        },
      ]);
    } finally {
      setLoading(false);
    }
  };

  const sendMessage = async () => {
    if (!inputText.trim()) return;

    const userMessage = {
      sender: "user",
      text: inputText.trim(),
      timestamp: new Date().toISOString(),
    };

    setMessages((prev) => [...prev, userMessage]);
    setInputText("");
    setLoading(true);

    try {
      const res = await axios.post(`${API_URL}/chatbot/chat`, {
        message: inputText,
        profile: profile,
      });

      const responseText =
        res.data?.output || "🤖 I couldn't process that request";
      simulateTyping(responseText);
    } catch (err) {
      console.error(err);
      setMessages((prev) => [
        ...prev,
        {
          sender: "bot",
          text: "❌ Something went wrong. Please try again.",
          timestamp: new Date().toISOString(),
        },
      ]);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  const formatMessage = (content) => {
    if (!content) return "";

    // Convert markdown-style links
    const linkRegex = /\[([^\]]+)\]\(([^)]+)\)/g;
    const formattedContent = content.replace(
      linkRegex,
      '<a href="$2" target="_blank" rel="noopener noreferrer">$1</a>'
    );

    // Split into paragraphs
    return formattedContent
      .split("\n\n")
      .map((paragraph, i) => (
        <p key={i} dangerouslySetInnerHTML={{ __html: paragraph }} />
      ));
  };

  return (
    <div className="chatbot-wrapper">
      {/* Floating chat toggle button */}
      <button
        className={`chat-toggle-btn ${isChatOpen ? "active" : ""}`}
        onClick={toggleChat}
        aria-label={isChatOpen ? "Close chat" : "Open financial advisor"}
      >
        <FontAwesomeIcon icon={faChartLine} />
        <span className="chat-toggle-text">Financial Advisor</span>
      </button>

      {/* Chat container */}
      <div className={`chatbot-container ${isChatOpen ? "open" : "closed"}`}>
        {/* Chat header */}
        <div className="chat-header">
          <div className="chat-title">
            <div className="chat-avatar">
              <img src={logo} alt="Financial Assistant" />
            </div>
            <div>
              <h5>AI Financial Advisor</h5>
              <p className="chat-status">
                <FontAwesomeIcon icon={faCircle} className="status-icon" />
                {loading ? "Processing..." : "Online"}
              </p>
            </div>
          </div>
          <button
            className="chat-close-btn"
            onClick={toggleChat}
            aria-label="Close chat"
          >
            <FontAwesomeIcon icon={faTimes} />
          </button>
        </div>

        {/* Chat messages */}
        <div className="chat-messages-container">
          {messages.length === 0 ? (
            <div className="chat-welcome">
              <img
                src={logo}
                alt="Financial Assistant"
                className="welcome-image"
              />
              <h4>Welcome to your AI Financial Advisor!</h4>
              <p>
                I can help you with investment strategies, financial planning,
                budgeting advice, and market insights. How can I assist you
                today?
              </p>
              <div className="quick-options">
                <button onClick={() => setGoal("investment") || sendProfile()}>
                  Investment Plan
                </button>
                <button
                  onClick={() => setGoal("life_management") || sendProfile()}
                >
                  Life Management
                </button>
              </div>
            </div>
          ) : (
            messages.map((msg, index) => (
              <div
                key={index}
                className={`message-row ${
                  msg.sender === "user" ? "user-row" : "bot-row"
                }`}
              >
                <div className="message-avatar">
                  <FontAwesomeIcon
                    icon={msg.sender === "user" ? faUser : faRobot}
                    size="sm"
                  />
                </div>
                <div
                  className={`message ${
                    msg.sender === "user" ? "user-message" : "bot-message"
                  }`}
                >
                  {formatMessage(msg.text)}
                  <div className="message-timestamp">
                    {new Date(msg.timestamp).toLocaleTimeString([], {
                      hour: "2-digit",
                      minute: "2-digit",
                    })}
                  </div>
                </div>
              </div>
            ))
          )}

          {/* Typing indicator */}
          {isTyping && (
            <div className="message-row bot-row">
              <div className="message-avatar">
                <FontAwesomeIcon icon={faRobot} size="sm" />
              </div>
              <div className="message bot-message typing-message">
                {typingContent}
                <span className="typing-cursor"></span>
              </div>
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>

        {/* Action bar for plan generation */}
        <div className="chat-action-bar">
          <select
            value={goal}
            onChange={(e) => setGoal(e.target.value)}
            disabled={loading}
          >
            <option value="investment">Investment Plan</option>
            <option value="life_management">Life Management</option>
          </select>
          <button
            onClick={sendProfile}
            disabled={loading}
            className="generate-btn"
          >
            Generate Plan
          </button>
        </div>

        {/* Message input */}
        <div className="chat-input-container">
          <textarea
            ref={inputRef}
            className="chat-input"
            placeholder="Ask about investments, savings, or financial advice..."
            value={inputText}
            onChange={(e) => setInputText(e.target.value)}
            onKeyDown={handleKeyDown}
            rows="1"
            disabled={loading}
          ></textarea>
          <button
            className="chat-send-btn"
            onClick={sendMessage}
            disabled={!inputText.trim() || loading}
          >
            <FontAwesomeIcon icon={faPaperPlane} />
          </button>
        </div>
      </div>
    </div>
  );
};

export default Chatbot;
