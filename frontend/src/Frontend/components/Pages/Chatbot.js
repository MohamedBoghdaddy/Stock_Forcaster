import React, { useState, useContext, useEffect, useRef } from "react";
import axios from "axios";
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import {
  faPaperPlane,
  faChartLine,
  faLightbulb,
} from "@fortawesome/free-solid-svg-icons";
import { DashboardContext } from "../../../context/DashboardContext";
import "../styles/chatbot.css";

const API_URL =
  process.env.REACT_APP_API_URL ||
  "http://127.0.0.1:8000" ||
  "http://localhost:8000";

const Chatbot = () => {
  const dashboardContext = useContext(DashboardContext);
  const profile = dashboardContext?.profile || {};
  const fetchProfile = dashboardContext?.actions?.fetchProfile;
  const messagesEndRef = useRef(null);

  const [goal, setGoal] = useState("investment");
  const [messages, setMessages] = useState([
    {
      sender: "bot",
      text: "Hi! I'm your AI Financial Advisor. What would you like help with?",
    },
  ]);
  const [loading, setLoading] = useState(false);
  const [inputText, setInputText] = useState("");
  const [isTyping, setIsTyping] = useState(false);
  const [typingContent, setTypingContent] = useState("");

  // Fetch profile if not already available
  useEffect(() => {
    if (!profile || Object.keys(profile).length === 0) {
      fetchProfile?.();
    }
  }, [fetchProfile, profile]);

  // Auto-scroll to bottom of messages
  useEffect(() => {
    scrollToBottom();
  }, [messages, isTyping]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  const simulateTyping = (response) => {
    setIsTyping(true);
    setTypingContent("");
    let index = 0;
    const typingInterval = setInterval(() => {
      if (index < response.length) {
        setTypingContent((prev) => prev + response.charAt(index));
        index++;
      } else {
        clearInterval(typingInterval);
        setMessages((prev) => [...prev, { sender: "bot", text: response }]);
        setTypingContent("");
        setIsTyping(false);
      }
    }, 20);
  };

  const sendProfile = async () => {
    if (!profile || Object.keys(profile).length === 0) {
      setMessages((prev) => [
        ...prev,
        { sender: "bot", text: "⚠️ Please complete your profile first." },
      ]);
      return;
    }

    setLoading(true);
    const userMessage = `Generate ${goal.replace("_", " ")} plan`;
    setMessages((prev) => [...prev, { sender: "user", text: userMessage }]);

    const endpoint = `${API_URL}/chatbot/generate/${
      goal === "life_management" ? "life" : "investment"
    }`;

    try {
      const res = await axios.post(endpoint, profile);
      const formatted = JSON.stringify(res.data, null, 2);
      simulateTyping(formatted);
    } catch (err) {
      console.error(err);
      setMessages((prev) => [
        ...prev,
        {
          sender: "bot",
          text: "❌ Something went wrong while generating your plan.",
        },
      ]);
    } finally {
      setLoading(false);
    }
  };

  const sendFreeText = async () => {
    if (!inputText.trim()) return;

    setMessages((prev) => [...prev, { sender: "user", text: inputText }]);
    setLoading(true);
    setInputText("");

    try {
      const res = await axios.post(`${API_URL}/chatbot/chat`, {
        message: inputText,
        profile: profile,
      });
      const responseText = res.data?.output || "🤖 I didn't understand that";
      simulateTyping(responseText);
    } catch (err) {
      console.error(err);
      setMessages((prev) => [
        ...prev,
        {
          sender: "bot",
          text: "❌ Something went wrong. Please try again.",
        },
      ]);
    } finally {
      setLoading(false);
    }
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
            <h1>AI Financial Advisor</h1>
            <p className="subtitle">Ask anything about finances</p>
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
            <button onClick={() => setGoal("investment") || sendProfile()}>
              Investment Plan
            </button>
            <button onClick={() => setGoal("life_management") || sendProfile()}>
              Life Management
            </button>
            <button
              onClick={() => {
                setInputText("Show me investment opportunities");
                setTimeout(sendFreeText, 300);
              }}
            >
              Investment Ideas
            </button>
          </div>
        </div>

        <div className="chat-controls">
          <div className="input-group">
            <input
              type="text"
              placeholder="Ask a question..."
              value={inputText}
              onChange={(e) => setInputText(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && sendFreeText()}
              disabled={loading}
            />
            <button
              onClick={sendFreeText}
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
