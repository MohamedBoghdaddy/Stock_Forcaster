import React, {
  useState,
  useEffect,
  useRef,
  useCallback,
  useContext,
} from "react";
import {
  Container,
  Card,
  Form,
  Button,
  Spinner,
  Alert,
  Badge,
} from "react-bootstrap";
import { FaStopCircle, FaCopy, FaLanguage } from "react-icons/fa";
import Select from "react-select";
import ReactMarkdown from "react-markdown";
import axios from "axios";
import { ErrorBoundary } from "react-error-boundary";
import { useAuthContext } from "../../../context/AuthContext";
import { DashboardContext } from "../../../context/DashboardContext";
import "../styles/AIChat.css";

const API_URL = process.env.REACT_APP_API_URL || "http://localhost:8000" ||"http://127.0.0.1:8000";

const languageOptions = [
  { label: "English", value: "en" },
  { label: "Arabic", value: "ar" },
  { label: "Spanish", value: "es" },
  { label: "French", value: "fr" },
  { label: "German", value: "de" },
  { label: "Hindi", value: "hi" },
];

const useSpeech = () => {
  const synth = useRef(
    typeof window !== "undefined" ? window.speechSynthesis : null
  );

  useEffect(() => {
    return () => synth.current?.cancel();
  }, []);

  const speak = useCallback((text) => {
    if (!synth.current) return;
    const utterance = new SpeechSynthesisUtterance(text.slice(0, 250));
    utterance.lang = "en-US";
    synth.current.speak(utterance);
  }, []);

  const stop = useCallback(() => synth.current?.cancel(), []);

  return { speak, stop };
};

const MessageBubble = ({ msg, onCopy, onTranslate }) => (
  <div className={`message ${msg.role}-message`}>
    <ReactMarkdown className="message-content">{msg.text}</ReactMarkdown>
    <div className="message-footer">
      <small className="message-timestamp">{msg.time}</small>
      {msg.role === "ai" && (
        <div className="message-actions">
          <Button variant="link" onClick={() => onCopy(msg.text)}>
            <FaCopy size={14} />
          </Button>
          <Button variant="link" onClick={() => onTranslate(msg.text)}>
            <FaLanguage size={14} />
          </Button>
        </div>
      )}
    </div>
  </div>
);

const ErrorFallback = () => (
  <Alert variant="danger" className="m-3">
    Something went wrong. Please refresh the page.
  </Alert>
);

const AIChat = () => {
  const { state } = useAuthContext();
  const { profileData } = useContext(DashboardContext);
  const { user, token } = state;
  const { speak, stop } = useSpeech();

  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [selectedLanguage, setSelectedLanguage] = useState(languageOptions[0]);
  const [error, setError] = useState(null);
  const [dots, setDots] = useState("");
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  const formatTime = () =>
    new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });

  const getAxiosConfig = () => {
    const config = {};
    if (token) {
      config.headers = { Authorization: `Bearer ${token}` };
    }
    return config;
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;

    const userMsg = { role: "user", text: input, time: formatTime() };
    setMessages((prev) => [...prev, userMsg]);
    setInput("");
    setIsLoading(true);
    setError(null);

    try {
      const response = await axios.post(
        `${API_URL}/api/chat`,
        {
          message: input,
          userId: user?._id || "anonymous",
          salary: profileData?.salary,
          conversationHistory: messages
            .filter((m) => m.role === "user" || m.role === "ai")
            .map((m) => ({ role: m.role, content: m.text })),
        },
        getAxiosConfig()
      );

      const aiMsg = {
        role: "ai",
        text: response.data?.response || "🤖 No response from AI.",
        time: formatTime(),
      };

      setMessages((prev) => [...prev, aiMsg]);
      speak(aiMsg.text);
    } catch (err) {
      console.error("API Error:", err);
      setError(
        err.response?.data?.message ||
          "Failed to process your message. Please try again."
      );

      setMessages((prev) => [
        ...prev,
        {
          role: "error",
          text: "❌ AI is currently unavailable. Please try again later.",
          time: formatTime(),
        },
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  const copyToClipboard = async (text) => {
    try {
      await navigator.clipboard.writeText(text);
      alert("Copied to clipboard!");
    } catch (err) {
      console.error("Copy failed:", err);
    }
  };

  const translateMessage = async (text) => {
    try {
      const res = await axios.post(
        `${API_URL}/api/translate`,
        {
          text,
          targetLang: selectedLanguage.value,
        },
        getAxiosConfig()
      );

      setMessages((prev) => [
        ...prev,
        {
          role: "ai",
          text: `🌍 **${selectedLanguage.label} Translation:** ${res.data.translation}`,
          time: formatTime(),
        },
      ]);
    } catch (err) {
      console.error("Translation Error:", err);
      setMessages((prev) => [
        ...prev,
        {
          role: "error",
          text: "❌ Translation service unavailable",
          time: formatTime(),
        },
      ]);
    }
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  useEffect(() => {
    if (isLoading) {
      const interval = setInterval(() => {
        setDots((prev) => (prev.length >= 3 ? "" : prev + "."));
      }, 500);
      return () => clearInterval(interval);
    }
  }, [isLoading]);

  return (
    <ErrorBoundary FallbackComponent={ErrorFallback}>
      <Container className="centered-chat-container">
        <Card className="large-chat-card">
          <Card.Header className="chat-header">
            <div className="chat-title">
              <h2>💬 AI Financial Advisor</h2>
              <div className="profile-badges">
                {user?.username && (
                  <Badge bg="primary" className="me-2">
                    {user.username}
                  </Badge>
                )}
                {profileData?.salary && (
                  <Badge bg="success">
                    Annual Salary: ${profileData.salary.toLocaleString()}
                  </Badge>
                )}
              </div>
            </div>
          </Card.Header>

          <Card.Body className="chat-body">
            {messages.length === 0 && (
              <div className="welcome-message">
                <p>Welcome! Ask me about:</p>
                <ul>
                  <li>Investment strategies</li>
                  <li>Retirement planning</li>
                  <li>Budget optimization</li>
                  <li>Tax-saving methods</li>
                </ul>
              </div>
            )}

            {messages.map((msg, idx) => (
              <div key={`${msg.time}-${idx}`} className="message-row">
                <MessageBubble
                  msg={msg}
                  onCopy={copyToClipboard}
                  onTranslate={translateMessage}
                />
              </div>
            ))}

            {isLoading && (
              <div className="typing-indicator">
                <Spinner animation="border" size="sm" className="me-2" />
                <span>Generating response{dots}</span>
              </div>
            )}
            <div ref={messagesEndRef} />
          </Card.Body>

          <Card.Footer className="chat-footer">
            {error && <Alert variant="danger">{error}</Alert>}

            <Form onSubmit={handleSubmit} className="chat-form">
              <div className="chat-input-container">
                <Form.Control
                  as="textarea"
                  rows={2}
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  placeholder="Ask about investments, savings, or financial planning..."
                  className="enhanced-input"
                  disabled={isLoading}
                />

                <div className="controls-wrapper">
                  <Select
                    className="language-select"
                    options={languageOptions}
                    value={selectedLanguage}
                    onChange={setSelectedLanguage}
                    isSearchable={false}
                    isDisabled={isLoading}
                  />

                  <Button
                    variant="primary"
                    type="submit"
                    className="send-button"
                    disabled={isLoading || !input.trim()}
                  >
                    {isLoading ? (
                      <Spinner animation="border" size="sm" />
                    ) : (
                      "Send"
                    )}
                  </Button>

                  <Button
                    variant="danger"
                    onClick={stop}
                    className="stop-button"
                    title="Stop speech"
                    disabled={isLoading}
                  >
                    <FaStopCircle />
                  </Button>
                </div>
              </div>
            </Form>
          </Card.Footer>
        </Card>
      </Container>
    </ErrorBoundary>
  );
};

export default AIChat;
