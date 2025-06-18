import { useState, useEffect, useRef, useCallback, FormEvent } from "react";
import { FaStopCircle, FaCopy, FaLanguage } from "react-icons/fa";
import axios from "axios";
import ReactMarkdown from "react-markdown";
import Select from "react-select";
import { ErrorBoundary } from "react-error-boundary";

type Message = {
  role: "user" | "ai";
  text: string;
  time: string;
};

const fallbackUrl =
  window.location.hostname === "localhost"
    ? "http://localhost:8000"
    : "http://127.0.0.1:8000";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || fallbackUrl;

const languageOptions = [
  { label: "Arabic", value: "ar" },
  { label: "Spanish", value: "es" },
  { label: "French", value: "fr" },
  { label: "German", value: "de" },
  { label: "Hindi", value: "hi" },
];

const useSpeech = () => {
  const synth = useRef<SpeechSynthesis | null>(null);
  useEffect(() => {
    if ("speechSynthesis" in window) {
      synth.current = window.speechSynthesis;
    }
    return () => synth.current?.cancel();
  }, []);

  const speak = useCallback((text: string) => {
    if (!synth.current) return;
    synth.current.cancel();
    const utterance = new SpeechSynthesisUtterance(text.slice(0, 200));
    utterance.lang = "en-US";
    synth.current.speak(utterance);
  }, []);

  const stop = useCallback(() => synth.current?.cancel(), []);
  return { speak, stop };
};

const MessageBubble = ({
  msg,
  onCopy,
  onTranslate,
}: {
  msg: Message;
  onCopy: (text: string) => void;
  onTranslate: (text: string) => void;
}) => (
  <div
    className={`p-2 rounded-lg ${
      msg.role === "user"
        ? "bg-blue-100 text-blue-900 self-end"
        : "bg-gray-100 text-gray-800"
    }`}
  >
    <ReactMarkdown>{msg.text}</ReactMarkdown>
    <div className="flex justify-between items-center mt-1">
      <span className="text-[10px] text-gray-400">{msg.time}</span>
      {msg.role === "ai" && (
        <div className="flex space-x-2">
          <button onClick={() => onCopy(msg.text)} aria-label="Copy message">
            <FaCopy size={12} />
          </button>
          <button
            onClick={() => onTranslate(msg.text)}
            aria-label="Translate message"
          >
            <FaLanguage size={12} />
          </button>
        </div>
      )}
    </div>
  </div>
);

const ErrorFallback = () => (
  <div className="p-4 bg-red-100 text-red-800 rounded-lg">
    Something went wrong. Please refresh the page.
  </div>
);

const AIChat = () => {
  const [input, setInput] = useState("");
  const [chat, setChat] = useState<Message[]>([]);
  const [loading, setLoading] = useState(false);
  const [showChat, setShowChat] = useState(true);
  const [selectedLanguage, setSelectedLanguage] = useState(languageOptions[0]);
  const [isTyping, setIsTyping] = useState(false);
  const [dots, setDots] = useState("");
  const chatEndRef = useRef<HTMLDivElement>(null);
  const { speak, stop } = useSpeech();

  const formatTime = () =>
    new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });

  useEffect(() => {
    const savedChat = localStorage.getItem("aiChatHistory");
    if (savedChat) setChat(JSON.parse(savedChat));
  }, []);

  useEffect(() => {
    if (chat.length)
      localStorage.setItem("aiChatHistory", JSON.stringify(chat));
  }, [chat]);

  useEffect(() => {
    if (isTyping) {
      let dotCount = 0;
      const interval = setInterval(() => {
        setDots(".".repeat((dotCount % 3) + 1));
        dotCount++;
      }, 500);
      return () => clearInterval(interval);
    }
  }, [isTyping]);

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [chat]);

  const copyToClipboard = async (text: string) => {
    try {
      await navigator.clipboard.writeText(text);
      alert("Copied!");
    } catch (err) {
      console.error("Copy failed:", err);
    }
  };

  const translateMessage = async (text: string) => {
    try {
      const res = await axios.post(
        `${API_BASE_URL}/chatbot/translate`,
        {
          text,
          langCode: selectedLanguage.value.toLowerCase(),
        },
        {
          headers: { "Content-Type": "application/json" },
        }
      );
      setChat((prev) => [
        ...prev,
        {
          role: "ai",
          text: `🌍 ${selectedLanguage.label}: ${res.data.translatedText}`,
          time: formatTime(),
        },
      ]);
    } catch {
      setChat((prev) => [
        ...prev,
        {
          role: "ai",
          text: "❌ Translation service unavailable",
          time: formatTime(),
        },
      ]);
    }
  };

  const sendMessage = async (e: FormEvent) => {
    e.preventDefault();
    if (!input.trim()) return;

    setChat((prev) => [
      ...prev,
      { role: "user", text: input, time: formatTime() },
    ]);
    setInput("");
    setLoading(true);
    setIsTyping(true);

    try {
      await axios.get(`${API_BASE_URL}/chatbot/health`);

      const res = await axios.post(
        `${API_BASE_URL}/chatbot/chat`,
        { message: input },
        { headers: { "Content-Type": "application/json" }, timeout: 10000 }
      );

      const reply = res.data?.reply || res.data?.output || "🤖 No response.";
      setChat((prev) => [
        ...prev,
        { role: "ai", text: `🤖 ${reply}`, time: formatTime() },
      ]);
      speak(reply);
    } catch (err) {
      console.error("Error:", err);
      setChat((prev) => [
        ...prev,
        {
          role: "ai",
          text: "❌ Server error or network issue",
          time: formatTime(),
        },
      ]);
    } finally {
      setLoading(false);
      setIsTyping(false);
    }
  };

  return (
    <ErrorBoundary FallbackComponent={ErrorFallback}>
      {showChat && (
        <div className="fixed bottom-20 right-4 bg-white shadow-lg rounded-lg w-80 max-h-[90vh] overflow-y-auto border z-40 flex flex-col">
          <div className="p-4 border-b flex justify-between items-center">
            <h2 className="text-lg font-bold">💬 Financial AI Advisor</h2>
            <Select
              options={languageOptions}
              value={selectedLanguage}
              onChange={(o) => o && setSelectedLanguage(o)}
              className="w-32"
              classNamePrefix="select"
              isSearchable={false}
            />
          </div>

          <div className="p-3 text-sm space-y-2 flex-1 overflow-y-auto">
            {chat.map((msg, i) => (
              <MessageBubble
                key={`${msg.role}-${i}`}
                msg={msg}
                onCopy={copyToClipboard}
                onTranslate={translateMessage}
              />
            ))}
            {isTyping && (
              <div className="bg-gray-100 text-gray-800 p-2 rounded-lg">
                Bot is typing{dots}
              </div>
            )}
            <div ref={chatEndRef} />
          </div>

          <form onSubmit={sendMessage} className="p-3 border-t flex gap-2">
            <input
              value={input}
              onChange={(e) => setInput(e.target.value)}
              className="flex-1 p-2 border rounded text-sm"
              placeholder="Ask a question..."
              disabled={loading}
            />
            <button
              type="submit"
              disabled={loading}
              className="bg-blue-600 text-white px-3 py-1 rounded text-sm"
            >
              {loading ? "..." : "Send"}
            </button>
          </form>
        </div>
      )}

      <div className="fixed bottom-4 right-4 flex flex-col items-center space-y-2">
        <button
          onClick={() => setShowChat(!showChat)}
          className="bg-blue-600 p-3 rounded-full text-white"
        >
          {showChat ? "👁️" : "💬"}
        </button>
        <button
          onClick={stop}
          className="bg-gray-600 p-3 rounded-full text-white"
        >
          <FaStopCircle size={18} />
        </button>
      </div>
    </ErrorBoundary>
  );
};

export default AIChat;
