import React, { useState, useEffect, useRef } from "react";
import {  FaStopCircle, FaCopy, FaLanguage } from "react-icons/fa";
import axios from "axios";
import { useAuthContext } from "../../hooks/useAuthContext";
import ReactMarkdown from "react-markdown";
import Select from "react-select";

const API_BASE_URL = "http://localhost:8000"; // Base URL without endpoint

const languageOptions = [
  { label: "Arabic", value: "ar" },
  { label: "Spanish", value: "es" },
  { label: "French", value: "fr" },
  { label: "German", value: "de" },
  { label: "Hindi", value: "hi" },
];

type Message = {
  role: "user" | "ai";
  text: string;
  time: string;
};

const AIChat = () => {
  const { state } = useAuthContext();
  const { user } = state;

  const [input, setInput] = useState("");
  const [chat, setChat] = useState<Message[]>([]);
  const [loading, setLoading] = useState(false);
  const [showChat, setShowChat] = useState(true);
  const chatEndRef = useRef<HTMLDivElement>(null);
  const [isTyping, setIsTyping] = useState(false);
  const [selectedLanguage, setSelectedLanguage] = useState(languageOptions[0]);
  const [isRecording, setIsRecording] = useState(false);
  const [dots, setDots] = useState("");
  const speechSynth = useRef<SpeechSynthesis | null>(null);

  useEffect(() => {
    speechSynth.current = window.speechSynthesis;
    return () => {
      if (speechSynth.current) {
        speechSynth.current.cancel();
      }
    };
  }, []);

  useEffect(() => {
    let dotInterval: NodeJS.Timeout;
    if (isTyping) {
      let dotCount = 0;
      dotInterval = setInterval(() => {
        setDots(".".repeat((dotCount % 3) + 1));
        dotCount++;
      }, 500);
    }
    return () => clearInterval(dotInterval);
  }, [isTyping]);

  const formatTime = () =>
    new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });

  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [chat]);

const sendMessage = async (e: React.FormEvent) => {
  e.preventDefault();
  if (!input.trim()) return;

  const userMsg: Message = { role: "user", text: input, time: formatTime() };
  setChat((prev) => [...prev, userMsg]);
  setInput("");
  setLoading(true);
  setIsTyping(true);

  try {
    // First check if the API is reachable
    const healthCheck = await axios.get(`${API_BASE_URL}/api/health`);
    if (healthCheck.status !== 200) {
      throw new Error("API server not responding");
    }

    // Then make the chat request
    const response = await axios.post(
      `${API_BASE_URL}/api/chat`,
      {
        message: input,
        userId: user?.id,
      },
      {
        headers: {
          "Content-Type": "application/json",
        },
        timeout: 10000, // 10 second timeout
      }
    );

    const aiMsg: Message = {
      role: "ai",
      text: `🤖 ${response.data?.reply || "No response"}`,
      time: formatTime(),
    };
    setChat((prev) => [...prev, aiMsg]);
    speak(aiMsg.text);
  } catch (error) {
    console.error("API Error:", error);
    let errorMessage = "❌ Error connecting to AI service";

    if (axios.isAxiosError(error)) {
      if (error.response) {
        errorMessage = `❌ Server error: ${error.response.status}`;
      } else if (error.request) {
        errorMessage = "❌ No response from server";
      } else {
        errorMessage = `❌ Request error: ${error.message}`;
      }
    }

    const errorMsg: Message = {
      role: "ai",
      text: errorMessage,
      time: formatTime(),
    };
    setChat((prev) => [...prev, errorMsg]);
  } finally {
    setLoading(false);
    setIsTyping(false);
  }
};

  const speak = (text: string) => {
    if (!speechSynth.current) return;

    speechSynth.current.cancel();
    const utterance = new SpeechSynthesisUtterance();
    utterance.text = text.length > 200 ? `${text.substring(0, 200)}...` : text;
    utterance.lang = "en-US";
    utterance.pitch = 1.0;
    speechSynth.current.speak(utterance);
  };

  const stopSpeaking = () => {
    speechSynth.current?.cancel();
  };

  const copyToClipboard = async (text: string) => {
    try {
      await navigator.clipboard.writeText(text);
      alert("Copied to clipboard!");
    } catch (err) {
      console.error("Copy failed:", err);
    }
  };

  const translateMessage = async (text: string) => {
    try {
      const response = await axios.post(
        `${API_BASE_URL}/api/translate`,
        {
          text,
          langCode: selectedLanguage.value,
        },
        {
          headers: {
            "Content-Type": "application/json",
          },
        }
      );

      const translatedMsg: Message = {
        role: "ai",
        text: `🌍 ${selectedLanguage.label}: ${response.data.translatedText}`,
        time: formatTime(),
      };
      setChat((prev) => [...prev, translatedMsg]);
    } catch (error) {
      console.error("Translation error:", error);
      const errorMsg: Message = {
        role: "ai",
        text: "❌ Translation service unavailable",
        time: formatTime(),
      };
      setChat((prev) => [...prev, errorMsg]);
    }
  };

  // ... (rest of the component remains the same)
  return (
    <>
      {showChat && (
        <div className="fixed bottom-20 right-4 bg-white shadow-xl rounded-lg w-80 max-h-[90vh] overflow-y-auto border border-gray-300 z-40 flex flex-col">
          <div className="p-4 border-b flex justify-between items-center">
            <h2 className="text-lg font-bold">💬 Financial AI Advisor</h2>
            <Select
              options={languageOptions}
              value={selectedLanguage}
              onChange={(option) => option && setSelectedLanguage(option)}
              className="w-32"
              classNamePrefix="select"
              isSearchable={false}
            />
          </div>

          <div className="p-3 text-sm space-y-2 flex-1 overflow-y-auto">
            {chat.map((msg, i) => (
              <div
                key={i}
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
                      <button onClick={() => copyToClipboard(msg.text)}>
                        <FaCopy size={12} />
                      </button>
                      <button onClick={() => translateMessage(msg.text)}>
                        <FaLanguage size={12} />
                      </button>
                    </div>
                  )}
                </div>
              </div>
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
          onClick={stopSpeaking}
          className="bg-gray-600 p-3 rounded-full text-white"
        >
          <FaStopCircle size={18} />
        </button>
      </div>
    </>
  );
};

export default AIChat;
