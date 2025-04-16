import { useState, useEffect, useRef } from "react";
import { useAuthContext } from "../../hooks/useAuthContext";
import ReactMarkdown from "react-markdown";
import { FiMessageCircle } from "react-icons/fi";

const CHAT_APIS = [
  "http://localhost:4000/api/chat",
  "http://127.0.0.1:8000/api/chat",
];

const AIChat = () => {
  const { state } = useAuthContext();
  const { user } = state;

  const [input, setInput] = useState("");
  const [chat, setChat] = useState<
    { role: "user" | "ai"; text: string; time: string }[]
  >([]);
  const [loading, setLoading] = useState(false);
  const [showChat, setShowChat] = useState(false);
  const chatEndRef = useRef<HTMLDivElement | null>(null);

  const formatTime = () =>
    new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });

  // Scroll to bottom
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [chat]);

  // Load chat per user from localStorage
  useEffect(() => {
    const stored = localStorage.getItem(`chatHistory_${user?.id}`);
    if (stored) setChat(JSON.parse(stored));
  }, [user?.id]);

  // Save chat per user to localStorage
  useEffect(() => {
    if (user?.id) {
      localStorage.setItem(`chatHistory_${user.id}`, JSON.stringify(chat));
    }
  }, [chat, user?.id]);

  const sendMessage = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || !user?.id) return;

    const time = formatTime();
    const userMsg = { role: "user" as const, text: input, time };
    setChat((prev) => [...prev, userMsg]);
    setInput("");
    setLoading(true);

    try {
      let response;
      for (const api of CHAT_APIS) {
        response = await fetch(api, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            message: input,
            userId: user.id,
          }),
        });
        if (response.ok) break;
      }

      if (!response?.ok) throw new Error("All APIs failed");

      const data = await response.json();
      setChat((prev) => [
        ...prev,
        {
          role: "ai",
          text: `🤖 ${data.response || "No response"}`,
          time: formatTime(),
        },
      ]);
    } catch {
      setChat((prev) => [
        ...prev,
        {
          role: "ai",
          text: "❌ Unable to reach AI server.",
          time: formatTime(),
        },
      ]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <>
      {/* Toggle Button */}
      <button
        onClick={() => setShowChat(!showChat)}
        className="fixed bottom-4 right-4 bg-blue-600 hover:bg-blue-700 text-white p-3 rounded-full shadow-lg z-50"
      >
        <FiMessageCircle size={24} />
      </button>

      {/* Chat Box */}
      {showChat && (
        <div className="fixed bottom-20 right-4 bg-white shadow-xl rounded-lg w-80 max-h-[90vh] overflow-y-auto border border-gray-300 z-40 flex flex-col">
          <div className="p-4 border-b">
            <h2 className="text-lg font-bold">💬 Financial AI Advisor</h2>
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
                <span className="block text-right text-[10px] text-gray-400">
                  {msg.time}
                </span>
              </div>
            ))}
            <div ref={chatEndRef} />
          </div>

          {/* Chat Input */}
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
    </>
  );
};

export default AIChat;
