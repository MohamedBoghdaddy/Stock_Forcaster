import { useState, useEffect, useRef } from "react";
import { useAuthContext } from "../../../context/AuthContext"; // Import AuthContext for user ID

const API_URL =
  process.env.REACT_APP_API_URL || "http://localhost:4000/api/chat/chat";

const Chatbot = () => {
  const { user } = useAuthContext(); // Get logged-in user
  const [messages, setMessages] = useState([]); // ✅ Use local state
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const chatEndRef = useRef(null);

  // ✅ Load chat history from local storage on mount
  useEffect(() => {
    const savedChat = localStorage.getItem("chatHistory");
    if (savedChat) {
      setMessages(JSON.parse(savedChat)); // Restore messages from storage
    }
  }, []);

  // ✅ Save chat history to local storage when messages change
  useEffect(() => {
    localStorage.setItem("chatHistory", JSON.stringify(messages));
  }, [messages]);

  // ✅ Scroll to the latest message
  useEffect(() => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  // ✅ Handle sending messages
  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!input.trim()) return;

    console.log("🔍 Sending message:", input);
    console.log("🔍 User ID:", user?._id);

    const userMessage = { text: input, sender: "user" };
    setMessages((prevMessages) => [...prevMessages, userMessage]); // Add user message
    setInput("");
    setLoading(true);

    try {
      console.log("Sending Token:", user?.token); // DEBUGGING

      const response = await fetch(API_URL, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${user?.token}`,
        },
        body: JSON.stringify({
          message: input,
          userId: user?._id || null,
        }),
      });

      if (!response.ok) {
        throw new Error(`Error: ${response.status} ${response.statusText}`);
      }

      const data = await response.json();
      const botMessage = { text: data.response, sender: "bot" };

      setMessages((prevMessages) => [...prevMessages, botMessage]); // Add bot response
    } catch (error) {
      console.error("Chatbot API Error:", error);
      setMessages((prevMessages) => [
        ...prevMessages,
        {
          text: "❌ Error fetching response. Please try again.",
          sender: "bot",
        },
      ]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="chat-container">
      <div className="chat-messages">
        {messages.map((msg, index) => (
          <div key={index} className={`message ${msg.sender}-message`}>
            <span dangerouslySetInnerHTML={{ __html: msg.text }} />
          </div>
        ))}
        <div ref={chatEndRef} /> {/* Ensures auto-scroll */}
      </div>

      <form onSubmit={handleSubmit} className="chat-form">
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          className="chat-input"
          placeholder="Ask a financial question..."
          disabled={loading}
        />
        <button type="submit" className="chat-button" disabled={loading}>
          {loading ? "Thinking..." : "Send"}
        </button>
      </form>
    </div>
  );
};

export default Chatbot;
