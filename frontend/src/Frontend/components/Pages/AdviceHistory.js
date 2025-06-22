import React, { useEffect, useState } from "react";
import axios from "axios";
import { useAuthContext } from "../../../context/AuthContext";

const AdviceHistory = () => {
  const { state } = useAuthContext();
  const { user } = state || {};
  const [history, setHistory] = useState([]);

  useEffect(() => {
    const fetchHistory = async () => {
      try {
        const res = await axios.get(
          "http://localhost:4000/api/advice/history",
          {
            headers: { Authorization: `Bearer ${user.token}` },
          }
        );
        setHistory(res.data);
      } catch (err) {
        console.error("❌ Failed to fetch history:", err);
      }
    };
    if (user?.token) fetchHistory();
  }, [user]);

  return (
    <div className="history-container">
      <h2>📜 My AI Advice History</h2>
      {history.map((entry, index) => (
        <div key={index} className="history-card">
          <p>
            <strong>Date:</strong> {new Date(entry.createdAt).toLocaleString()}
          </p>
          <p>
            <strong>Summary:</strong> {entry.tips.summary}
          </p>
          <ul>
            {entry.tips.advice.map((tip, i) => (
              <li key={i}>✅ {tip}</li>
            ))}
          </ul>
        </div>
      ))}
    </div>
  );
};

export default AdviceHistory;
