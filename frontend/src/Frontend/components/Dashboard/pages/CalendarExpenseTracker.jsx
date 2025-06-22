import React, { useState, useEffect } from "react";
import { useAuthContext } from "../../../../context/AuthContext";
import { format } from "date-fns";
import "../../styles/CalendarExpenseTracker.css";

const CalendarExpenseTracker = () => {
  const { state } = useAuthContext();
  const { user, loading } = state;
  const [date, setDate] = useState(format(new Date(), "yyyy-MM-dd"));
  const [form, setForm] = useState({ description: "", amount: "" });
  const [expenses, setExpenses] = useState([]);

  useEffect(() => {
    if (loading || !user || !user._id) return;

    fetch(`http://localhost:4000/api/expenses?userId=${user._id}`)
      .then((res) => res.json())
      .then((data) => setExpenses(data))
      .catch((err) => console.error("❌ Fetch failed:", err));
  }, [loading, user?._id]);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (loading || !user?._id) return;

    const newExpense = {
      userId: user._id,
      description: form.description,
      amount: parseFloat(form.amount),
      date,
    };

    try {
      const res = await fetch("http://localhost:4000/api/expenses", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(newExpense),
      });

      const saved = await res.json();
      setExpenses((prev) => [saved, ...prev]);
      setForm({ description: "", amount: "" });
    } catch (err) {
      console.error("❌ Expense submission error:", err);
    }
  };

  const handleDelete = async (id) => {
    try {
      await fetch(`http://localhost:4000/api/expenses/${id}`, { method: "DELETE" });
      setExpenses((prev) => prev.filter((e) => e._id !== id));
    } catch (err) {
      console.error("❌ Delete failed:", err);
    }
  };

  const expensesForDate = expenses.filter(
    (e) => format(new Date(e.date), "yyyy-MM-dd") === date
  );

  if (loading) return <p>⏳ Loading auth...</p>;

  return (
    <div className="calendar-expense-container">
      <h2>📅 Expense Tracker</h2>

      <div className="date-picker">
        <label>Select Date: </label>
        <input type="date" value={date} onChange={(e) => setDate(e.target.value)} />
      </div>

      {user ? (
        <>
          <form onSubmit={handleSubmit} className="expense-form">
            <input
              type="text"
              placeholder="Description"
              value={form.description}
              onChange={(e) => setForm({ ...form, description: e.target.value })}
            />
            <input
              type="number"
              placeholder="Amount"
              value={form.amount}
              onChange={(e) => setForm({ ...form, amount: e.target.value })}
            />
            <button type="submit">➕ Add</button>
          </form>

          <ul className="expense-list">
            {expensesForDate.length === 0 ? (
              <li>No expenses for this date</li>
            ) : (
              expensesForDate.map((e) => (
                <li key={e._id}>
                  💸 {e.description} - ${e.amount.toFixed(2)}
                  <button onClick={() => handleDelete(e._id)}>🗑️</button>
                </li>
              ))
            )}
          </ul>
        </>
      ) : (
        <p>🔐 Please log in to track your expenses</p>
      )}
    </div>
  );
};

export default CalendarExpenseTracker;
