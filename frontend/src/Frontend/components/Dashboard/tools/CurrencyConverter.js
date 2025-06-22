import React, { useState, useEffect } from "react";
import axios from "axios";
import "../../styles/CurrencyConverter.css";


const CurrencyConverter = () => {
  const [amount, setAmount] = useState(1);
  const [fromCurrency, setFromCurrency] = useState("USD");
  const [toCurrency, setToCurrency] = useState("EGP");
  const [convertedAmount, setConvertedAmount] = useState(null);
  const [currencyList, setCurrencyList] = useState([]);

  useEffect(() => {
    axios
      .get("http://localhost:4000/api/currency/symbols")
      .then((res) => {
        if (res.data.success) {
          setCurrencyList(res.data.symbols);
        } else {
          console.error("Failed to load symbols:", res.data.error);
        }
      })
      .catch((err) => {
        console.error("Error loading currency list:", err);
      });
  }, []);

  const convertCurrency = () => {
    axios
      .get("http://localhost:4000/api/currency/convert", {
        params: {
          from: fromCurrency,
          to: toCurrency,
          amount: amount,
        },
      })
      .then((res) => {
        if (res.data.success) {
          setConvertedAmount(res.data.result);
        } else {
          alert("Conversion failed. Try again.");
        }
      })
      .catch((err) => {
        console.error("Conversion error:", err);
        alert("Conversion error. Check console.");
      });
  };

  return (
    <div className="currency-converter" style={{ maxWidth: "500px", margin: "auto" }}>
      <h2>💱 Currency Converter</h2>

      <div style={{ marginBottom: "1rem" }}>
        <label>Amount:</label>
        <input
          type="number"
          value={amount}
          min="0"
          onChange={(e) => setAmount(e.target.value)}
          style={{ width: "100%", padding: "8px", marginBottom: "10px" }}
        />

        <label>From Currency:</label>
        <select
          value={fromCurrency}
          onChange={(e) => setFromCurrency(e.target.value)}
          style={{ width: "100%", padding: "8px", marginBottom: "10px" }}
        >
          {currencyList.map(({ code, description }) => (
            <option key={code} value={code}>
              {code} - {description}
            </option>
          ))}
        </select>

        <label>To Currency:</label>
        <select
          value={toCurrency}
          onChange={(e) => setToCurrency(e.target.value)}
          style={{ width: "100%", padding: "8px", marginBottom: "10px" }}
        >
          {currencyList.map(({ code, description }) => (
            <option key={code} value={code}>
              {code} - {description}
            </option>
          ))}
        </select>

        <button
          onClick={convertCurrency}
          style={{
            marginTop: "12px",
            padding: "10px 20px",
            backgroundColor: "#007bff",
            color: "white",
            border: "none",
            cursor: "pointer",
          }}
        >
          Convert
        </button>
      </div>

      {convertedAmount && (
        <p>
          <strong>Result:</strong> {convertedAmount}
        </p>
      )}
    </div>
  );
};

export default CurrencyConverter;