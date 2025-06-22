import React, { useState } from 'react';
import "../../styles/FinanceTools.css";

const FinanceTools = () => {
  // Net Worth States
  const [cash, setCash] = useState('');
  const [investments, setInvestments] = useState('');
  const [realEstate, setRealEstate] = useState('');
  const [loans, setLoans] = useState('');
  const [creditCard, setCreditCard] = useState('');
  const [mortgage, setMortgage] = useState('');
  const [netWorth, setNetWorth] = useState(null);

  // Loan States
  const [amount, setAmount] = useState('');
  const [rate, setRate] = useState('');
  const [years, setYears] = useState('');
  const [monthlyPayment, setMonthlyPayment] = useState(null);

  const calculateNetWorth = () => {
    const totalAssets = parseFloat(cash || 0) + parseFloat(investments || 0) + parseFloat(realEstate || 0);
    const totalLiabilities = parseFloat(loans || 0) + parseFloat(creditCard || 0) + parseFloat(mortgage || 0);
    setNetWorth((totalAssets - totalLiabilities).toFixed(2));
  };

  const calculateLoan = () => {
    const principal = parseFloat(amount);
    const interestRate = parseFloat(rate) / 100 / 12;
    const payments = parseInt(years) * 12;
    const x = Math.pow(1 + interestRate, payments);
    const monthly = (principal * x * interestRate) / (x - 1);
    setMonthlyPayment(monthly.toFixed(2));
  };

  return (
    <div className="finance-tools">
      <h2>💼 Finance Tools</h2>

      <div className="tool-wrapper">
        {/* Net Worth */}
        <div className="tool net-worth">
          <h3>Net Worth Calculator</h3>
          <label>
            💵 Cash (Amount you currently hold in hand or in your bank account):
            <input type="number" value={cash} onChange={e => setCash(e.target.value)} />
          </label>
          <label>
            📈 Investments (Stocks, bonds, crypto, etc.):
            <input type="number" value={investments} onChange={e => setInvestments(e.target.value)} />
          </label>
          <label>
            🏠 Real Estate (Total estimated property value you own):
            <input type="number" value={realEstate} onChange={e => setRealEstate(e.target.value)} />
          </label>
          <label>
            💳 Loans (Outstanding personal/car/student loans):
            <input type="number" value={loans} onChange={e => setLoans(e.target.value)} />
          </label>
          <label>
            🧾 Credit Card (Total unpaid balance across cards):
            <input type="number" value={creditCard} onChange={e => setCreditCard(e.target.value)} />
          </label>
          <label>
            🏡 Mortgage (Remaining loan on real estate):
            <input type="number" value={mortgage} onChange={e => setMortgage(e.target.value)} />
          </label>
          <button onClick={calculateNetWorth}>Calculate Net Worth</button>
          {netWorth !== null && <p className="result">Net Worth: <strong>${netWorth}</strong></p>}
        </div>

        {/* Loan Calculator */}
        <div className="tool loan-calculator">
          <h3>Loan Calculator</h3>
          <label>
            💰 Loan Amount (How much you want to borrow):
            <input type="number" value={amount} onChange={e => setAmount(e.target.value)} />
          </label>
          <label>
            📊 Interest Rate (%) (Annual interest charged by the lender):
            <input type="number" value={rate} onChange={e => setRate(e.target.value)} />
          </label>
          <label>
            ⏳ Term (Years) (How many years to repay the loan):
            <input type="number" value={years} onChange={e => setYears(e.target.value)} />
          </label>
          <button onClick={calculateLoan}>Calculate Payment</button>
          {monthlyPayment && <p className="result">Monthly Payment: <strong>${monthlyPayment}</strong></p>}
        </div>
      </div>
    </div>
  );
};

export default FinanceTools;
