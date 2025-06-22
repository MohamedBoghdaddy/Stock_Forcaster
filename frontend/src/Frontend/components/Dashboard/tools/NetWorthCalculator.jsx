// import React, { useState } from 'react';
// import '../../styles/FinanceTools.css';

// const NetWorthCalculator = () => {
//   const [cash, setCash] = useState('');
//   const [investments, setInvestments] = useState('');
//   const [realEstate, setRealEstate] = useState('');
//   const [loans, setLoans] = useState('');
//   const [creditCard, setCreditCard] = useState('');
//   const [mortgage, setMortgage] = useState('');
//   const [netWorth, setNetWorth] = useState(null);

//   const calculateNetWorth = () => {
//     const totalAssets =
//       parseFloat(cash || 0) +
//       parseFloat(investments || 0) +
//       parseFloat(realEstate || 0);

//     const totalLiabilities =
//       parseFloat(loans || 0) +
//       parseFloat(creditCard || 0) +
//       parseFloat(mortgage || 0);

//     const result = totalAssets - totalLiabilities;
//     setNetWorth(result.toFixed(2));
//   };

//   return (
//     <div className="net-worth-calculator">
//       <h2>Net Worth Calculator</h2>

//       <label>
//         💵 Cash in hand / bank account:
//         <input type="number" value={cash} onChange={e => setCash(e.target.value)} />
//       </label>
//       <label>
//         📈 Total value of investments (stocks, crypto, etc.):
//         <input type="number" value={investments} onChange={e => setInvestments(e.target.value)} />
//       </label>
//       <label>
//         🏠 Estimated value of real estate you own:
//         <input type="number" value={realEstate} onChange={e => setRealEstate(e.target.value)} />
//       </label>

//       <hr />

//       <label>
//         💳 Outstanding loan balances (personal/car/student loans):
//         <input type="number" value={loans} onChange={e => setLoans(e.target.value)} />
//       </label>
//       <label>
//         🧾 Unpaid credit card balance:
//         <input type="number" value={creditCard} onChange={e => setCreditCard(e.target.value)} />
//       </label>
//       <label>
//         🏡 Remaining mortgage on home(s):
//         <input type="number" value={mortgage} onChange={e => setMortgage(e.target.value)} />
//       </label>

//       <button onClick={calculateNetWorth}>Calculate Net Worth</button>

//       {netWorth !== null && (
//         <div className="result">
//           Your Net Worth is: <strong>${netWorth}</strong>
//         </div>
//       )}
//     </div>
//   );
// };

// export default NetWorthCalculator;
