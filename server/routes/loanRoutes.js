import express from 'express';
const router = express.Router();

router.post('/calculate', (req, res) => {
  const { amount, rate, years } = req.body;
  const principal = parseFloat(amount);
  const interestRate = parseFloat(rate) / 100 / 12;
  const payments = parseInt(years) * 12;
  const x = Math.pow(1 + interestRate, payments);
  const monthly = (principal * x * interestRate) / (x - 1);

  res.json({ monthlyPayment: monthly.toFixed(2) });
});

export default router;
