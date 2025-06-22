import React from "react";
import { Badge } from "react-bootstrap";
import PropTypes from "prop-types";
import { format } from "date-fns";

const TransactionList = ({ transactions }) => {
  return (
    <div className="transaction-list">
      {transactions.map((transaction) => (
        <div key={transaction.id} className="transaction-item">
          <div className="transaction-header">
            <span className="transaction-date">
              {format(new Date(transaction.date), "MMM dd, yyyy")}
            </span>
            <Badge
              bg={transaction.amount > 0 ? "success" : "danger"}
              className="amount-badge"
            >
              {transaction.amount > 0 ? "+" : ""}
              {transaction.amount.toLocaleString("en-US", {
                style: "currency",
                currency: "USD",
              })}
            </Badge>
          </div>
          <div className="transaction-details">
            <span className="description">{transaction.description}</span>
            <Badge pill bg="secondary" className="category">
              {transaction.category}
            </Badge>
          </div>
        </div>
      ))}
    </div>
  );
};

TransactionList.propTypes = {
  transactions: PropTypes.arrayOf(
    PropTypes.shape({
      id: PropTypes.oneOfType([PropTypes.string, PropTypes.number]).isRequired,
      date: PropTypes.string.isRequired,
      description: PropTypes.string.isRequired,
      amount: PropTypes.number.isRequired,
      category: PropTypes.string.isRequired,
    })
  ).isRequired,
};

export default TransactionList;
