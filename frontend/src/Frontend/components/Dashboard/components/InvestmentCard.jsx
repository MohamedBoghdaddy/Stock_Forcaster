import React from "react";
import { Card, Badge } from "react-bootstrap";
import PropTypes from "prop-types";

const InvestmentCard = ({ investment }) => {
  const { name, value, growth } = investment;
  const isPositive = growth >= 0;

  return (
    <Card className="investment-card h-100 shadow-sm">
      <Card.Body>
        <div className="d-flex justify-content-between align-items-center mb-3">
          <Card.Title className="mb-0">{name}</Card.Title>
          <Badge
            bg={isPositive ? "success" : "danger"}
            className="growth-badge"
          >
            {isPositive ? "↑" : "↓"} {Math.abs(growth)}%
          </Badge>
        </div>

        <div className="investment-details">
          <div className="d-flex justify-content-between align-items-center">
            <span className="text-muted">Current Value:</span>
            <span className="investment-value">${value.toLocaleString()}</span>
          </div>
        </div>

        <div className="performance-indicator mt-3">
          <div className="progress">
            <div
              className={`progress-bar ${
                isPositive ? "bg-success" : "bg-danger"
              }`}
              role="progressbar"
              style={{ width: `${Math.min(Math.abs(growth), 100)}%` }}
              aria-valuenow={Math.abs(growth)}
              aria-valuemin="0"
              aria-valuemax="100"
            >
              {Math.abs(growth)}%
            </div>
          </div>
        </div>
      </Card.Body>
    </Card>
  );
};

InvestmentCard.propTypes = {
  investment: PropTypes.shape({
    id: PropTypes.oneOfType([PropTypes.string, PropTypes.number]).isRequired,
    name: PropTypes.string.isRequired,
    value: PropTypes.number.isRequired,
    growth: PropTypes.number.isRequired,
  }).isRequired,
};

export default InvestmentCard;
