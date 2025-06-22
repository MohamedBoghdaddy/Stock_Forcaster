import React, { useState } from "react";
import { Tabs, Tab, Row, Col, Card, Button } from "react-bootstrap";
import InvestmentCard from "../components/InvestmentCard";
import SavingsProgress from "../components/charts/SavingsChart";
import TransactionList from "../components/TransactionList";
import ExpenseChart from "../components/charts/ExpenseChart"; // ✅ Add the chart import

const TabsView = ({
  handleGetAiAdvice,
  isLoading,
  dashboardData,
  investments,
  transactions,
}) => {
  const [activeKey, setActiveKey] = useState("overview");

  const savingsProgress =
    dashboardData && dashboardData.savingsGoal
      ? (dashboardData.currentSavings / dashboardData.savingsGoal) * 100
      : 0;

  return (
    <Tabs
      activeKey={activeKey}
      onSelect={(k) => setActiveKey(k)}
      className="mb-4"
      justify
    >
      {/* Overview Tab */}
      <Tab eventKey="overview" title="Overview">
        <div className="p-3">
          <h2 className="tab-title">Financial Overview</h2>

          {/* Account balance section */}
          <Row className="mb-4">{/* Optional: Add summary cards here */}</Row>

          {/* Savings progress */}
          <Row>
            <Col lg={6} className="mb-4">
              <SavingsProgress
                progress={savingsProgress}
                current={dashboardData.currentSavings}
                goal={dashboardData.savingsGoal}
              />
            </Col>
            {/* Add additional tips or visualizations if needed */}
          </Row>

          {/* Recent transactions */}
          <Row>
            <Col>
              <TransactionList
                transactions={dashboardData.recentTransactions.slice(0, 3)}
              />
            </Col>
          </Row>

          {/* ✅ Expense Chart */}
          <ExpenseChart
            data={{
              labels: ["Food", "Transport", "Bills", "Shopping", "Other"],
              values: [500, 300, 400, 250, 150],
              income: 5000,
              savings: 1500,
              totalExpenses: 1600,
            }}
            onAddExpense={() => console.log("Open Add Expense Modal")}
          />
        </div>
      </Tab>

      {/* Investments Tab */}
      <Tab eventKey="investments" title="Investments">
        <div className="p-3">
          <h2 className="tab-title">Investment Portfolio</h2>
          <Row>
            {investments.map((investment) => (
              <Col lg={4} md={6} className="mb-4" key={investment.id}>
                <InvestmentCard investment={investment} />
              </Col>
            ))}
          </Row>
        </div>
      </Tab>

      {/* Advice Tab */}
      <Tab eventKey="advice" title="AI Advice">
        <div className="p-3">
          <h2 className="tab-title">AI Financial Advice</h2>
          <Row>
            <Col lg={6} className="mb-4">
              <Card className="dashboard-card">
                <Card.Body>
                  <p>
                    Click below to get AI-powered financial recommendations.
                  </p>
                  <Button
                    onClick={handleGetAiAdvice}
                    disabled={isLoading}
                    variant="primary"
                  >
                    {isLoading ? "Analyzing..." : "Get AI Advice"}
                  </Button>
                </Card.Body>
              </Card>
            </Col>
          </Row>
        </div>
      </Tab>

      {/* Transactions Tab */}
      <Tab eventKey="transactions" title="Transactions">
        <div className="p-3">
          <h2 className="tab-title">Transaction History</h2>
          <Card className="dashboard-card">
            <Card.Body>
              <TransactionList transactions={transactions} />
            </Card.Body>
          </Card>
        </div>
      </Tab>
    </Tabs>
  );
};

export default TabsView;
