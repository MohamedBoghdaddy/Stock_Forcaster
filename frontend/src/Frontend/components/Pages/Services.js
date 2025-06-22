import React from "react";
import { Container, Row, Col } from "react-bootstrap";
import "../styles/Pages.css";
import aiImage from "../../../assets/img/ai-2.jpg";

const Services = () => {
  const services = [
    {
      id: 1,
      title: "Personal Finance Management",
      description:
        "Track your income, expenses, and savings goals with our intuitive dashboard. Our AI analyzes your spending patterns and suggests ways to optimize your budget.",
      icon: (
        <svg
          width="40"
          height="40"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
          strokeLinecap="round"
          strokeLinejoin="round"
        >
          <path d="M3 12h18M3 6h18M3 18h18" />
        </svg>
      ),
    },
    {
      id: 2,
      title: "Investment Advisory",
      description:
        "Receive personalized investment recommendations based on your financial goals, risk tolerance, and time horizon. Our AI keeps track of market trends and adjusts suggestions accordingly.",
      icon: (
        <svg
          width="40"
          height="40"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
          strokeLinecap="round"
          strokeLinejoin="round"
        >
          <polyline points="3 6 6 6 6 21" />
          <path d="M18 21V9a4 4 0 0 0-8 0v12" />
          <circle cx="18" cy="6" r="3" />
        </svg>
      ),
    },
    {
      id: 3,
      title: "Debt Management",
      description:
        "Develop a strategic plan to pay off debts efficiently. Our AI creates personalized debt reduction strategies to help you become debt-free faster.",
      icon: (
        <svg
          width="40"
          height="40"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
          strokeLinecap="round"
          strokeLinejoin="round"
        >
          <path d="M12 1v22" />
          <path d="M17 5H9.5a3.5 3.5 0 0 0 0 7H14a3.5 3.5 0 0 1 0 7H6" />
        </svg>
      ),
    },
    {
      id: 4,
      title: "Retirement Planning",
      description:
        "Plan for a secure retirement with personalized projections and savings strategies. Our AI helps you determine how much you need to save and the best ways to reach your retirement goals.",
      icon: (
        <svg
          width="40"
          height="40"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
          strokeLinecap="round"
          strokeLinejoin="round"
        >
          <circle cx="12" cy="12" r="10" />
          <path d="M12 6v6l4 2" />
        </svg>
      ),
    },
    {
      id: 5,
      title: "Financial Education",
      description:
        "Access a wealth of educational resources tailored to your financial situation and goals. Our AI recommends relevant articles, videos, and tutorials based on your needs.",
      icon: (
        <svg
          width="40"
          height="40"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
          strokeLinecap="round"
          strokeLinejoin="round"
        >
          <path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20" />
          <path d="M4 4v16" />
          <path d="M20 4H8a2 2 0 0 0-2 2v0a2 2 0 0 0 2 2h12" />
        </svg>
      ),
    },
    {
      id: 6,
      title: "Tax Optimization",
      description:
        "Maximize your tax efficiency with personalized suggestions for tax-saving strategies. Our AI keeps up with tax law changes and helps you plan accordingly.",
      icon: (
        <svg
          width="40"
          height="40"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
          strokeLinecap="round"
          strokeLinejoin="round"
        >
          <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
          <polyline points="7 10 12 15 17 10" />
          <line x1="12" y1="15" x2="12" y2="3" />
        </svg>
      ),
    },
  ];

  return (
    <div className="services-page">
      <section className="section hero-section">
        <Container>
          <Row className="align-items-center">
            <Col lg={12} className="text-center">
              <h1 className="section-title">Our Services</h1>
              <p className="lead">
                Comprehensive financial solutions powered by artificial
                intelligence
              </p>
            </Col>
          </Row>
        </Container>
      </section>

      <section className="section">
        <Container>
          <Row>
            {services.map((service) => (
              <Col lg={4} md={6} className="mb-4" key={service.id}>
                <div className="service-card card h-100">
                  <div className="card-body text-center">
                    <div className="mb-3">{service.icon}</div>
                    <h3>{service.title}</h3>
                    <p>{service.description}</p>
                  </div>
                </div>
              </Col>
            ))}
          </Row>
        </Container>
      </section>

      <section className="section bg-light">
        <Container>
          <Row className="align-items-center">
            <Col lg={6}>
              <h2 className="mb-4">How Our AI Works</h2>
              <p>
                Our advanced artificial intelligence platform analyzes your
                financial data, market trends, and economic indicators to
                provide personalized recommendations tailored to your unique
                situation.
              </p>
              <ul className="feature-list">
                <li>Data analysis from multiple financial sources</li>
                <li>Pattern recognition to identify spending habits</li>
                <li>Predictive modeling for investment opportunities</li>
                <li>Risk assessment based on your financial profile</li>
                <li>Continuous learning to improve recommendations</li>
              </ul>
            </Col>
            <Col lg={6} className="text-center">
              <div className="services-image ai-works-img">
                <img
                  src={aiImage}
                  alt="About our mission"
                  className="about-image"
                />
              </div>
            </Col>
          </Row>
        </Container>
      </section>

      <section className="section cta-section text-center">
        <Container>
          <h2 className="mb-4">Ready to Transform Your Financial Future?</h2>
          <p className="lead mb-4">
            Join thousands of users who have improved their financial well-being
            with our AI advisor.
          </p>
          <button className="btn btn-primary btn-lg">
            Start Your Free Trial
          </button>
        </Container>
      </section>
    </div>
  );
};

export default Services;
