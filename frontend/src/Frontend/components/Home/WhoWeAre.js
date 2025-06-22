import React from "react";
import { Container, Row, Col } from "react-bootstrap";
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import {
  faChartLine,
  faShieldAlt,
  faLightbulb,
  faClock,
} from "@fortawesome/free-solid-svg-icons";
import "../styles/Home.css";
import AboutUs from "../../../assets/about.png";

const WhoWeAre = () => {
  const features = [
    {
      icon: faChartLine,
      title: "Data-Driven Insights",
      description:
        "Our AI analyzes financial markets and your personal data to provide actionable recommendations.",
    },
    {
      icon: faShieldAlt,
      title: "Bank-Level Security",
      description:
        "Your financial data is protected with the highest level of encryption and security protocols.",
    },
    {
      icon: faLightbulb,
      title: "Smart Recommendations",
      description:
        "Receive personalized advice on savings, investments, and budgeting tailored to your goals.",
    },
    {
      icon: faClock,
      title: "Real-Time Updates",
      description:
        "Get timely alerts and updates about your financial status and important market changes.",
    },
  ];

  return (
    <Container>
      <Row className="justify-content-center mb-5">
        <Col lg={8} className="text-center">
          <h2 className="mission_title">Who We Are</h2>
          <p className="lead mb-0">
            Financial expertise powered by artificial intelligence
          </p>
        </Col>
      </Row>

      <Row className="align-items-center mb-5">
        <Col lg={6} className="mb-4 mb-lg-0">
          <div className="about-image-container">
            <img
              src={AboutUs}
              alt="About Financial AI Advisor"
              className="img-fluid rounded"
            />
          </div>
        </Col>
        <Col lg={6}>
          <div className="about-content">
            <p className="mission_description">
              We are a team of financial experts and data scientists, united by
              a common goal: to provide everyone with access to intelligent
              financial advice. Our AI-based personal financial advisor is
              designed to make financial planning simple and accessible to all.
            </p>
            <p className="mission_description">
              By leveraging advanced machine learning algorithms and real-world
              financial expertise, we help users make informed decisions on
              budgeting, saving, and investing. Our platform is your trusted
              ally in achieving financial security.
            </p>
          </div>
        </Col>
      </Row>

      <Row className="mt-5">
        <Col lg={12}>
          <h3 className="text-center mb-4">
            Why Choose Our AI Financial Advisor
          </h3>
        </Col>

        {features.map((feature, index) => (
          <Col lg={3} md={6} className="mb-4" key={index}>
            <div className="feature-box text-center">
              <div className="feature-icon">
                <FontAwesomeIcon icon={feature.icon} />
              </div>
              <h4 className="feature-title">{feature.title}</h4>
              <p className="feature-description">{feature.description}</p>
            </div>
          </Col>
        ))}
      </Row>

      <Row className="stats-row text-center mt-5">
        <Col md={3} sm={6} className="mb-4">
          <div className="stat-item">
            <h2 className="stat-number">87%</h2>
            <p className="stat-label">Accuracy Rate</p>
          </div>
        </Col>
        <Col md={3} sm={6} className="mb-4">
          <div className="stat-item">
            <h2 className="stat-number">92%</h2>
            <p className="stat-label">Customer Growth</p>
          </div>
        </Col>
        <Col md={3} sm={6} className="mb-4">
          <div className="stat-item">
            <h2 className="stat-number">75%</h2>
            <p className="stat-label">Average Savings</p>
          </div>
        </Col>
        <Col md={3} sm={6} className="mb-4">
          <div className="stat-item">
            <h2 className="stat-number">95%</h2>
            <p className="stat-label">Customer Trust</p>
          </div>
        </Col>
      </Row>
    </Container>
  );
};

export default WhoWeAre;
