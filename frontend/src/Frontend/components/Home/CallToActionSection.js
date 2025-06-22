import React from "react";
import { useNavigate } from "react-router-dom";
import "../styles/Home.css";
import { Container, Row, Col, Button } from "react-bootstrap";

const CallToActionSection = () => {
  const navigate = useNavigate();

  const handleGetStartedClick = () => {
    navigate("/signup");
  };

  return (
    <div className="cta-section">
      <Container>
        <Row className="justify-content-center">
          <Col lg={8} md={10} className="text-center">
            <h2 className="cta-title">Transform Your Financial Future Today</h2>
            <p className="cta-description">
              Join thousands of users who have improved their financial
              well-being with our AI-powered advisor. Get personalized
              recommendations, real-time insights, and achieve your financial
              goals faster.
            </p>
            <Button className="cta-btn" onClick={handleGetStartedClick}>
              Start Your Free 14-Day Trial
            </Button>
            <p className="mt-3 text-light opacity-75">
              No credit card required. Cancel anytime.
            </p>
          </Col>
        </Row>
      </Container>
    </div>
  );
};

export default CallToActionSection;
