import React from "react";
import { Container, Row, Col, Button } from "react-bootstrap";
import { Link } from "react-router-dom";
import "../styles/Home.css";
// import logoblack from "../assets/images/logo-black.png";
import HeroImage from "../../../assets/hero.jpg";

const HeroSection = () => {
  return (
    <Container>
      <Row className="align-items-center">
        <Col lg={6} className="order-lg-2">
          <div className="hero-image-container">
            <img
              className="hero-image"
              src={HeroImage}
              alt="Financial Advisor AI"
            />
          </div>
        </Col>
        <Col lg={6} className="order-lg-1 mb-5 mb-lg-0">
          <div className="hero-content">
            <h1 className="hero_title">Your Intelligent Financial Advisor</h1>
            <p className="hero_description">
              Get personalized financial guidance powered by AI. Budget smarter,
              invest wisely, and achieve your financial goals with intelligent
              recommendations tailored specifically to your needs.
            </p>
            <div className="hero-buttons">
              <Button
                variant="dark"
                as={Link}
                to="/signup"
                className="btn-primary me-3"
              >
                Start Free Trial
              </Button>
              <Link to="/services" className="btn btn-secondary">
                Learn More
              </Link>
            </div>
          </div>
        </Col>
      </Row>
    </Container>
  );
};

export default HeroSection;
