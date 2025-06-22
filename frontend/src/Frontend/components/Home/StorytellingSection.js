import React from "react";
import { Row, Col } from "react-bootstrap";
import "../styles/Home.css";
import MissionImage from "../../../assets/mission.jpg";

const StorytellingSection = () => {
  return (
    <>
      <Row className="justify-content-center mb-5">
        <Col lg={8} className="text-center">
          <h2 className="storytelling-title">Our Mission</h2>
          <p className="lead">
            Democratizing financial advice through artificial intelligence
          </p>
        </Col>
      </Row>

      <Row className="align-items-center">
        <Col lg={6} md={12} className="mb-4 mb-lg-0">
          <div className="storytelling-content">
            <p className="story-paragraph">
              We believe that everyone should have access to expert financial
              advice, no matter their background or financial status. That's why
              we built an AI-powered platform to provide personalized,
              data-driven insights to help you manage your finances smarter and
              more efficiently.
            </p>
            <p className="story-paragraph">
              Our team of data scientists and financial experts combined their
              knowledge to create a tool that understands your financial goals
              and delivers actionable advice—from budgeting to investment
              strategies.
            </p>
            <p className="story-paragraph">
              This journey is about more than just technology; it's about
              empowering you to take charge of your financial future with
              confidence.
            </p>
          </div>
        </Col>
        <Col lg={6} md={12}>
          <div className="mission-image-container">
            <img
              src={MissionImage}
              alt="Our Mission"
              className="img-fluid rounded mission-image"
            />
          </div>
        </Col>
      </Row>
    </>
  );
};

export default StorytellingSection;
